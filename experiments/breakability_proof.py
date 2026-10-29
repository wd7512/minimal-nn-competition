"""Temporary research experiment: generate compact policies and run official benchmark."""
import argparse
import json
import random
import subprocess
import sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from minimal_nn_competition.benchmarks.registry import create_benchmark
from minimal_nn_competition.benchmarks.common.control import TASKS, action, environment
from minimal_nn_competition.model.entry import LoadedSubmission, Submission, SubmissionConfig, materialize_submission

torch.set_num_threads(1)
random.seed(7512)
np.random.seed(7512)
torch.manual_seed(7512)
BASE = {"cartpole": ("cartpole-gym-v1", 1.0, 10),
        "pendulum": ("pendulum-gym-v1", 1.0, 4481),
        "snake": ("snake-obstacles-8x8", 0.19, 145156),
        "lunarlander": ("lunarlander-gym-v1", 0.95, 4996)}

def make_mlp(nin, hidden, nout):
    layers = []
    dims = [nin, *hidden, nout]
    for i, (a, b) in enumerate(zip(dims, dims[1:])):
        layers.append(nn.Linear(a, b))
        if i < len(dims)-2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)

def evaluate(benchmark, model, config, seeds=range(100)):
    result = create_benchmark(benchmark, seeds=tuple(seeds)).evaluate(
        LoadedSubmission(config, model.state_dict(), model.eval()))
    return float(result[0]), result[2]

def save(bench, model, root, name, hidden, indices=None):
    dest = root / bench / name
    dest.mkdir(parents=True, exist_ok=True)
    linear = [m for m in model.modules() if isinstance(m, nn.Linear)]
    config = dict(benchmark=bench, name=name, author="GPT-6 experimental proof",
                  architecture="linear" if isinstance(model, nn.Linear) else "mlp",
                  input_size=linear[0].in_features, output_size=linear[-1].out_features,
                  hidden_sizes=hidden, activation="relu",
                  description="Research experiment; do not promote before review")
    if indices is not None:
        config["input_indices"] = list(indices)
    (dest/"config.json").write_text(json.dumps(config, indent=2)+"\n")
    torch.save(model.state_dict(), dest/"model.pt")
    return dest, SubmissionConfig.model_validate(config)

def validate_official(dest):
    p = subprocess.run([sys.executable, "-m", "minimal_nn_competition",
                        "validate-submission", str(dest)],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       text=True, timeout=610)
    print("OFFICIAL_VALIDATION", dest, "EXIT", p.returncode, p.stdout[-4500:], flush=True)
    return dict(exit_code=p.returncode, output=p.stdout[-4500:])

def cartpole(teacher):
    old = teacher.model
    w, b = old.weight.detach(), old.bias.detach()
    student = make_mlp(4, [1], 2)
    with torch.no_grad():
        student[0].weight.copy_((w[1]-w[0])[None])
        student[0].bias.copy_((b[1]-b[0])[None])
        student[2].weight.copy_(torch.tensor([[0.0],[1.0]]))
        student[2].bias.zero_()
    return [("cartpole-exact-9p", student, [1])]

def pendulum(teacher):
    key = "pendulum-gym-v1"
    samples = []
    with environment(key) as env, torch.no_grad():
        for seed in range(100, 180):
            obs, _ = env.reset(seed=seed, options=TASKS[key]["reset"])
            for _ in range(200):
                samples.append(obs.copy())
                ctrl = teacher.model(torch.from_numpy(obs.copy())[None])[0].numpy()
                obs, _, term, trunc, _ = env.step(action(key, ctrl))
                if term or trunc: break
    angles = np.random.uniform(-np.pi, np.pi, 16000)
    vel = np.random.uniform(-8, 8, 16000)
    random_obs = np.stack([np.cos(angles), np.sin(angles), vel], axis=1).astype(np.float32)
    X = torch.from_numpy(np.concatenate([np.asarray(samples), random_obs]))
    with torch.no_grad():
        Y = teacher.model(X).clip(-2,2)
    print("PENDULUM_TRAIN_POINTS",len(X),flush=True)
    candidates = []
    for width in [8,16,32]:
        torch.manual_seed(123+width)
        net = make_mlp(3, [width,width], 1)
        optim = torch.optim.Adam(net.parameters(), lr=0.003)
        for _ in range(1800):
            idx = torch.randint(len(X),(512,))
            loss = nn.functional.mse_loss(net(X[idx]),Y[idx])
            optim.zero_grad(); loss.backward(); optim.step()
        print("PENDULUM_STUDENT",width,"MSE",nn.functional.mse_loss(net(X),Y).item(),flush=True)
        candidates.append((f"pendulum-distill-{width}",net.eval(),[width,width]))
    return candidates

def lunarlander(teacher):
    key = "lunarlander-gym-v1"
    states, logits = [], []
    with environment(key) as env, torch.no_grad():
        for seed in range(100,300):
            obs,_ = env.reset(seed=seed,options=TASKS[key]["reset"])
            for _ in range(1000):
                output = teacher.model(torch.from_numpy(obs.copy())[None])[0].numpy()
                states.append(obs.copy()); logits.append(output.copy())
                obs,_,term,trunc,_ = env.step(action(key,output))
                if term or trunc: break
    X = torch.from_numpy(np.asarray(states))
    Y = torch.from_numpy(np.asarray(logits))
    Y -= Y.mean(dim=1,keepdim=True)
    print("LANDER_TRAIN_POINTS",len(X),flush=True)
    candidates = []
    for width in [32,48,56]:
        torch.manual_seed(321+width)
        net = make_mlp(8,[width,width],4)
        optim = torch.optim.Adam(net.parameters(),lr=0.0015)
        for _ in range(2400):
            ids = torch.randint(len(X),(512,))
            pred = net(X[ids]); pred = pred-pred.mean(dim=1,keepdim=True)
            loss = nn.functional.mse_loss(pred,Y[ids])+0.2*nn.functional.cross_entropy(pred,Y[ids].argmax(1))
            optim.zero_grad(); loss.backward(); optim.step()
        with torch.no_grad():
            agreement = (net(X).argmax(1)==Y.argmax(1)).float().mean().item()
        print("LANDER_STUDENT",width,"action_match",agreement,flush=True)
        candidates.append((f"lunarlander-distill-{width}",net.eval(),[width,width]))
    return candidates

def snake(teacher):
    original = teacher.model
    state = original.state_dict()
    last = list(state)[-1]
    assert last.endswith("bias")
    initial = state[last].clone()
    winners = []
    for i in range(48):
        scale = [0.01,0.03,0.1,0.3][i//12]
        delta = torch.randn_like(initial)*scale
        with torch.no_grad():
            state[last].copy_(initial+delta)
        train_score,_ = evaluate("snake-obstacles-8x8",original,teacher.config,seeds=range(100,140))
        print("SNAKE_SEARCH",i,scale,train_score,flush=True)
        winners.append((train_score,delta.clone()))
    winners.sort(key=lambda v:v[0],reverse=True)
    import copy
    output = []
    for i,(train_score,delta) in enumerate(winners[:8]):
        candidate=copy.deepcopy(original)
        with torch.no_grad():
            candidate.state_dict()[last].copy_(initial+delta)
        print("SNAKE_FINALIST",i,train_score,flush=True)
        output.append((f"snake-bias-search-{i}",candidate,list(teacher.config.hidden_sizes)))
    return output

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--task",required=True,choices=list(BASE))
    parser.add_argument("--output",type=Path,default=Path("/tmp/breakability-proof"))
    args=parser.parse_args()
    task=args.task
    benchmark, incumbent, old_params=BASE[task]
    baseline_path=Path("models")/benchmark/("sl19-512-256" if task=="snake" else "example")
    teacher=materialize_submission(Submission.from_directory(baseline_path))
    root=args.output;root.mkdir(parents=True,exist_ok=True)
    candidates=globals()[task](teacher)
    rows=[]
    for name,model,hidden in candidates:
        dest,config=save(benchmark,model,root,name,hidden,
                         teacher.config.input_indices if task=="snake" else None)
        try:
            value,detail=evaluate(benchmark,model,config)
            params=sum(v.numel() for v in model.parameters())
            win=(value>incumbent+1e-9 or (abs(value-incumbent)<1e-9 and params<old_params))
            rows.append(dict(name=name,score=value,parameters=params,beats_incumbent=win,
                             directory=str(dest),mean_return=detail.get("mean_return"),
                             seed_success=detail.get("seed_success"),
                             seed_passed=detail.get("seed_passed")))
            print("CANDIDATE",name,"score",value,"params",params,"WIN",win,flush=True)
        except Exception as exc:
            rows.append(dict(name=name,error=repr(exc),directory=str(dest)))
            print("EVAL_ERROR",name,repr(exc),flush=True)
    good=sorted((r for r in rows if "score" in r),key=lambda v:(-v["score"],v["parameters"]))
    if good:
        good[0]["official_validation"]=validate_official(Path(good[0]["directory"]))
    report=dict(task=task,benchmark=benchmark,incumbent_score=incumbent,
                incumbent_parameters=old_params,results=rows,
                best=good[0] if good else None,training_seeds="100-299",
                evaluation_seeds="0-99",source="real pinned repository Gymnasium/Box2D and Snake")
    (root/f"{task}-report.json").write_text(json.dumps(report,indent=2))
    print("FINAL_REPORT",json.dumps(report)[:14000],flush=True)

if __name__=="__main__":
    main()
