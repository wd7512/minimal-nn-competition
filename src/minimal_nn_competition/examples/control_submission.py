"""Train native control policies and save strictly validated state-dict submissions."""

import json
import logging

from minimal_nn_competition.benchmarks.common.control import TASKS, environment
from minimal_nn_competition.examples.ppo import train_ppo


def train(
    benchmark,
    output_dir,
    iterations=500,
    episodes_per_iteration=8,
    device_name="auto",
    seed=0,
    name=None,
    author="unknown",
):
    import torch

    task = TASKS[benchmark]
    if benchmark == "cartpole-gym-v1":
        policy = torch.nn.Linear(4, 2)
        with torch.no_grad():
            policy.weight.copy_(torch.tensor([[-0.05, -0.25, -1, -0.5], [0.05, 0.25, 1, 0.5]]))
            policy.bias.zero_()
        architecture, hidden = "linear", []
    else:
        with environment(benchmark) as env:
            policy = train_ppo(
                env,
                iterations,
                episodes_per_iteration=episodes_per_iteration,
                device_name=device_name,
                seed=seed,
                reset_options=task["reset"],
            )
        architecture, hidden = "mlp", [64, 64]
    output_dir.mkdir(parents=True, exist_ok=True)
    config = dict(
        benchmark=benchmark,
        name=name or output_dir.name,
        author=author,
        architecture=architecture,
        input_size=task["inputs"],
        output_size=task["outputs"],
        hidden_sizes=hidden,
        activation="relu",
        description="Native Gymnasium reference policy.",
    )
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    state_dict = {key: value.cpu() for key, value in policy.state_dict().items()}
    torch.save(state_dict, output_dir / "model.pt")
    logging.info("Wrote %s", output_dir / "model.pt")
    return state_dict
