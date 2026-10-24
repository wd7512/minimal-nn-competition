"""Train a PPO lunar lander policy and write the submission files."""

import argparse
import json
import logging
import random
import statistics
from pathlib import Path

from minimal_nn_competition.benchmarks.lunarlander import engine
from minimal_nn_competition.examples.ppo import train_ppo

HIDDEN_SIZES = (64, 64)
INPUT_SIZE = 8
OUTPUT_SIZE = 4
MAX_STEPS = 300


class LunarLanderEnvironment:
    """Adapter exposing the engine through the shared PPO interface."""

    name = "lunarlander"
    input_size = INPUT_SIZE
    output_size = OUTPUT_SIZE
    max_steps = MAX_STEPS

    def __init__(self):
        self.rng = random.Random(0)

    def seed(self, seed):
        self.rng = random.Random(seed)

    def reset(self):
        return engine.reset(self.rng)

    def observation(self, state):
        return engine.observation(state)

    def step(self, state, action):
        return engine.step(state, action)


def evaluate(policy, seeds, device, max_steps=MAX_STEPS):
    """Return greedy episode returns and landing flags for the given seeds."""
    import torch

    returns = []
    landings = []
    with torch.no_grad():
        for seed in seeds:
            state = engine.reset(random.Random(seed))
            total = 0.0
            landed = False
            for _ in range(max_steps):
                features = torch.tensor(
                    engine.observation(state), dtype=torch.float32, device=device
                )
                action = int(torch.argmax(policy(features[None]), dim=1).item())
                state, reward, done = engine.step(state, action)
                total += reward
                if done:
                    landed = state[1] <= 0.0 and engine.landed(state)
                    break
            returns.append(total)
            landings.append(landed)
    return returns, landings


def train(
    output_dir,
    iterations=1500,
    episodes_per_iteration=8,
    max_steps=MAX_STEPS,
    hidden_sizes=HIDDEN_SIZES,
    device_name="auto",
    seed=0,
    name=None,
    author="unknown",
):
    """Train the example LunarLander policy with PPO and write the submission files."""
    import torch

    logger = logging.getLogger(__name__)
    environment = LunarLanderEnvironment()
    environment.max_steps = max_steps
    policy = train_ppo(
        environment,
        iterations=iterations,
        episodes_per_iteration=episodes_per_iteration,
        hidden_sizes=hidden_sizes,
        device_name=device_name,
        seed=seed,
    )
    device = next(policy.parameters()).device
    returns, landings = evaluate(policy, range(100), device, max_steps)
    logger.info(
        "Greedy evaluation: mean %.2f, std %.2f, landed %d/100",
        statistics.fmean(returns),
        statistics.pstdev(returns),
        sum(landings),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    state_dict = {key: value.cpu() for key, value in policy.state_dict().items()}
    torch.save(state_dict, output_dir / "model.pt")
    config = {
        "benchmark": "lunarlander",
        "name": name or output_dir.name,
        "author": author,
        "architecture": "mlp",
        "input_size": INPUT_SIZE,
        "output_size": OUTPUT_SIZE,
        "hidden_sizes": list(hidden_sizes),
        "activation": "relu",
        "description": "A small PPO-trained lunar lander policy.",
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    logger.info("Wrote %s", output_dir / "model.pt")
    return state_dict


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("models/lunarlander/example"))
    parser.add_argument("--iterations", type=int, default=1500)
    parser.add_argument("--episodes-per-iteration", type=int, default=8)
    parser.add_argument("--device", default="auto", choices=("auto", "mps", "cpu"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", default=None, help="Display name")
    parser.add_argument("--author", default="unknown", help="Submission author")
    args = parser.parse_args(argv)
    train(
        args.output,
        iterations=args.iterations,
        episodes_per_iteration=args.episodes_per_iteration,
        device_name=args.device,
        seed=args.seed,
        name=args.name,
        author=args.author,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
