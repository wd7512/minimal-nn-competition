"""Train a PPO pendulum policy and write the submission files."""

import argparse
import json
import logging
import random
import statistics
from pathlib import Path

from minimal_nn_competition.benchmarks import pendulum as engine
from minimal_nn_competition.examples.ppo import train_ppo

HIDDEN_SIZES = (64, 64)
INPUT_SIZE = 3
OUTPUT_SIZE = 5
HORIZON = 200


class PendulumEnvironment:
    """Adapter exposing the engine through the shared PPO interface."""

    name = "pendulum"
    input_size = INPUT_SIZE
    output_size = OUTPUT_SIZE
    max_steps = HORIZON

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


def evaluate(policy, seeds, device, horizon=HORIZON):
    """Return greedy episode returns for the given seeds."""
    import torch

    returns = []
    with torch.no_grad():
        for seed in seeds:
            state = engine.reset(random.Random(seed))
            total = 0.0
            for _ in range(horizon):
                features = torch.tensor(
                    engine.observation(state), dtype=torch.float32, device=device
                )
                action = int(torch.argmax(policy(features[None]), dim=1).item())
                state, reward, _ = engine.step(state, action)
                total += reward
            returns.append(total)
    return returns


def train(
    output_dir,
    iterations=300,
    episodes_per_iteration=8,
    horizon=HORIZON,
    hidden_sizes=HIDDEN_SIZES,
    device_name="auto",
    seed=0,
    name=None,
    author="unknown",
):
    """Train the example Pendulum policy with PPO and write the submission files."""
    import torch

    logger = logging.getLogger(__name__)
    environment = PendulumEnvironment()
    environment.max_steps = horizon
    policy = train_ppo(
        environment,
        iterations=iterations,
        episodes_per_iteration=episodes_per_iteration,
        hidden_sizes=hidden_sizes,
        device_name=device_name,
        seed=seed,
    )
    device = next(policy.parameters()).device
    returns = evaluate(policy, range(100), device, horizon)
    logger.info(
        "Greedy evaluation: mean %.2f, std %.2f",
        statistics.fmean(returns),
        statistics.pstdev(returns),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    state_dict = {key: value.cpu() for key, value in policy.state_dict().items()}
    torch.save(state_dict, output_dir / "model.pt")
    config = {
        "benchmark": "pendulum",
        "name": name or output_dir.name,
        "author": author,
        "architecture": "mlp",
        "input_size": INPUT_SIZE,
        "output_size": OUTPUT_SIZE,
        "hidden_sizes": list(hidden_sizes),
        "activation": "relu",
        "description": "A small PPO-trained discrete-torque pendulum policy.",
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    logger.info("Wrote %s", output_dir / "model.pt")
    return state_dict


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("models/pendulum/example"))
    parser.add_argument("--iterations", type=int, default=300)
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
