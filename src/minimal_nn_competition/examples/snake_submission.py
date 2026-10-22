"""load the eval function for the mnist benchmark
and train a NN that passes the benchmark with PPO"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any

HIDDEN_SIZES = (64, 64)
BOARD_SIZE = 8
INPUT_SIZE = 7
OUTPUT_SIZE = 4


def create_model() -> Any:
    """Return the MLP policy shape used by the example submission."""
    from torch import nn

    return nn.Sequential(
        nn.Linear(INPUT_SIZE, HIDDEN_SIZES[0]),
        nn.ReLU(),
        nn.Linear(HIDDEN_SIZES[0], HIDDEN_SIZES[1]),
        nn.ReLU(),
        nn.Linear(HIDDEN_SIZES[1], OUTPUT_SIZE),
    )


def _select_device(requested: str) -> Any:
    import torch

    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _distance(observation: tuple[int, ...]) -> int:
    return abs(observation[0] - observation[2]) + abs(observation[1] - observation[3])


def _shape_reward(
    observation: tuple[int, ...], next_observation: tuple[int, ...], reward: float
) -> float:
    shaped = reward + 0.05 * (_distance(observation) - _distance(next_observation)) - 0.001
    if reward > 0:
        shaped += 0.5
    elif reward < 0:
        shaped -= 0.5
    return shaped


def _gae(
    rewards: list[float], values: list[float], dones: list[bool], gamma: float, lam: float
) -> tuple[list[float], list[float]]:
    advantages = [0.0] * len(rewards)
    gae = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        next_value = values[index + 1] if index + 1 < len(values) else 0.0
        nonterminal = 0.0 if dones[index] else 1.0
        delta = rewards[index] + gamma * next_value * nonterminal - values[index]
        gae = delta + gamma * lam * nonterminal * gae
        advantages[index] = gae
    returns = [advantage + value for advantage, value in zip(advantages, values, strict=True)]
    return advantages, returns


def _evaluate(policy: Any, device: Any, seeds: tuple[int, ...]) -> list[int]:
    import torch

    from minimal_nn_competition.benchmarks.snake.environment import SnakeEnvironment

    lengths = []
    with torch.no_grad():
        for seed in seeds:
            environment = SnakeEnvironment(seed=seed, max_moves=4096)
            observation = environment.reset()
            done = False
            while not done:
                state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
                action = int(torch.argmax(policy(state), dim=1).item())
                observation, _, done = environment.step(action)
            lengths.append(len(environment.snake))
    return lengths


def train(
    output_dir: Path,
    iterations: int = 600,
    episodes_per_iteration: int = 4,
    max_moves: int = 256,
    update_epochs: int = 4,
    batch_size: int = 64,
    device_name: str = "auto",
    seed: int = 0,
) -> dict[str, Any]:
    """Train the example Snake policy with PPO and write the submission files."""
    import torch
    from torch import nn
    from torch.distributions import Categorical

    from minimal_nn_competition.benchmarks.snake.environment import SnakeEnvironment

    logger = logging.getLogger(__name__)
    torch.manual_seed(seed)
    random.seed(seed)
    device = _select_device(device_name)
    logger.info("Training Snake example on %s", device)

    policy = create_model().to(device)
    value_net = nn.Sequential(
        nn.Linear(INPUT_SIZE, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    ).to(device)
    optimizer = torch.optim.Adam(list(policy.parameters()) + list(value_net.parameters()), lr=3e-4)
    gamma = 0.99
    lam = 0.95
    clip = 0.2
    entropy_coef = 0.01
    value_coef = 0.5

    for iteration in range(iterations):
        states = []
        actions = []
        old_log_probs = []
        returns_all = []
        advantages_all = []
        episode_lengths = []
        episode_food = []
        for episode in range(episodes_per_iteration):
            environment = SnakeEnvironment(
                seed=seed + iteration * episodes_per_iteration + episode,
                max_moves=max_moves,
            )
            observation = environment.reset()
            episode_rewards = []
            episode_values = []
            episode_dones = []
            for _ in range(max_moves):
                state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    distribution = Categorical(logits=policy(state))
                    action = int(distribution.sample().item())
                    log_prob = distribution.log_prob(torch.tensor(action, device=device))
                    value = float(value_net(state).item())
                next_observation, reward, done = environment.step(action)
                states.append(state.squeeze(0))
                actions.append(action)
                old_log_probs.append(float(log_prob.item()))
                episode_rewards.append(_shape_reward(observation, next_observation, reward))
                episode_values.append(value)
                episode_dones.append(done)
                observation = next_observation
                if done:
                    break
            advantages, returns = _gae(episode_rewards, episode_values, episode_dones, gamma, lam)
            returns_all.extend(returns)
            advantages_all.extend(advantages)
            episode_lengths.append(len(environment.snake))
            episode_food.append(len(environment.snake) - 1)

        states_t = torch.stack(states).to(device)
        actions_t = torch.tensor(actions, dtype=torch.int64, device=device)
        old_log_probs_t = torch.tensor(old_log_probs, dtype=torch.float32, device=device)
        advantages_t = torch.tensor(advantages_all, dtype=torch.float32, device=device)
        returns_t = torch.tensor(returns_all, dtype=torch.float32, device=device)
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        for _ in range(update_epochs):
            order = torch.randperm(len(states_t), device=device)
            for start in range(0, len(states_t), batch_size):
                index = order[start : start + batch_size]
                distribution = Categorical(logits=policy(states_t[index]))
                new_log_probs = distribution.log_prob(actions_t[index])
                ratio = torch.exp(new_log_probs - old_log_probs_t[index])
                clipped = torch.clamp(ratio, 1.0 - clip, 1.0 + clip)
                policy_loss = -torch.min(
                    ratio * advantages_t[index], clipped * advantages_t[index]
                ).mean()
                value_loss = nn.functional.mse_loss(
                    value_net(states_t[index]).squeeze(-1), returns_t[index]
                )
                loss = (
                    policy_loss
                    + value_coef * value_loss
                    - entropy_coef * distribution.entropy().mean()
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(policy.parameters()) + list(value_net.parameters()), 0.5
                )
                optimizer.step()

        if (iteration + 1) % 5 == 0 or iteration == 0:
            logger.info(
                "iteration %d/%d length %.1f food %.2f",
                iteration + 1,
                iterations,
                sum(episode_lengths) / len(episode_lengths),
                sum(episode_food) / len(episode_food),
            )

    lengths = _evaluate(policy, device, (0, 1, 2))
    logger.info("Greedy evaluation lengths: %s", lengths)

    output_dir.mkdir(parents=True, exist_ok=True)
    state_dict = {key: value.detach().cpu() for key, value in policy.state_dict().items()}
    torch.save(state_dict, output_dir / "model.pt")
    config = {
        "benchmark": "snake",
        "architecture": "mlp",
        "input_size": INPUT_SIZE,
        "output_size": OUTPUT_SIZE,
        "hidden_sizes": list(HIDDEN_SIZES),
        "activation": "relu",
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    logger.info("Wrote %s", output_dir / "model.pt")
    return state_dict


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("models/snake/example"))
    parser.add_argument("--iterations", type=int, default=600)
    parser.add_argument("--device", default="auto", choices=("auto", "mps", "cpu"))
    args = parser.parse_args(argv)
    train(args.output, iterations=args.iterations, device_name=args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
