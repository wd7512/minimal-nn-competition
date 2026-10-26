"""load the eval function for the mnist benchmark
and train a NN that passes the benchmark with PPO"""

import argparse
import json
import logging
import random
import statistics
from pathlib import Path

HIDDEN_SIZES = (64, 64)
INPUT_SIZE = 24
OUTPUT_SIZE = 4


def create_model(output_size=OUTPUT_SIZE):
    """Return the MLP policy shape used by the example submission."""
    from torch import nn

    first, second = HIDDEN_SIZES
    return nn.Sequential(
        nn.Linear(INPUT_SIZE, first),
        nn.ReLU(),
        nn.Linear(first, second),
        nn.ReLU(),
        nn.Linear(second, output_size),
    )


def _select_device(requested):
    import torch

    return torch.device(
        requested if requested != "auto" else "mps" if torch.backends.mps.is_available() else "cpu"
    )


def _distance(environment):
    food = environment.food
    if food is None:
        return 0
    head_x, head_y = environment.snake[0]
    food_x, food_y = food
    return abs(head_x - food_x) + abs(head_y - food_y)


def _shape_reward(previous_distance, next_distance, reward):
    return (
        reward
        + 0.05 * (previous_distance - next_distance)
        - 0.001
        + 0.5 * ((reward > 0) - (reward < 0))
    )


def _gae(rewards, values, dones, gamma, lam):
    advantages = [0.0] * len(rewards)
    gae = 0.0
    next_value = values[len(rewards)] if len(rewards) < len(values) else 0.0
    for index in reversed(range(len(rewards))):
        nonterminal = not dones[index]
        delta = rewards[index] + gamma * next_value * nonterminal - values[index]
        gae = delta + gamma * lam * nonterminal * gae
        advantages[index] = gae
        next_value = values[index]
    return advantages, [
        advantage + value for advantage, value in zip(advantages, values, strict=True)
    ]


def _evaluate(policy, device, seeds, benchmark):
    import torch

    from minimal_nn_competition.benchmarks.common.snake.environment import SnakeEnvironment

    lengths = []
    wins = []
    with torch.no_grad():
        for seed in seeds:
            environment = SnakeEnvironment(
                seed=seed,
                max_moves=4096,
                **benchmark.environment_options(seed),
            )
            observation = environment.reset()
            done = False
            while not done:
                state = torch.tensor(observation, dtype=torch.float32, device=device)[None]
                action = int(torch.argmax(policy(state), dim=1).item())
                observation, _, done = environment.step(action)
            lengths.append(len(environment.snake))
            wins.append(bool(getattr(environment, "won", False)))
    logging.getLogger(__name__).info("Greedy evaluation wins %d/%d", sum(wins), len(wins))
    return lengths


def train(
    output_dir,
    iterations=600,
    episodes_per_iteration=4,
    max_moves=None,
    update_epochs=4,
    batch_size=64,
    device_name="auto",
    seed=0,
    name=None,
    author="unknown",
    benchmark="snake-simple-8x8",
):
    """Train the example Snake policy with PPO and write the submission files."""
    import torch
    from torch import nn
    from torch.distributions import Categorical

    from minimal_nn_competition.benchmarks.common.snake.environment import SnakeEnvironment
    from minimal_nn_competition.benchmarks.registry import SNAKE_BENCHMARKS, create_benchmark
    from minimal_nn_competition.benchmarks.snake_simple_8x8 import DEFAULT_SEEDS

    if benchmark not in SNAKE_BENCHMARKS:
        raise ValueError(f"Unknown Snake benchmark {benchmark!r}")
    benchmark_instance = create_benchmark(benchmark)
    size = benchmark_instance.size
    max_moves = max_moves or size * size * 4

    logger = logging.getLogger(__name__)
    torch.manual_seed(seed)
    random.seed(seed)
    device = _select_device(device_name)
    logger.info(f"Training {benchmark} example on {device}")

    policy = create_model().to(device)
    value_net = create_model(1).to(device)
    parameters = list(policy.parameters()) + list(value_net.parameters())
    optimizer = torch.optim.Adam(parameters, lr=3e-4)
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
        for episode in range(episodes_per_iteration):
            episode_seed = seed + iteration * episodes_per_iteration + episode
            environment = SnakeEnvironment(
                seed=episode_seed,
                max_moves=max_moves,
                **benchmark_instance.environment_options(episode_seed),
            )
            observation = environment.reset()
            episode_rewards = []
            episode_values = []
            episode_dones = []
            for _ in range(max_moves):
                state = torch.tensor(observation, dtype=torch.float32, device=device)[None]
                with torch.no_grad():
                    distribution = Categorical(logits=policy(state))
                    action = int(distribution.sample().item())
                    log_prob = distribution.log_prob(torch.tensor(action, device=device))
                    value = value_net(state).item()
                previous_distance = _distance(environment)
                observation, reward, done = environment.step(action)
                shaped_reward = _shape_reward(previous_distance, _distance(environment), reward)
                if done and getattr(environment, "won", False):
                    shaped_reward += 1.0
                states.append(state.squeeze(0))
                actions.append(action)
                old_log_probs.append(log_prob.item())
                episode_rewards.append(shaped_reward)
                episode_values.append(value)
                episode_dones.append(done)
                if done:
                    break
            advantages, returns = _gae(episode_rewards, episode_values, episode_dones, gamma, lam)
            returns_all.extend(returns)
            advantages_all.extend(advantages)
            episode_lengths.append(len(environment.snake))

        states_t = torch.stack(states)
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
                nn.utils.clip_grad_norm_(parameters, 0.5)
                optimizer.step()

        if (iteration + 1) % 5 == 0 or iteration == 0:
            mean_length = sum(episode_lengths) / len(episode_lengths)
            logger.info(
                f"iteration {iteration + 1}/{iterations} length {mean_length:.1f} food {mean_length - 1:.2f}"
            )

    lengths = _evaluate(policy, device, DEFAULT_SEEDS, benchmark_instance)
    logger.info(
        f"Greedy evaluation: mean {statistics.fmean(lengths):.2f}, std {statistics.pstdev(lengths):.2f}, lengths {lengths}"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    state_dict = {key: value.cpu() for key, value in policy.state_dict().items()}
    torch.save(state_dict, output_dir / "model.pt")
    config = dict(
        benchmark=benchmark,
        name=name or output_dir.name,
        author=author or "unknown",
        architecture="mlp",
        input_size=INPUT_SIZE,
        output_size=OUTPUT_SIZE,
        hidden_sizes=list(HIDDEN_SIZES),
        activation="relu",
    )
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    logger.info(f"Wrote {output_dir / 'model.pt'}")
    return state_dict


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("models/snake-simple-8x8/example"))
    parser.add_argument("--iterations", type=int, default=600)
    parser.add_argument("--max-moves", type=int, default=None)
    parser.add_argument("--device", default="auto", choices=("auto", "mps", "cpu"))
    parser.add_argument(
        "--name", default=None, help="Display name (defaults to the output directory)"
    )
    parser.add_argument("--author", default="unknown", help="Submission author")
    args = parser.parse_args(argv)
    train(
        args.output,
        iterations=args.iterations,
        max_moves=args.max_moves,
        device_name=args.device,
        name=args.name,
        author=args.author,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
