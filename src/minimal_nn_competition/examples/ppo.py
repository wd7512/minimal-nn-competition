"""Small native Gymnasium PPO trainer for discrete logits or continuous torque."""

import logging
import random

logger = logging.getLogger(__name__)


def select_device(requested):
    """Resolve ``auto`` to MPS when available, otherwise CPU."""
    import torch

    if requested != "auto":
        return torch.device(requested)
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def build_policy(input_size, output_size, hidden_sizes=(64, 64)):
    """Return the ReLU MLP policy shape used by the control examples."""
    from torch import nn

    layers = []
    width = input_size
    for hidden_size in hidden_sizes:
        layers.extend((nn.Linear(width, hidden_size), nn.ReLU()))
        width = hidden_size
    layers.append(nn.Linear(width, output_size))
    return nn.Sequential(*layers)


def _gae(rewards, values, dones, gamma, lam):
    advantages = [0.0] * len(rewards)
    gae = 0.0
    next_value = 0.0
    for index in reversed(range(len(rewards))):
        nonterminal = not dones[index]
        delta = rewards[index] + gamma * next_value * nonterminal - values[index]
        gae = delta + gamma * lam * nonterminal * gae
        advantages[index] = gae
        next_value = values[index]
    return advantages, [
        advantage + value for advantage, value in zip(advantages, values, strict=True)
    ]


def _distribution(outputs, continuous):
    from torch.distributions import Categorical, Independent, Normal

    return Independent(Normal(outputs, 1.0), 1) if continuous else Categorical(logits=outputs)


def _collect_episode(
    environment, policy, value_net, device, gamma, lam, seed, reset_options, continuous
):
    import torch

    state, _ = environment.reset(seed=seed, options=reset_options)
    states = []
    actions = []
    log_probs = []
    rewards = []
    values = []
    dones = []
    while True:
        inputs = torch.tensor(state, dtype=torch.float32, device=device)[None]
        with torch.no_grad():
            distribution = _distribution(policy(inputs), continuous)
            sample = distribution.sample()
            action = sample[0].cpu().numpy().clip(-2, 2) if continuous else int(sample.item())
            log_prob = distribution.log_prob(sample)
            value = value_net(inputs).item()
        next_state, reward, terminated, truncated, _ = environment.step(action)
        done = terminated or truncated
        states.append(inputs.squeeze(0))
        actions.append(sample[0].cpu())
        log_probs.append(log_prob.item())
        rewards.append(reward)
        values.append(value)
        dones.append(done)
        state = next_state
        if done:
            break
    advantages, returns = _gae(rewards, values, dones, gamma, lam)
    return states, actions, log_probs, advantages, returns, sum(rewards)


def train_ppo(
    environment,
    iterations,
    episodes_per_iteration=8,
    update_epochs=4,
    batch_size=64,
    hidden_sizes=(64, 64),
    learning_rate=3e-4,
    gamma=0.99,
    lam=0.95,
    clip=0.2,
    entropy_coef=0.01,
    value_coef=0.5,
    device_name="auto",
    seed=0,
    reset_options=None,
):
    """Train a policy on seeds >=100; deterministic inference uses its mean/logits."""
    import torch
    from torch import nn

    torch.manual_seed(seed)
    random.seed(seed)
    device = select_device(device_name)
    if iterations <= 0 or episodes_per_iteration <= 0 or seed < 0:
        raise ValueError("Training requires positive iterations/episodes and a nonnegative seed")
    continuous = len(environment.action_space.shape) > 0
    outputs = environment.action_space.shape[0] if continuous else environment.action_space.n
    logger.info("Training %s on %s", environment.spec.id, device)

    policy = build_policy(environment.observation_space.shape[0], outputs, hidden_sizes).to(device)
    value_net = build_policy(environment.observation_space.shape[0], 1, hidden_sizes).to(device)
    parameters = list(policy.parameters()) + list(value_net.parameters())
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)

    for iteration in range(iterations):
        states = []
        actions = []
        old_log_probs = []
        returns_all = []
        advantages_all = []
        episode_returns = []
        for episode in range(episodes_per_iteration):
            collected = _collect_episode(
                environment,
                policy,
                value_net,
                device,
                gamma,
                lam,
                100 + seed + iteration * episodes_per_iteration + episode,
                reset_options,
                continuous,
            )
            (
                episode_states,
                episode_actions,
                episode_log_probs,
                advantages,
                returns,
                total_reward,
            ) = collected
            states.extend(episode_states)
            actions.extend(episode_actions)
            old_log_probs.extend(episode_log_probs)
            advantages_all.extend(advantages)
            returns_all.extend(returns)
            episode_returns.append(total_reward)

        states_t = torch.stack(states)
        actions_t = torch.stack(actions).to(device)
        old_log_probs_t = torch.tensor(old_log_probs, dtype=torch.float32, device=device)
        advantages_t = torch.tensor(advantages_all, dtype=torch.float32, device=device)
        returns_t = torch.tensor(returns_all, dtype=torch.float32, device=device)
        advantages_t = (advantages_t - advantages_t.mean()) / (
            advantages_t.std(unbiased=False) + 1e-8
        )

        for _ in range(update_epochs):
            order = torch.randperm(len(states_t), device=device)
            for start in range(0, len(states_t), batch_size):
                index = order[start : start + batch_size]
                distribution = _distribution(policy(states_t[index]), continuous)
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

        if (iteration + 1) % 10 == 0 or iteration == 0:
            logger.info(
                "iteration %d/%d mean return %.2f",
                iteration + 1,
                iterations,
                sum(episode_returns) / len(episode_returns),
            )

    policy.eval()
    return policy
