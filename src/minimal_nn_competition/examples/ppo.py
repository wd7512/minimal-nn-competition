"""Small discrete-action PPO trainer shared by the control examples."""

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


def _collect_episode(environment, policy, value_net, device, gamma, lam):
    import torch
    from torch.distributions import Categorical

    state = environment.reset()
    states = []
    actions = []
    log_probs = []
    rewards = []
    values = []
    dones = []
    for _ in range(environment.max_steps):
        inputs = torch.tensor(environment.observation(state), dtype=torch.float32, device=device)[
            None
        ]
        with torch.no_grad():
            distribution = Categorical(logits=policy(inputs))
            action = int(distribution.sample().item())
            log_prob = distribution.log_prob(torch.tensor(action, device=device))
            value = value_net(inputs).item()
        next_state, reward, done = environment.step(state, action)
        states.append(inputs.squeeze(0))
        actions.append(action)
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
):
    """Train a discrete-action PPO policy and return it in eval mode."""
    import torch
    from torch import nn
    from torch.distributions import Categorical

    torch.manual_seed(seed)
    random.seed(seed)
    device = select_device(device_name)
    logger.info("Training %s on %s", environment.name, device)

    policy = build_policy(environment.input_size, environment.output_size, hidden_sizes).to(device)
    value_net = build_policy(environment.input_size, 1, hidden_sizes).to(device)
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
            environment.seed(seed + iteration * episodes_per_iteration + episode)
            collected = _collect_episode(environment, policy, value_net, device, gamma, lam)
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

        if (iteration + 1) % 10 == 0 or iteration == 0:
            logger.info(
                "iteration %d/%d mean return %.2f",
                iteration + 1,
                iterations,
                sum(episode_returns) / len(episode_returns),
            )

    policy.eval()
    return policy
