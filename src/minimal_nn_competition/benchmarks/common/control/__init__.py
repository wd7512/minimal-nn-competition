"""Native Gymnasium control evaluation; the JSON manifest freezes the contract."""

import hashlib
import json
import statistics
from importlib.metadata import version

import gymnasium as gym
import numpy as np
from gymnasium.envs.box2d.lunar_lander import LunarLander
from gymnasium.envs.registration import EnvSpec

from minimal_nn_competition.benchmarks.registry import CONTROL_SPEC as SPEC
from minimal_nn_competition.model.entry import materialize_submission

TASKS = SPEC["tasks"]


class WideLander(LunarLander):
    """Add reset force before native reset's step(0), including shaping initialization."""

    def step(self, action):
        if self.prev_shaping is None:
            assert self.lander is not None
            self.lander.ApplyForceToCenter(self.np_random.uniform(-500, 500, 2).tolist(), True)
        return super().step(action)


def environment(name, render=False, max_steps=None):
    if any(version(package) != expected for package, expected in SPEC["dependencies"].items()):
        raise RuntimeError("Gymnasium scoring requires pinned dependencies; run uv sync --locked")
    task = TASKS[name]
    horizon = task["horizon"] if max_steps is None else max_steps
    if horizon <= 0:
        raise ValueError("max_steps must be positive")
    entry = task["environment"]
    if name == "lunarlander-gym-v1":
        entry = EnvSpec(
            "LunarLanderWideStart-v1",
            entry_point="minimal_nn_competition.benchmarks.common.control:WideLander",
        )
    return gym.make(
        entry,
        render_mode="rgb_array" if render else None,
        max_episode_steps=horizon,
        **task["kwargs"],
    )


def action(name, outputs):
    values = np.asarray(outputs, dtype=np.float32)
    if values.shape != (TASKS[name]["outputs"],) or not np.isfinite(values).all():
        raise ValueError("Policy output has wrong shape or non-finite values")
    return np.clip(values, -2, 2) if name == "pendulum-gym-v1" else int(values.argmax())


def rollout(env, name, model, seed, video=None):
    """One native episode, optionally encoded from the same scoring transitions."""
    import torch

    from minimal_nn_competition.benchmarks.common.control.video import encoder

    observation, _ = env.reset(seed=seed, options=TASKS[name]["reset"])
    total, steps = 0.0, 0
    digest = hashlib.sha256(observation.tobytes())
    with torch.inference_mode(), encoder(env, video) as frames:
        while True:
            outputs = model(torch.as_tensor(observation, dtype=torch.float32)[None]).cpu().numpy()
            if outputs.shape != (1, TASKS[name]["outputs"]):
                raise ValueError("Policy output must have batch shape (1, output_size)")
            control = action(name, outputs[0])
            observation, reward, terminated, truncated, _ = env.step(control)
            total += reward
            steps += 1
            digest.update(observation.tobytes())
            digest.update(np.asarray([reward, terminated, truncated], dtype=np.float64).tobytes())
            if frames:
                frames.write(env.render().tobytes())
            if terminated or truncated:
                break
    return dict(
        reward=total,
        steps=steps,
        terminated=bool(terminated),
        truncated=bool(truncated),
        trajectory_sha256=digest.hexdigest(),
    )


class ControlBenchmark:
    name = ""

    def __init__(self, seeds=tuple(SPEC["seeds"]), max_steps=None):
        if not seeds or any(type(seed) is not int or seed < 0 for seed in seeds):
            raise ValueError("Control benchmarks require nonnegative integer seeds")
        self.seeds = tuple(seeds)
        self.max_steps = TASKS[self.name]["horizon"] if max_steps is None else max_steps
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")

    def load_submission(self, submission):
        task = TASKS[self.name]
        config = submission.config
        if config.benchmark != self.name:
            raise ValueError("Submission benchmark does not match versioned Gymnasium benchmark")
        if config.input_size != task["inputs"] or config.output_size != task["outputs"]:
            raise ValueError(
                f"{self.name} requires input_size {task['inputs']}, output_size {task['outputs']}"
            )
        return materialize_submission(submission)

    def evaluate(self, submission):
        with environment(self.name, max_steps=self.max_steps) as env:
            episodes = [rollout(env, self.name, submission.model, seed) for seed in self.seeds]
        rewards = [ep["reward"] for ep in episodes]
        mean = statistics.fmean(rewards)
        if self.name == "lunarlander-gym-v1":
            score, passed = mean, mean >= TASKS[self.name]["threshold"]
        else:
            successes = [
                (ep["steps"] == self.max_steps and not ep["terminated"])
                if self.name == "cartpole-gym-v1"
                else ep["reward"] >= TASKS[self.name]["threshold"]
                for ep in episodes
            ]
            score, passed = statistics.fmean(successes), all(successes)
        manifest = {
            **SPEC,
            "seeds": self.seeds,
            "benchmark": self.name,
            "max_steps": self.max_steps,
        }
        return (
            score,
            passed,
            dict(
                mean_return=mean,
                seed_returns=rewards,
                seed_steps=[ep["steps"] for ep in episodes],
                seed_terminated=[ep["terminated"] for ep in episodes],
                seed_truncated=[ep["truncated"] for ep in episodes],
                trajectory_sha256=[ep["trajectory_sha256"] for ep in episodes],
                evaluation_spec=manifest,
                evaluation_sha256=hashlib.sha256(
                    json.dumps(manifest, sort_keys=True).encode()
                ).hexdigest(),
            ),
        )
