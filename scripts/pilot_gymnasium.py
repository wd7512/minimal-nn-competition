# /// script
# requires-python = "==3.12.*"
# dependencies = ["gymnasium[box2d]==1.2.3", "numpy==2.2.6", "pygame==2.6.1", "box2d==2.3.10", "swig==4.5.1"]
# ///
"""Issue #229 experiment only; never loads submissions or changes benchmark registration."""

import argparse
import hashlib
import json
import logging
import math
import platform
import subprocess
import time
from importlib.metadata import version
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium.envs.box2d.lunar_lander import LunarLander
from gymnasium.envs.registration import EnvSpec


class WideLander(LunarLander):
    """Initialization-only extra force; relies on pinned reset calling step(0)."""

    def step(self, action):
        if self.prev_shaping is None:
            self.lander.ApplyForceToCenter(self.np_random.uniform(-500, 500, 2).tolist(), True)
        return super().step(action)


TASKS = {
    "cartpole": ("CartPole-v1", {"low": -0.1, "high": 0.1}, 500, 2),
    "pendulum": ("Pendulum-v1", {"x_init": math.pi, "y_init": 2.0}, 200, 1),
    "lunarlander": ("LunarLander-v3", {}, 1000, 4),
}


def environment(task, challenge, render=False):
    name, _, horizon, _ = TASKS[task]
    if task == "lunarlander" and challenge:
        name = EnvSpec("LunarLanderWideStart-v0", entry_point=WideLander, max_episode_steps=horizon)
    return gym.make(name, render_mode="rgb_array" if render else None)


def action(task, outputs):
    """Unbatched policy output: finite logits or one native continuous torque."""
    values = np.asarray(outputs, dtype=np.float32)
    if values.shape != (TASKS[task][3],) or not np.isfinite(values).all():
        raise ValueError("Policy output has wrong shape or non-finite values")
    return np.clip(values, -2, 2) if task == "pendulum" else int(values.argmax())


def rollout(env, task, seed, options, video=None):
    observation, _ = env.reset(seed=seed, options=options)
    initial = observation.tolist()
    digest = hashlib.sha256(observation.tobytes())
    total, steps = 0.0, 0
    encoder = None
    try:
        if video:
            frame = env.render()
            height, width, _ = frame.shape
            encoder = subprocess.Popen(
                [
                    "ffmpeg",
                    "-loglevel",
                    "error",
                    "-y",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "-s",
                    f"{width}x{height}",
                    "-r",
                    str(env.metadata["render_fps"]),
                    "-i",
                    "-",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-crf",
                    "28",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(video),
                ],
                stdin=subprocess.PIPE,
            )
            encoder.stdin.write(frame.tobytes())
        while True:
            # A fixed zero-weight linear network: idle/zero torque, not a trained submission.
            control = action(task, np.zeros(TASKS[task][3], dtype=np.float32))
            assert env.action_space.contains(control)
            observation, reward, terminated, truncated, _ = env.step(control)
            total += reward
            steps += 1
            digest.update(observation.tobytes())
            digest.update(np.asarray([reward, terminated, truncated], dtype=np.float64).tobytes())
            if encoder:
                encoder.stdin.write(env.render().tobytes())
            if terminated or truncated:
                break
    finally:
        if encoder:
            encoder.stdin.close()
            if encoder.wait():
                raise RuntimeError("ffmpeg failed")
    return dict(
        initial=initial,
        steps=steps,
        reward=total,
        terminated=bool(terminated),
        truncated=bool(truncated),
        sha256=digest.hexdigest(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--verify", type=Path, help="Compare trajectory hashes from another process"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    spec = dict(
        tasks=TASKS,
        seeds=list(range(100)),
        lander_extra_force=[-500, 500],
        policy="zero linear outputs",
        dependencies={
            name: version(name) for name in ("gymnasium", "numpy", "box2d", "pygame", "swig")
        },
        python="3.12",
        environment_kwargs={"continuous": False, "gravity": -10.0, "enable_wind": False},
        candidate_scoring={
            "cartpole": "fraction: 500 steps and not terminated; passed=all",
            "pendulum": "fraction: return >= -500; passed=all",
            "lunarlander": "mean return; passed=mean >= 200",
        },
        video="seed 0; native fps; libx264 crf=28 yuv420p faststart",
    )
    report = {
        "spec": spec,
        "spec_sha256": hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest(),
        "runs": {},
        "platform": platform.platform(),
        "python": platform.python_version(),
    }
    started = time.perf_counter()
    for task, (_, options, _, _) in TASKS.items():
        for challenge in (False, True):
            key = f"{task}-{'challenge' if challenge else 'stock'}"
            with environment(task, challenge) as env:
                before = time.perf_counter()
                episodes = [
                    rollout(env, task, seed, options if challenge else {}) for seed in range(100)
                ]
                elapsed = time.perf_counter() - before
                assert episodes == [
                    rollout(env, task, seed, options if challenge else {}) for seed in range(100)
                ]
            if task == "cartpole":
                assert all(abs(ep["initial"][2]) < math.radians(12) for ep in episodes)
            assert len({tuple(ep["initial"]) for ep in episodes}) == 100
            report["runs"][key] = dict(seconds=elapsed, episodes=episodes)
            logging.info(
                "%s: %.3fs, mean return %.3f",
                key,
                elapsed,
                np.mean([e["reward"] for e in episodes]),
            )
        with environment(task, True, render=True) as env:
            path = args.output / f"{task}.mp4"
            episode = rollout(env, task, 0, options, path)
            assert episode == report["runs"][f"{task}-challenge"]["episodes"][0]
            report["runs"][f"{task}-challenge"]["video_bytes"] = path.stat().st_size
    if args.verify:
        previous = json.loads(args.verify.read_text())
        assert report["spec_sha256"] == previous["spec_sha256"]
        for key, run in report["runs"].items():
            assert run["episodes"] == previous["runs"][key]["episodes"]
    report["total_seconds"] = time.perf_counter() - started
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    logging.info(
        "Repeated trajectories and rendered seed 0 match; total %.3fs", report["total_seconds"]
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
