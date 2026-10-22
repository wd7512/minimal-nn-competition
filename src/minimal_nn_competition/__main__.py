from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from minimal_nn_competition.benchmarks.register import available_benchmarks, create_benchmark
from minimal_nn_competition.leaderboard import load_results, rank_results, save_results
from minimal_nn_competition.leaderboard.render import render_site
from minimal_nn_competition.model.entry import Submission, SubmissionError
from minimal_nn_competition.runner.entry import evaluate_submission


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate minimal neural-network submissions")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list", help="List available benchmarks")
    evaluate = subparsers.add_parser("evaluate", help="Evaluate a submission directory")
    evaluate.add_argument("benchmark", choices=available_benchmarks())
    evaluate.add_argument("submission", type=Path)
    evaluate.add_argument("--data-path", type=Path)
    leaderboard = subparsers.add_parser("leaderboard", help="Print a saved leaderboard")
    leaderboard.add_argument("path", type=Path)
    render = subparsers.add_parser("render", help="Render a leaderboard as static HTML")
    render.add_argument("path", type=Path)
    render.add_argument("output", type=Path)
    render.add_argument(
        "--replays",
        type=Path,
        default=None,
        help="Directory of snake replay JSON files recorded by collect",
    )
    speed = subparsers.add_parser("speed", help="Benchmark the local Snake environment")
    speed.add_argument("--duration", type=float, default=1.0)
    speed.add_argument("--runs", type=int, default=10)
    speed.add_argument("--warmup", type=float, default=1.0)
    speed.add_argument("--output", type=Path, default=Path("speed-leaderboard.json"))
    render_speed = subparsers.add_parser(
        "render-speed", help="Render the hidden Snake speed leaderboard"
    )
    render_speed.add_argument("path", type=Path)
    render_speed.add_argument("output", type=Path)
    internal = subparsers.add_parser(
        "internal", help="Score this repo's own code against its test scopes"
    )
    internal.add_argument("--root", type=Path, default=Path("."))
    internal.add_argument("--output", type=Path, default=Path("internal-leaderboard.json"))
    render_internal = subparsers.add_parser(
        "render-internal", help="Render the internal token leaderboard"
    )
    render_internal.add_argument("path", type=Path)
    render_internal.add_argument("output", type=Path)
    collect = subparsers.add_parser("collect", help="Evaluate every model under models/")
    collect.add_argument("--models", type=Path, default=Path("models"))
    collect.add_argument("--data-path", type=Path, default=Path("data/mnist-test.pt"))
    collect.add_argument("--output", type=Path, default=Path("leaderboard.json"))
    collect.add_argument(
        "--replay-dir",
        type=Path,
        default=None,
        help="Record one seed-0 snake game per submission into this directory",
    )
    download = subparsers.add_parser("download-mnist", help="Download the official MNIST caches")
    download.add_argument("--output", type=Path, default=Path("data"))
    download.add_argument("--force", action="store_true")
    train_mnist = subparsers.add_parser("train-mnist", help="Train the MNIST example submission")
    train_mnist.add_argument("--data-path", type=Path, default=Path("data/mnist-train.pt"))
    train_mnist.add_argument("--output", type=Path, default=Path("models/mnist/example"))
    train_mnist.add_argument("--epochs", type=int, default=20)
    train_mnist.add_argument("--device", default="auto", choices=("auto", "mps", "cpu"))
    train_mnist.add_argument("--normalise", action="store_true")
    train_mnist.add_argument("--name", default=None, help="Display name")
    train_mnist.add_argument("--author", default="unknown", help="Submission author")
    train_snake = subparsers.add_parser("train-snake", help="Train the Snake example submission")
    train_snake.add_argument("--output", type=Path, default=Path("models/snake/example"))
    train_snake.add_argument("--iterations", type=int, default=600)
    train_snake.add_argument("--device", default="auto", choices=("auto", "mps", "cpu"))
    train_snake.add_argument("--name", default=None, help="Display name")
    train_snake.add_argument("--author", default="unknown", help="Submission author")
    return parser


def _record_replay(replay_dir: Path, name: str, benchmark: Any, submission: Submission) -> None:
    from minimal_nn_competition.benchmarks.snake.replay import (
        REPLAY_SEED,
        record_game,
        replay_slug,
        save_replay,
    )

    loaded = benchmark.load_submission(submission)
    payload = record_game(loaded.model, seed=REPLAY_SEED, max_moves=benchmark.max_moves)
    save_replay(replay_dir / f"{replay_slug(name)}.json", payload)


def _load_replays(replay_dir: Path | None) -> dict[str, dict[str, Any]]:
    from minimal_nn_competition.benchmarks.snake.replay import load_replay

    replays: dict[str, dict[str, Any]] = {}
    if replay_dir is None or not replay_dir.is_dir():
        return replays
    for path in sorted(replay_dir.glob("*.json")):
        replays[path.stem] = load_replay(path)
    return replays


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parser().parse_args(argv)
    if args.command == "list":
        sys.stdout.write("\n".join(available_benchmarks()) + "\n")
        return 0
    if args.command == "leaderboard":
        try:
            results = rank_results(load_results(args.path))
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            logging.getLogger(__name__).error("%s", exc)
            return 2
        sys.stdout.write(
            json.dumps([asdict(result) for result in results], indent=2, sort_keys=True) + "\n"
        )
        return 0
    if args.command == "render":
        try:
            replays = _load_replays(args.replays)
            index = render_site(load_results(args.path), args.output, replays)
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            logging.getLogger(__name__).error("%s", exc)
            return 2
        sys.stdout.write(f"{index}\n")
        return 0
    if args.command == "speed":
        from minimal_nn_competition.speed.benchmark import measure_snake_speed
        from minimal_nn_competition.speed.leaderboard import (
            load_speed_results,
            save_speed_results,
            upsert_speed_result,
        )

        try:
            result = measure_snake_speed(
                duration=args.duration,
                runs=args.runs,
                warmup=args.warmup,
            )
            results = load_speed_results(args.output) if args.output.exists() else []
            save_speed_results(args.output, upsert_speed_result(results, result))
        except (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as exc:
            logging.getLogger(__name__).error("%s", exc)
            return 2
        sys.stdout.write(result.to_json() + "\n")
        return 0
    if args.command == "render-speed":
        from minimal_nn_competition.speed.leaderboard import load_speed_results
        from minimal_nn_competition.speed.render import render_speed_site

        try:
            index = render_speed_site(load_speed_results(args.path), args.output)
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            logging.getLogger(__name__).error("%s", exc)
            return 2
        sys.stdout.write(f"{index}\n")
        return 0
    if args.command == "internal":
        from minimal_nn_competition.internal import (
            SCOPES,
            load_internal_results,
            measure_scope,
            save_internal_results,
            upsert_internal_result,
        )

        try:
            results = load_internal_results(args.output) if args.output.exists() else []
            for scope in SCOPES:
                result = measure_scope(scope, args.root)
                results = upsert_internal_result(results, result)
                sys.stdout.write(result.to_json() + "\n")
            save_internal_results(args.output, results)
        except (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as exc:
            logging.getLogger(__name__).error("%s", exc)
            return 2
        return 0
    if args.command == "render-internal":
        from minimal_nn_competition.internal import load_internal_results
        from minimal_nn_competition.internal.render import render_internal_site

        try:
            index = render_internal_site(load_internal_results(args.path), args.output)
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            logging.getLogger(__name__).error("%s", exc)
            return 2
        sys.stdout.write(f"{index}\n")
        return 0
    if args.command == "collect":
        results = []
        for config_path in sorted(args.models.glob("*/*/config.json")):
            try:
                submission = Submission.from_directory(config_path.parent)
                benchmark_kwargs = {}
                if submission.config.benchmark == "mnist":
                    benchmark_kwargs["data_path"] = args.data_path
                benchmark = create_benchmark(submission.config.benchmark, **benchmark_kwargs)
                results.append(evaluate_submission(submission, benchmark))
                if args.replay_dir is not None and submission.config.benchmark == "snake":
                    _record_replay(args.replay_dir, config_path.parent.name, benchmark, submission)
            except (OSError, RuntimeError, SubmissionError, TypeError, ValueError) as exc:
                logging.getLogger(__name__).error("%s: %s", config_path.parent, exc)
                return 2
        save_results(args.output, results)
        sys.stdout.write(f"{args.output}\n")
        return 0
    if args.command == "download-mnist":
        try:
            from minimal_nn_competition.benchmarks.mnist.data_loader import download_mnist

            train_path, test_path = download_mnist(args.output, force=args.force)
        except (OSError, ValueError) as exc:
            logging.getLogger(__name__).error("%s", exc)
            return 2
        sys.stdout.write(f"{train_path}\n{test_path}\n")
        return 0
    if args.command == "train-mnist":
        from minimal_nn_competition.examples.mnist_submission import train

        try:
            train(
                args.data_path,
                args.output,
                epochs=args.epochs,
                device_name=args.device,
                normalise=args.normalise,
                name=args.name,
                author=args.author,
            )
        except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
            logging.getLogger(__name__).error("%s", exc)
            return 2
        return 0
    if args.command == "train-snake":
        from minimal_nn_competition.examples.snake_submission import train

        try:
            train(
                args.output,
                iterations=args.iterations,
                device_name=args.device,
                name=args.name,
                author=args.author,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            logging.getLogger(__name__).error("%s", exc)
            return 2
        return 0
    if args.data_path is not None and args.benchmark != "mnist":
        logging.getLogger(__name__).error("--data-path is only supported for the mnist benchmark")
        return 2
    benchmark_kwargs = {}
    if args.data_path is not None:
        benchmark_kwargs["data_path"] = args.data_path
    try:
        benchmark = create_benchmark(args.benchmark, **benchmark_kwargs)
        result = evaluate_submission(Submission.from_directory(args.submission), benchmark)
    except (FileNotFoundError, RuntimeError, SubmissionError, TypeError, ValueError) as exc:
        logging.getLogger(__name__).error("%s", exc)
        return 2
    sys.stdout.write(result.to_json() + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
