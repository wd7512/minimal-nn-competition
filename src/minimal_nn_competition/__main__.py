from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

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
    collect = subparsers.add_parser("collect", help="Evaluate every model under models/")
    collect.add_argument("--models", type=Path, default=Path("models"))
    collect.add_argument("--data-path", type=Path, default=Path("data/mnist-test.pt"))
    collect.add_argument("--output", type=Path, default=Path("leaderboard.json"))
    download = subparsers.add_parser("download-mnist", help="Download the official MNIST caches")
    download.add_argument("--output", type=Path, default=Path("data"))
    download.add_argument("--force", action="store_true")
    train_mnist = subparsers.add_parser("train-mnist", help="Train the MNIST example submission")
    train_mnist.add_argument("--data-path", type=Path, default=Path("data/mnist-train.pt"))
    train_mnist.add_argument("--output", type=Path, default=Path("models/mnist/example"))
    train_mnist.add_argument("--epochs", type=int, default=20)
    train_mnist.add_argument("--device", default="auto", choices=("auto", "mps", "cpu"))
    train_mnist.add_argument("--normalise", action="store_true")
    train_snake = subparsers.add_parser("train-snake", help="Train the Snake example submission")
    train_snake.add_argument("--output", type=Path, default=Path("models/snake/example"))
    train_snake.add_argument("--iterations", type=int, default=600)
    train_snake.add_argument("--device", default="auto", choices=("auto", "mps", "cpu"))
    return parser


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
            index = render_site(load_results(args.path), args.output)
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
            except (FileNotFoundError, RuntimeError, SubmissionError, TypeError, ValueError) as exc:
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
            )
        except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
            logging.getLogger(__name__).error("%s", exc)
            return 2
        return 0
    if args.command == "train-snake":
        from minimal_nn_competition.examples.snake_submission import train

        try:
            train(args.output, iterations=args.iterations, device_name=args.device)
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
