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

DEVICES = ("auto", "mps", "cpu")
logger = logging.getLogger(__name__)


def _emit(message):
    sys.stdout.write(f"{message}\n")


def _fail(exc):
    logger.error("%s", exc)
    return 2


def _render_args(parser):
    parser.add_argument("path", type=Path)
    parser.add_argument("output", type=Path)


def _train_common(parser):
    parser.add_argument("--device", default="auto", choices=DEVICES)
    parser.add_argument("--name", default=None, help="Display name")
    parser.add_argument("--author", default="unknown", help="Submission author")


def _classifier_train_args(parser, data, output):
    parser.add_argument("--data-path", type=Path, default=Path(data))
    parser.add_argument("--output", type=Path, default=Path(output))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--normalise", action="store_true")
    _train_common(parser)


def _download_args(parser):
    parser.add_argument("--output", type=Path, default=Path("data"))
    parser.add_argument("--force", action="store_true")


def _parser():
    parser = argparse.ArgumentParser(description="Evaluate minimal neural-network submissions")
    subparsers = parser.add_subparsers(dest="command", required=True)
    add = subparsers.add_parser
    add("list", help="List available benchmarks")
    evaluate = add("evaluate", help="Evaluate a submission directory")
    evaluate.add_argument("benchmark", choices=available_benchmarks())
    evaluate.add_argument("submission", type=Path)
    evaluate.add_argument("--data-path", type=Path)
    leaderboard = add("leaderboard", help="Print a saved leaderboard")
    leaderboard.add_argument("path", type=Path)
    render = add("render", help="Render a leaderboard as static HTML")
    _render_args(render)
    render.add_argument(
        "--replays",
        type=Path,
        default=None,
        help="Directory of snake replay JSON files recorded by collect",
    )
    speed = add("speed", help="Benchmark the local Snake environment")
    speed.add_argument("--duration", type=float, default=1.0)
    speed.add_argument("--runs", type=int, default=10)
    speed.add_argument("--warmup", type=float, default=1.0)
    speed.add_argument("--output", type=Path, default=Path("speed-leaderboard.json"))
    render_speed = add("render-speed", help="Render the hidden Snake speed leaderboard")
    _render_args(render_speed)
    internal = add("internal", help="Score this repo's own code against its test scopes")
    internal.add_argument("--root", type=Path, default=Path("."))
    internal.add_argument("--output", type=Path, default=Path("internal-leaderboard.json"))
    render_internal = add("render-internal", help="Render the internal token leaderboard")
    _render_args(render_internal)
    collect = add("collect", help="Evaluate every model under models/")
    collect.add_argument("--models", type=Path, default=Path("models"))
    collect.add_argument("--data-path", type=Path, default=Path("data/mnist-test.pt"))
    collect.add_argument(
        "--fashion-mnist-data-path",
        type=Path,
        default=Path("data/fashion-mnist-test.pt"),
    )
    collect.add_argument("--output", type=Path, default=Path("leaderboard.json"))
    collect.add_argument(
        "--replay-dir",
        type=Path,
        default=None,
        help="Record one seed-0 snake game per submission into this directory",
    )
    download = add("download-mnist", help="Download the official MNIST caches")
    _download_args(download)
    download_fashion = add(
        "download-fashion-mnist", help="Download the official Fashion-MNIST caches"
    )
    _download_args(download_fashion)
    train_mnist = add("train-mnist", help="Train the MNIST example submission")
    _classifier_train_args(train_mnist, "data/mnist-train.pt", "models/mnist/example")
    train_fashion = add("train-fashion-mnist", help="Train the Fashion-MNIST example submission")
    _classifier_train_args(
        train_fashion, "data/fashion-mnist-train.pt", "models/fashion-mnist/example"
    )
    train_snake = add("train-snake", help="Train the Snake example submission")
    train_snake.add_argument("--output", type=Path, default=Path("models/snake/example"))
    train_snake.add_argument("--iterations", type=int, default=600)
    _train_common(train_snake)
    return parser


def _record_replay(replay_dir, name, benchmark, submission):
    from minimal_nn_competition.benchmarks.snake.benchmark import selected_input_indices
    from minimal_nn_competition.benchmarks.snake.replay import (
        REPLAY_SEED,
        record_game,
        replay_slug,
        save_replay,
    )

    loaded = benchmark.load_submission(submission)
    save_replay(
        replay_dir / f"{replay_slug(name)}.json",
        record_game(
            loaded.model,
            seed=REPLAY_SEED,
            max_moves=benchmark.max_moves,
            input_indices=selected_input_indices(submission.config),
        ),
    )


def _load_replays(replay_dir):
    from minimal_nn_competition.benchmarks.snake.replay import load_replay

    if replay_dir is None or not replay_dir.is_dir():
        return {}
    return {path.stem: load_replay(path) for path in sorted(replay_dir.glob("*.json"))}


def main(argv=None):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parser().parse_args(argv)
    if args.command == "list":
        _emit("\n".join(available_benchmarks()))
        return 0
    if args.command == "leaderboard":
        try:
            results = rank_results(load_results(args.path))
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            return _fail(exc)
        _emit(json.dumps([asdict(result) for result in results], indent=2, sort_keys=True))
        return 0
    if args.command == "render":
        try:
            index = render_site(load_results(args.path), args.output, _load_replays(args.replays))
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            return _fail(exc)
        _emit(index)
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
            return _fail(exc)
        _emit(result.to_json())
        return 0
    if args.command in ("render-speed", "render-internal"):
        if args.command == "render-speed":
            from minimal_nn_competition.speed.leaderboard import load_speed_results as load
            from minimal_nn_competition.speed.render import render_speed_site as render
        else:
            from minimal_nn_competition.internal import load_internal_results as load
            from minimal_nn_competition.internal.render import render_internal_site as render

        try:
            index = render(load(args.path), args.output)
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            return _fail(exc)
        _emit(index)
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
                _emit(result.to_json())
            save_internal_results(args.output, results)
        except (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as exc:
            return _fail(exc)
        return 0
    if args.command == "collect":
        results = []
        for config_path in sorted(args.models.glob("*/*/config.json")):
            try:
                submission = Submission.from_directory(config_path.parent)
                kwargs = {}
                if submission.config.benchmark == "mnist":
                    kwargs["data_path"] = args.data_path
                elif submission.config.benchmark == "fashion-mnist":
                    kwargs["data_path"] = args.fashion_mnist_data_path
                benchmark = create_benchmark(submission.config.benchmark, **kwargs)
                results.append(evaluate_submission(submission, benchmark))
                if args.replay_dir is not None and submission.config.benchmark == "snake":
                    _record_replay(args.replay_dir, config_path.parent.name, benchmark, submission)
            except (OSError, RuntimeError, SubmissionError, TypeError, ValueError) as exc:
                logger.error("%s: %s", config_path.parent, exc)
                return 2
        save_results(args.output, results)
        _emit(args.output)
        return 0
    if args.command in ("download-mnist", "download-fashion-mnist"):
        if args.command == "download-mnist":
            from minimal_nn_competition.benchmarks.mnist.data_loader import download_mnist
        else:
            from minimal_nn_competition.benchmarks.fashion_mnist.data_loader import (
                download_fashion_mnist,
            )

            download_mnist = download_fashion_mnist
        try:
            train_path, test_path = download_mnist(args.output, force=args.force)
        except (OSError, ValueError) as exc:
            return _fail(exc)
        _emit(f"{train_path}\n{test_path}")
        return 0
    if args.command in ("train-mnist", "train-fashion-mnist"):
        if args.command == "train-mnist":
            from minimal_nn_competition.examples.mnist_submission import train
        else:
            from minimal_nn_competition.examples.fashion_mnist_submission import train

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
            return _fail(exc)
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
            return _fail(exc)
        return 0
    if args.data_path is not None and args.benchmark == "snake":
        logger.error("--data-path is only supported for the mnist and fashion-mnist benchmarks")
        return 2
    kwargs = {}
    if args.data_path is not None:
        kwargs["data_path"] = args.data_path
    try:
        benchmark = create_benchmark(args.benchmark, **kwargs)
        result = evaluate_submission(Submission.from_directory(args.submission), benchmark)
    except (FileNotFoundError, RuntimeError, SubmissionError, TypeError, ValueError) as exc:
        return _fail(exc)
    _emit(result.to_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
