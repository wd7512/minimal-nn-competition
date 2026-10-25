import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

from minimal_nn_competition.benchmarks.chess_eval.names import CHESS_BENCHMARKS
from minimal_nn_competition.benchmarks.control.names import RL_BENCHMARKS
from minimal_nn_competition.benchmarks.register import available_benchmarks, create_benchmark
from minimal_nn_competition.benchmarks.snake.names import SNAKE_BENCHMARKS
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
    validate_obstacles = add(
        "validate-snake-obstacles-8x8",
        help="Prove every seeded obstacle layout has a closed tour and is passable",
    )
    validate_obstacles.add_argument("--seeds", type=int, default=100)
    evaluate = add("evaluate", help="Evaluate a submission directory")
    evaluate.add_argument("benchmark", choices=available_benchmarks())
    evaluate.add_argument("submission_dir", type=Path)
    evaluate.add_argument("--data-path", type=Path)
    leaderboard = add("leaderboard", help="Print a saved leaderboard")
    leaderboard.add_argument("path", type=Path)
    render = add("render", help="Render a leaderboard as static HTML")
    _render_args(render)
    render.add_argument(
        "--replays",
        type=Path,
        default=None,
        help="Directory of Snake and control replay JSON files recorded by collect",
    )
    speed = add("speed", help="Benchmark the local Snake Simple 8x8 environment")
    speed.add_argument("--duration", type=float, default=1.0)
    speed.add_argument("--runs", type=int, default=10)
    speed.add_argument("--warmup", type=float, default=1.0)
    speed.add_argument("--output", type=Path, default=Path("docs/internal/speed/leaderboard.json"))
    render_speed = add("render-speed", help="Render the hidden Snake Simple 8x8 speed leaderboard")
    _render_args(render_speed)
    internal = add("internal", help="Score this repo's own code against its test scopes")
    internal.add_argument("--root", type=Path, default=Path("."))
    internal.add_argument("--output", type=Path, default=Path("docs/internal/leaderboard.json"))
    render_internal = add("render-internal", help="Render the internal token leaderboard")
    _render_args(render_internal)
    collect = add("collect", help="Evaluate every model under models/")
    collect.add_argument("--models", type=Path, default=Path("models"))
    collect.add_argument("--data-path", type=Path, default=Path("data/mnist.pt"))
    collect.add_argument(
        "--fashion-mnist-data-path",
        type=Path,
        default=Path("data/fashion-mnist.pt"),
    )
    submissions = add("validate-submissions", help="Validate and promote open submission PRs")
    target = submissions.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--all", action="store_true", help="Process every open PR targeting submission"
    )
    target.add_argument("--pr", type=int, help="Process one open submission PR")
    submissions.add_argument("--data-root", type=Path, default=Path("data"))
    collect.add_argument("--output", type=Path, default=Path("docs/leaderboard.json"))
    collect.add_argument(
        "--replay-dir",
        type=Path,
        default=None,
        help="Record one seed-0 Snake or control game per submission into this directory",
    )
    download = add("download-mnist", help="Download the official MNIST dataset")
    _download_args(download)
    download_fashion = add(
        "download-fashion-mnist", help="Download the official Fashion-MNIST dataset"
    )
    _download_args(download_fashion)
    train_mnist = add("train-mnist", help="Train the MNIST example submission")
    _classifier_train_args(train_mnist, "data/mnist.pt", "models/mnist/example")
    train_fashion = add("train-fashion-mnist", help="Train the Fashion-MNIST example submission")
    _classifier_train_args(train_fashion, "data/fashion-mnist.pt", "models/fashion-mnist/example")
    for benchmark in SNAKE_BENCHMARKS:
        train_snake = add(f"train-{benchmark}", help=f"Train the {benchmark} example submission")
        train_snake.add_argument("--output", type=Path, default=Path(f"models/{benchmark}/example"))
        train_snake.add_argument("--iterations", type=int, default=600)
        train_snake.add_argument("--max-moves", type=int, default=None)
        _train_common(train_snake)
    for benchmark in RL_BENCHMARKS:
        train_rl = add(f"train-{benchmark}", help=f"Train the {benchmark} example submission")
        train_rl.add_argument("--output", type=Path, default=Path(f"models/{benchmark}/example"))
        if benchmark != "cartpole":
            train_rl.add_argument(
                "--iterations", type=int, default=500 if benchmark == "pendulum" else 1500
            )
            train_rl.add_argument("--episodes-per-iteration", type=int, default=8)
        train_rl.add_argument("--seed", type=int, default=0)
        _train_common(train_rl)
    for benchmark in CHESS_BENCHMARKS:
        train_chess = add(f"train-{benchmark}", help=f"Train the {benchmark} example submission")
        train_chess.add_argument("--output", type=Path, default=Path(f"models/{benchmark}/example"))
        train_chess.add_argument("--data-path", type=Path, default=None)
        train_chess.add_argument("--epochs", type=int, default=16_000)
        train_chess.add_argument("--batch-size", type=int, default=256)
        train_chess.add_argument("--learning-rate", type=float, default=1e-3)
        train_chess.add_argument("--hidden-sizes", type=int, nargs="+", default=[64, 64])
        train_chess.add_argument("--seed", type=int, default=0)
        _train_common(train_chess)
    return parser


def _record_replay(replay_dir, name, benchmark, submission):
    if benchmark.name in RL_BENCHMARKS:
        from minimal_nn_competition.benchmarks.control.replay import (
            REPLAY_SEED,
            record_game,
            replay_stem,
            save_replay,
        )

        steps = benchmark.max_steps
        loaded = benchmark.load_submission(submission)
        save_replay(
            replay_dir / f"{replay_stem(benchmark.name, name)}.json",
            record_game(benchmark.name, loaded.model, seed=REPLAY_SEED, steps=steps),
        )
        return
    from minimal_nn_competition.benchmarks.snake.benchmark import selected_input_indices
    from minimal_nn_competition.benchmarks.snake.replay import (
        REPLAY_SEED,
        record_game,
        replay_stem,
        save_replay,
    )

    loaded = benchmark.load_submission(submission)
    save_replay(
        replay_dir / f"{replay_stem(benchmark.name, name)}.json",
        record_game(
            loaded.model,
            seed=REPLAY_SEED,
            max_moves=benchmark.max_moves,
            input_indices=selected_input_indices(submission.config, benchmark.name),
            **benchmark.environment_options(REPLAY_SEED),
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
    if args.command == "validate-submissions":
        from minimal_nn_competition.submissions.workflow import run

        try:
            return run(Path.cwd(), None if args.all else args.pr, args.data_root)
        except (OSError, RuntimeError, ValueError, TypeError) as exc:
            return _fail(exc)
    if args.command == "list":
        _emit("\n".join(available_benchmarks()))
        return 0
    if args.command == "validate-snake-obstacles-8x8":
        from minimal_nn_competition.benchmarks.snake.layout import validate_seed

        failures = 0
        for seed in range(args.seeds):
            try:
                result = validate_seed(8, seed)
            except (AssertionError, RuntimeError, ValueError) as exc:
                logger.error("seed %d failed: %s", seed, exc)
                failures += 1
            else:
                logger.info(
                    "seed %d passed: %d blocked cells, tour %d, %d moves",
                    seed,
                    len(result["blocked"]),
                    result["tour_length"],
                    result["moves"],
                )
        return 1 if failures else 0
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
                if args.replay_dir is not None and submission.config.benchmark in (
                    SNAKE_BENCHMARKS + RL_BENCHMARKS
                ):
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
            cache_path = download_mnist(args.output, force=args.force)
        except (OSError, ValueError) as exc:
            return _fail(exc)
        _emit(cache_path)
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
    if args.command in tuple(f"train-{name}" for name in SNAKE_BENCHMARKS):
        from minimal_nn_competition.examples.snake_submission import train

        try:
            train(
                args.output,
                iterations=args.iterations,
                max_moves=args.max_moves,
                device_name=args.device,
                name=args.name,
                author=args.author,
                benchmark=args.command.removeprefix("train-"),
            )
        except (OSError, RuntimeError, ValueError) as exc:
            return _fail(exc)
        return 0
    if args.command in tuple(f"train-{name}" for name in RL_BENCHMARKS):
        try:
            if args.command == "train-cartpole":
                from minimal_nn_competition.examples.cartpole_submission import (
                    train as train_cartpole,
                )

                train_cartpole(
                    args.output,
                    seed=args.seed,
                    name=args.name,
                    author=args.author,
                )
            elif args.command == "train-pendulum":
                from minimal_nn_competition.examples.pendulum_submission import (
                    train as train_pendulum,
                )

                train_pendulum(
                    args.output,
                    iterations=args.iterations,
                    episodes_per_iteration=args.episodes_per_iteration,
                    device_name=args.device,
                    seed=args.seed,
                    name=args.name,
                    author=args.author,
                )
            else:
                from minimal_nn_competition.examples.lunarlander_submission import (
                    train as train_lunarlander,
                )

                train_lunarlander(
                    args.output,
                    iterations=args.iterations,
                    episodes_per_iteration=args.episodes_per_iteration,
                    device_name=args.device,
                    seed=args.seed,
                    name=args.name,
                    author=args.author,
                )
        except (OSError, RuntimeError, ValueError) as exc:
            return _fail(exc)
        return 0
    if args.command in tuple(f"train-{benchmark}" for benchmark in CHESS_BENCHMARKS):
        from minimal_nn_competition.examples.chess_eval_submission import train

        try:
            train(
                args.output,
                data_path=args.data_path,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                hidden_sizes=tuple(args.hidden_sizes),
                device_name=args.device,
                seed=args.seed,
                name=args.name,
                author=args.author,
                benchmark=args.command.removeprefix("train-"),
            )
        except (OSError, RuntimeError, ValueError) as exc:
            return _fail(exc)
        return 0
    if args.data_path is not None and args.benchmark not in (
        "mnist",
        "fashion-mnist",
        *CHESS_BENCHMARKS,
    ):
        logger.error(
            "--data-path is only supported for the mnist, fashion-mnist, and chess evaluation benchmarks"
        )
        return 2
    kwargs = {}
    if args.data_path is not None:
        kwargs["data_path"] = args.data_path
    elif args.benchmark == "mnist":
        kwargs["data_path"] = Path("data/mnist.pt")
    elif args.benchmark == "fashion-mnist":
        kwargs["data_path"] = Path("data/fashion-mnist.pt")
    try:
        benchmark = create_benchmark(args.benchmark, **kwargs)
        result = evaluate_submission(Submission.from_directory(args.submission_dir), benchmark)
    except (FileNotFoundError, RuntimeError, SubmissionError, TypeError, ValueError) as exc:
        return _fail(exc)
    _emit(result.to_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
