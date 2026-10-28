"""Install generated site artifacts with rollback on failure."""

import logging
import os
import shutil
import tempfile
from pathlib import Path

from minimal_nn_competition.benchmarks.registry import RL_BENCHMARKS

logger = logging.getLogger(__name__)
_SITE_FILES = ("leaderboard.json", "index.html", ".nojekyll")
_SITE_DIRS = ("submissions", "games")


def install_site(rendered: Path, output_dir: Path) -> None:
    """Swap rendered artifacts into place, restoring the previous tree on failure."""
    output_dir.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".mnnc-stage-", dir=output_dir))
    backup = Path(tempfile.mkdtemp(prefix=".mnnc-backup-", dir=output_dir))
    touched: list[str] = []
    preserve_backup = False
    try:
        for name in _SITE_FILES:
            shutil.copyfile(rendered / name, staging / name)
        for name in _SITE_DIRS:
            source = rendered / name
            if source.is_dir():
                shutil.copytree(source, staging / name)
            else:
                (staging / name).mkdir()
            _preserve_unrendered_files(output_dir / name, staging / name)
        for name in _SITE_FILES + _SITE_DIRS:
            target = output_dir / name
            if target.exists() or target.is_symlink():
                os.replace(target, backup / name)
            touched.append(name)
            os.replace(staging / name, target)
    except BaseException:
        try:
            for name in reversed(touched):
                target = output_dir / name
                if target.is_dir() and not target.is_symlink():
                    shutil.rmtree(target, ignore_errors=True)
                elif target.exists() or target.is_symlink():
                    target.unlink()
                saved = backup / name
                if saved.exists() or saved.is_symlink():
                    os.replace(saved, target)
        except BaseException:
            preserve_backup = True
            logger.exception(
                "Could not restore the previous site tree; backup retained at %s", backup
            )
            raise
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        if not preserve_backup:
            shutil.rmtree(backup, ignore_errors=True)


def _preserve_unrendered_files(source: Path, target: Path) -> None:
    """Keep files in generated directories that the renderer did not produce."""
    if not source.is_dir():
        return
    for path in source.rglob("*"):
        relative = path.relative_to(source)
        destination = target / relative
        if destination.exists() or destination.is_symlink():
            continue
        if path.is_dir() and not path.is_symlink():
            destination.mkdir(parents=True, exist_ok=True)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            if path.is_symlink():
                destination.symlink_to(os.readlink(path))
            else:
                shutil.copy2(path, destination)


def _record_replay(replay_dir, name, benchmark, submission):
    if benchmark.name in RL_BENCHMARKS:
        from minimal_nn_competition.benchmarks.common.control.replay import (
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
    from minimal_nn_competition.benchmarks.common.snake.replay import (
        REPLAY_SEED,
        record_game,
        replay_stem,
        save_replay,
    )
    from minimal_nn_competition.benchmarks.snake_simple_8x8 import selected_input_indices

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
    from minimal_nn_competition.benchmarks.common.snake.replay import load_replay

    if replay_dir is None or not replay_dir.is_dir():
        return {}
    return {path.stem: load_replay(path) for path in sorted(replay_dir.glob("*.json"))}
