"""Small, local development gates for this repository."""

from __future__ import annotations

import argparse
import contextlib
import subprocess
import sys
import tempfile
import unicodedata
from pathlib import Path

from minimal_nn_competition.internal import PACKAGE
from minimal_nn_competition.internal.scoring import score_files
from minimal_nn_competition.speed.benchmark import measure_snake_speed


def _repo_root() -> Path:
    return Path(
        subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], cwd=Path.cwd(), text=True
        ).strip()
    )


ROOT = _repo_root()
TEXT_EXTENSIONS = {".md", ".py", ".toml", ".json", ".yaml", ".yml", ".txt"}
SPECIAL_FILES = {"Dockerfile", "Makefile", "LICENSE"}
EVIDENCE_DIRECTORIES = {"evidence", "captures"}
IGNORED_MARKDOWN_PARTS = {".opencode"}
SPEED_TOLERANCE = 0.05


def _git(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def _baseline_ref() -> str:
    for ref in ("main", "origin/main", "dev", "origin/dev"):
        if (
            subprocess.run(
                ["git", "rev-parse", "--verify", ref], cwd=ROOT, capture_output=True
            ).returncode
            == 0
        ):
            return ref
    raise RuntimeError("Cannot find a baseline ref; expected main or dev")


@contextlib.contextmanager
def _baseline():
    directory = Path(tempfile.mkdtemp(prefix="dev-tools-main-"))
    try:
        with (directory / "tree.tar").open("wb") as archive:
            subprocess.run(
                ["git", "archive", _baseline_ref()], cwd=ROOT, stdout=archive, check=True
            )
        subprocess.run(["tar", "xf", str(directory / "tree.tar"), "-C", str(directory)], check=True)
        (directory / "tree.tar").unlink()
        yield directory
    finally:
        import shutil

        shutil.rmtree(directory, ignore_errors=True)


def _source_files(root: Path) -> list[Path]:
    return sorted(
        path for path in (root / PACKAGE).rglob("*.py") if "__pycache__" not in path.parts
    )


def _markdown_files(root: Path, *, respect_gitignore: bool = False) -> list[Path]:
    if respect_gitignore:
        paths = [
            root / path
            for path in _git(
                "ls-files", "--cached", "--others", "--exclude-standard", "*.md"
            ).splitlines()
        ]
    else:
        paths = sorted(root.rglob("*.md"))
    return [
        path for path in paths if IGNORED_MARKDOWN_PARTS.isdisjoint(path.relative_to(root).parts)
    ]


def _delta_message(delta: int, scale: int = 1) -> str:
    if delta > 100 * scale:
        return "this does not feel very minimal to me ..."
    if delta > 0:
        return "are you sure this is minimal?"
    if delta < -100 * scale:
        return "this is impressively minimal :D"
    if delta < 0:
        return "yay! minimalism achieved :D"
    return ""


def _signed(delta: int, scale: int = 1) -> str:
    value = f"{delta:+,}"
    message = _delta_message(delta, scale)
    return f"{value} ({message})" if message else value


def _score_scope(scope, root):
    """Score a scope, returning None when its files do not exist in the tree."""
    from minimal_nn_competition.internal import scope_files

    try:
        return score_files(scope_files(scope, root))
    except ValueError:
        return None


def _format_count(value):
    return f"{value:7,}" if value is not None else "    n/a"


def _format_delta(old, current, scale=1):
    if old is None:
        return "(new scope)"
    return _signed(current - old, scale)


def tokens() -> int:
    from minimal_nn_competition.internal import SCOPES, all_scope_files, scope_groups

    with _baseline() as baseline:
        scope_map = {scope.name: scope for scope in SCOPES}
        groups: list[tuple[str, list[tuple[str, int | None, int]]]] = []
        for title, names in scope_groups():
            rows = [
                (name, _score_scope(scope_map[name], baseline), _score_scope(scope_map[name], ROOT))
                for name in names
            ]
            groups.append((title, rows))
        try:
            old_overall: int | None = score_files(all_scope_files(baseline))
        except ValueError:
            old_overall = None
        current_overall = score_files(all_scope_files(ROOT))
        current_md = sum(
            len(path.read_text(encoding="utf-8"))
            for path in _markdown_files(ROOT, respect_gitignore=True)
        )
        old_md = sum(len(path.read_text(encoding="utf-8")) for path in _markdown_files(baseline))
    print("Benchmark scopes measure unique implementation cost;")
    print("shared helpers live in common, the rest in core.")
    print("Scope                         Main   Current       Delta")
    print("-" * 62)
    for title, rows in groups:
        print(title)
        for name, old, current in rows:
            print(
                f"  {name:26} {_format_count(old)} {current:8,} {_format_delta(old, current):>25}"
            )
    print(
        f"{'Total':28} {_format_count(old_overall)} {current_overall:8,} "
        f"{_format_delta(old_overall, current_overall):>25}"
    )
    print(
        f"  {'# chars in .md files':26} {old_md:7,} {current_md:8,} "
        f"{_signed(current_md - old_md, 10):>25}"
    )
    return 0


def snake_speed() -> int:
    current = measure_snake_speed(duration=0.1, runs=2, warmup=0.1)
    command = [
        sys.executable,
        "-c",
        "from minimal_nn_competition.speed.benchmark import measure_snake_speed; print(measure_snake_speed(duration=.1, runs=2, warmup=.1).ticks_per_second)",
    ]
    with _baseline() as baseline:
        old = float(subprocess.check_output(command, cwd=baseline, text=True).strip())
    change = (current.ticks_per_second - old) / old
    print(
        f"Snake speed: main {old:,.0f}, current {current.ticks_per_second:,.0f}, change {change:+.1%}"
    )
    return int(change < -SPEED_TOLERANCE)


def _files_to_scan() -> list[Path]:
    tracked = _git("ls-files").splitlines()
    return [
        ROOT / item
        for item in tracked
        if Path(item).suffix in TEXT_EXTENSIONS or Path(item).name in SPECIAL_FILES
    ]


def _display_path(path: Path) -> str:
    return str(path).removeprefix(f"{ROOT}/")


def char_encoding() -> int:
    findings = []
    for path in _files_to_scan():
        raw = path.read_bytes()
        if raw.startswith(b"\xef\xbb\xbf"):
            findings.append(f"{_display_path(path)}: UTF-8 BOM")
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            findings.append(f"{_display_path(path)}: invalid UTF-8 at byte {exc.start}")
            continue
        for line, content in enumerate(text.splitlines(), 1):
            for column, char in enumerate(content, 1):
                if ord(char) < 128:
                    continue
                category = unicodedata.category(char)
                codepoint = ord(char)
                dangerous = char != " " and (
                    category in {"So", "Cf", "Co", "Cn", "Cs", "Zs"}
                    or codepoint in {0, 27, 127, 0x2028, 0x2029}
                    or 0x80 <= codepoint <= 0x9F
                )
                confusable = 0x0400 <= codepoint <= 0x04FF or 0xFF00 <= codepoint <= 0xFFEF
                box = 0x2500 <= codepoint <= 0x257F
                if (
                    (dangerous or confusable)
                    and not box
                    and not (path.parent.name in EVIDENCE_DIRECTORIES and category == "Zs")
                ):
                    findings.append(
                        f"{_display_path(path)}:{line}:{column}: U+{codepoint:04X} {unicodedata.name(char, 'UNKNOWN')}"
                    )
    for finding in findings:
        print(f"error: {finding}")
    print(f"char-encoding: {len(findings)} finding(s)")
    return int(bool(findings))


def _run(label: str, command: list[str]) -> int:
    print(f"== {label} ==")
    return subprocess.run(command, cwd=ROOT).returncode


def check() -> int:
    checks = [
        ("ruff format", ["uv", "run", "ruff", "format", "--check"]),
        ("ruff", ["uv", "run", "ruff", "check"]),
        ("ty", ["uv", "run", "ty", "check"]),
        ("pytest", ["uv", "run", "pytest"]),
    ]
    for label, command in checks:
        if _run(label, command):
            return 1
    for label, function in (
        ("tokens", tokens),
        ("snake-speed", snake_speed),
        ("char-encoding", char_encoding),
    ):
        print(f"== {label} ==")
        if function():
            return 1
    return _run("build", ["uv", "build"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dev-tools")
    parser.add_argument("command", choices=("check", "tokens", "snake-speed", "char-encoding"))
    args = parser.parse_args(argv)
    return {
        "check": check,
        "tokens": tokens,
        "snake-speed": snake_speed,
        "char-encoding": char_encoding,
    }[args.command]()


if __name__ == "__main__":
    raise SystemExit(main())
