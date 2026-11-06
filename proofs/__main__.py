"""Command line entry point for the proof artifacts."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from proofs.pipeline import (
    ProofError,
    check,
    check_lean,
    generate,
    generate_lean,
)

ROOT = Path(__file__).resolve().parent.parent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m proofs")
    parser.add_argument(
        "command",
        choices=("generate", "check", "generate-lean", "check-lean"),
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        {
            "generate": generate,
            "check": check,
            "generate-lean": generate_lean,
            "check-lean": check_lean,
        }[args.command](ROOT)
    except ProofError as exc:
        logging.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
