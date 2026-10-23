"""Build the fixed chess-eval dataset from the Lichess CC0 evaluation database.

Run with:
    uv run --with duckdb python scripts/build_chess_eval_dataset.py
"""

import hashlib
import json
import logging
import random
from datetime import UTC, datetime
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "src/minimal_nn_competition/benchmarks/chess_eval"
SOURCE = "https://huggingface.co/datasets/Lichess/chess-position-evaluations"
FILES = (0, 4, 8, 12, 16)
URL = f"{SOURCE}/resolve/main/data/data_%04d.parquet"
MIN_DEPTH = 30
MIN_KNODES = 500_000
MAX_ABS_CP = 1_000
DRAW_CP = 50
SEED = "chess-eval-v1"
BANDS = {"black": 3_333, "draw": 3_334, "white": 3_333}

SQL = f"""
WITH filtered AS (
    SELECT fen, cp, depth,
           CASE WHEN cp < -{DRAW_CP} THEN 'black'
                WHEN cp > {DRAW_CP} THEN 'white'
                ELSE 'draw' END AS band,
           CASE WHEN split_part(fen, ' ', 2) = 'w' THEN cp ELSE -cp END AS mover_cp
    FROM read_parquet(?)
    WHERE depth >= ?
      AND knodes >= ?
      AND mate IS NULL
      AND cp IS NOT NULL
      AND abs(cp) <= ?
      AND split_part(fen, ' ', 4) = '-'
),
ranked AS (
    SELECT fen, cp, band,
           row_number() OVER (
               PARTITION BY fen
               ORDER BY depth DESC, mover_cp DESC
           ) AS rn
    FROM filtered
),
unique_positions AS (SELECT fen, cp, band FROM ranked WHERE rn = 1),
sampled AS (
    SELECT fen, cp, band,
           row_number() OVER (
               PARTITION BY band
               ORDER BY hash(fen || ?)
           ) AS rn
    FROM unique_positions
)
SELECT fen, cp, band
FROM sampled
WHERE rn <= CASE band WHEN 'draw' THEN ? ELSE ? END
ORDER BY band, fen
"""


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logger = logging.getLogger("chess-eval-dataset")
    urls = [URL % index for index in FILES]
    connection = duckdb.connect()
    connection.execute("INSTALL httpfs; LOAD httpfs;")
    rows = connection.execute(
        SQL,
        [urls, MIN_DEPTH, MIN_KNODES, MAX_ABS_CP, SEED, BANDS["draw"], BANDS["black"]],
    ).fetchall()
    positions = [(str(fen), int(cp)) for fen, cp, _ in rows]
    expected = sum(BANDS.values())
    if len(positions) != expected:
        raise RuntimeError(f"Expected {expected} positions, found {len(positions)}")
    if len({fen for fen, _ in positions}) != expected:
        raise RuntimeError("Duplicate FENs in sampled positions")
    if any(fen.count("/") != 7 for fen, _ in positions):
        raise RuntimeError("Malformed FEN in sampled positions")
    random.Random(0).shuffle(positions)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "positions.jsonl"
    path.write_text(
        "".join(json.dumps({"fen": fen, "cp": cp}, sort_keys=True) + "\n" for fen, cp in positions)
    )
    metadata = {
        "source": SOURCE,
        "source_files": urls,
        "license": "CC0-1.0",
        "retrieved": datetime.now(UTC).date().isoformat(),
        "positions": expected,
        "seed": SEED,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "filters": {
            "min_depth": MIN_DEPTH,
            "min_knodes": MIN_KNODES,
            "max_abs_cp": MAX_ABS_CP,
            "exclude_mate": True,
            "exclude_en_passant": True,
            "deduplicate_fen": True,
            "first_pv": "highest depth, best for the side to move",
        },
        "bands": BANDS,
        "draw_threshold_cp": DRAW_CP,
        "tolerance_pawns": 0.1,
        "input": {
            "size": 769,
            "planes": [
                "white P",
                "white N",
                "white B",
                "white R",
                "white Q",
                "white K",
                "black P",
                "black N",
                "black B",
                "black R",
                "black Q",
                "black K",
            ],
            "side_to_move": "white=1, black=-1",
            "squares": "FEN order: a8=0, h8=7, a1=56, h1=63",
        },
        "output": "white-perspective pawns (Stockfish cp / 100)",
    }
    (OUTPUT / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    logger.info(f"Wrote {len(positions)} positions to {path}")


if __name__ == "__main__":
    main()
