"""Build the fixed chess-eval datasets from the Lichess CC0 evaluation database.

Run with:
    uv run --with duckdb python scripts/build_chess_eval_dataset.py --tier both
"""

import argparse
import hashlib
import json
import logging
import random
from datetime import UTC, datetime
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "src/minimal_nn_competition/benchmarks/data/chess_eval"
SOURCE = "https://huggingface.co/datasets/Lichess/chess-position-evaluations"
FILES = (0, 4, 8, 12, 16)
URL = f"{SOURCE}/resolve/main/data/data_%04d.parquet"
MIN_DEPTH = 30
MIN_KNODES = 500_000
MAX_ABS_CP = 1_000
DRAW_CP = 50
TIERS = {
    "10k": {
        "positions": 10_000,
        "bands": {"black": 3_333, "draw": 3_334, "white": 3_333},
        "seed": "chess-eval-v1",
        "superset": None,
        "shuffle_seed": 0,
    },
    "100k": {
        "positions": 100_000,
        "bands": {"black": 33_333, "draw": 33_334, "white": 33_333},
        "seed": "chess-eval-100k-v1",
        "superset": "10k",
        "shuffle_seed": 1,
    },
}

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
SELECT fen, cp, band, rn
FROM sampled
WHERE rn <= CASE band WHEN 'draw' THEN ? ELSE ? END
ORDER BY band, rn
"""


def band_for(cp):
    return "black" if cp < -DRAW_CP else "white" if cp > DRAW_CP else "draw"


def load_positions(path):
    positions = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        positions.append((str(item["fen"]), int(item["cp"])))
    return positions


def query_rows(connection, urls, tier):
    base = TIERS[tier["superset"]] if tier["superset"] else None
    base_bands = base["bands"] if base else {"black": 0, "draw": 0, "white": 0}
    draw_limit = tier["bands"]["draw"] + base_bands["draw"]
    other_limit = tier["bands"]["black"] + base_bands["black"]
    return connection.execute(
        SQL,
        [urls, MIN_DEPTH, MIN_KNODES, MAX_ABS_CP, tier["seed"], draw_limit, other_limit],
    ).fetchall()


def build_positions(rows, tier, base_positions=()):
    base_fens = {fen for fen, _ in base_positions}
    selected = list(base_positions)
    counts = {band: 0 for band in tier["bands"]}
    for _, cp in base_positions:
        counts[band_for(cp)] += 1
    for fen, cp, band, _ in rows:
        if cp is None or fen in base_fens:
            continue
        if band != band_for(cp):
            raise RuntimeError(f"Band mismatch for {fen}")
        if counts[band] >= tier["bands"][band]:
            continue
        selected.append((fen, int(cp)))
        counts[band] += 1
    return selected


def write_dataset(path, positions, metadata):
    path.write_text(
        "".join(json.dumps({"fen": fen, "cp": cp}, sort_keys=True) + "\n" for fen, cp in positions)
    )
    metadata["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_name(
        path.name.replace("positions-", "metadata-").replace(".jsonl", ".json")
    ).write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", choices=("10k", "100k", "both"), default="both")
    args = parser.parse_args()
    logger = logging.getLogger("chess-eval-dataset")
    urls = [URL % index for index in FILES]
    connection = duckdb.connect()
    connection.execute("INSTALL httpfs; LOAD httpfs;")
    tiers = ("10k", "100k") if args.tier == "both" else (args.tier,)
    for name in tiers:
        tier = TIERS[name]
        base_positions = ()
        if tier["superset"]:
            base_path = OUTPUT / f"positions-{tier['superset']}.jsonl"
            if not base_path.is_file():
                raise FileNotFoundError(
                    f"Build the {tier['superset']} tier first; missing {base_path}"
                )
            base_positions = load_positions(base_path)
        rows = query_rows(connection, urls, tier)
        positions = build_positions(rows, tier, base_positions)
        expected = tier["positions"]
        if len(positions) != expected:
            raise RuntimeError(f"Expected {expected} {name} positions, found {len(positions)}")
        if len({fen for fen, _ in positions}) != expected:
            raise RuntimeError(f"Duplicate FENs in {name} positions")
        if any(fen.count("/") != 7 for fen, _ in positions):
            raise RuntimeError(f"Malformed FEN in {name} positions")
        bands = {band: 0 for band in tier["bands"]}
        for _, cp in positions:
            bands[band_for(cp)] += 1
        if bands != tier["bands"]:
            raise RuntimeError(f"Unexpected {name} bands: {bands}")
        if tier["superset"]:
            missing = {fen for fen, _ in base_positions} - {fen for fen, _ in positions}
            if missing:
                raise RuntimeError(f"{name} does not contain the {tier['superset']} positions")
        positions.sort(key=lambda item: (band_for(item[1]), item[0]))
        random.Random(tier["shuffle_seed"]).shuffle(positions)
        path = OUTPUT / f"positions-{name}.jsonl"
        metadata = {
            "source": SOURCE,
            "source_files": urls,
            "license": "CC0-1.0",
            "retrieved": datetime.now(UTC).date().isoformat(),
            "positions": expected,
            "seed": tier["seed"],
            "filters": {
                "min_depth": MIN_DEPTH,
                "min_knodes": MIN_KNODES,
                "max_abs_cp": MAX_ABS_CP,
                "exclude_mate": True,
                "exclude_en_passant": True,
                "deduplicate_fen": True,
                "first_pv": "highest depth, best for the side to move",
            },
            "bands": tier["bands"],
            "draw_threshold_cp": DRAW_CP,
            "tolerance_pawns": 0.1,
            "input": {
                "size": 773,
                "encoding": "castling-v2",
                "castling_rights": ["K", "Q", "k", "q"],
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
        if tier["superset"]:
            metadata["superset_of"] = f"chess-eval-{tier['superset']}"
        OUTPUT.mkdir(parents=True, exist_ok=True)
        write_dataset(path, positions, metadata)
        logger.info(f"Wrote {len(positions)} {name} positions to {path}")


if __name__ == "__main__":
    main()
