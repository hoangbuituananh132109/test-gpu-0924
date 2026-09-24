"""Deterministic offline DAPO train-only subset; never pass an eval parquet."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--rows", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Refusing to overwrite {args.output}")
    if any(part.lower() in {"aime", "amc", "math500", "test_data"} for part in args.source.parts):
        parser.error("Source path looks like evaluation data")
    table = pq.read_table(args.source)
    if not {"data_source", "prompt", "reward_model"}.issubset(table.column_names):
        parser.error("Source does not have the expected train schema")
    if not 0 < args.rows <= table.num_rows:
        parser.error(f"Requested {args.rows} rows from {table.num_rows}")
    indices = np.random.default_rng(args.seed).permutation(table.num_rows)[: args.rows]
    subset = table.take(pa.array(indices))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(subset, args.output)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(f"rows={args.rows} seed={args.seed} sha256={digest} output={args.output}")


if __name__ == "__main__":
    main()
