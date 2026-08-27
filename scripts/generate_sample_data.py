#!/usr/bin/env python
"""
Regenerate the synthetic sample dataset used by the NETRA demo.

Writes ``persons.csv``, ``relationships.csv``, ``crime_records.csv`` and
``documents.csv`` into the output directory (default ``data/samples/``). The generator
is fully seeded, so the same ``--seed`` always produces byte-identical files.

Uploading all four regenerated files reproduces the bundled sample analysis exactly,
which is what makes them usable as the worked example in ``docs/DATA_SCHEMA.md``.
Leave ``documents.csv`` out and the Entity Extraction tab has nothing to work on.

Examples
--------
    python scripts/generate_sample_data.py
    python scripts/generate_sample_data.py --persons 300 --syndicates 7 --seed 7
    python scripts/generate_sample_data.py --out data/samples --persons 220 --seed 42
"""

from __future__ import annotations

import argparse
import pathlib
import sys

# Make ``src`` importable when this script is run directly from the repo root.
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ingestion.sample_data import generate_sample_dataset  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Generate NETRA synthetic sample CSVs "
                    "(persons / relationships / crimes / documents).")
    ap.add_argument("--persons", type=int, default=220,
                    help="Number of actors in the network (40–2000). Default: 220.")
    ap.add_argument("--syndicates", type=int, default=6,
                    help="Number of hierarchical syndicates/modules (2–7). Default: 6.")
    ap.add_argument("--seed", type=int, default=42,
                    help="RNG seed — same seed yields the same network. Default: 42.")
    ap.add_argument("--out", type=pathlib.Path, default=ROOT / "data" / "samples",
                    help="Output directory. Default: data/samples/.")
    args = ap.parse_args(argv)

    out: pathlib.Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    print(f"Generating synthetic network — {args.persons} actors, "
          f"{args.syndicates} syndicates, seed {args.seed} …")
    data = generate_sample_dataset(
        n_persons=args.persons, n_syndicates=args.syndicates, seed=args.seed)

    written = []
    for name, df in data.items():
        path = out / f"{name}.csv"
        df.to_csv(path, index=False)
        written.append((path, len(df), len(df.columns)))

    print(f"\nWrote {len(written)} files to {out}{pathlib.os.sep}")
    for path, rows, cols in written:
        print(f"  • {path.name:20} {rows:>5} rows × {cols} cols")
    print("\nDone. Launch the app with:  streamlit run app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
