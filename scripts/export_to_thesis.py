"""
Copy/sync key artifacts to thesis folders for submission.

Defaults assume OUTPUT_ROOT = "chapter4_outputs" per src/config.py and
will place figures into thesis/figures and tabular metrics into thesis/tables.

Usage:
  python scripts/export_to_thesis.py \
    --output_root chapter4_outputs \
    --thesis_dir thesis
"""

import argparse
import json
import shutil
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_root", type=str, default="chapter4_outputs")
    p.add_argument("--thesis_dir", type=str, default="thesis")
    return p.parse_args()


def main():
    args = parse_args()
    out_root = Path(args.output_root)
    thesis_dir = Path(args.thesis_dir)
    figs_dir = thesis_dir / "figures"
    tables_dir = thesis_dir / "tables"
    figs_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Copy any PNG/JPG under results to figures
    results_dir = out_root / "results"
    if results_dir.exists():
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            for fp in results_dir.rglob(ext):
                dest = figs_dir / fp.name
                shutil.copy2(fp, dest)

    # Copy metrics json if present
    metrics_json = out_root / "test_metrics.json"
    if metrics_json.exists():
        shutil.copy2(metrics_json, tables_dir / "test_metrics.json")

    print("Exported artifacts to:", figs_dir, "and", tables_dir)


if __name__ == "__main__":
    main()


