"""
Thin wrapper to launch training with CSVs exported by scripts/prepare_rsna.py.

Example:
  python scripts/train_from_csv.py \
    --train_csv chapter4_outputs/rsna_prepared/rsna_train.csv \
    --val_csv   chapter4_outputs/rsna_prepared/rsna_val.csv \
    --test_csv  chapter4_outputs/rsna_prepared/rsna_test.csv \
    --epochs 10 --batch_size 32 --lr 1e-4
"""

import argparse
from pathlib import Path

from src.train import main as train_main


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out_dir", type=str, default="chapter4_outputs")
    return parser.parse_args()


def main():
    # Reuse src.train CLI by building argv
    args = parse_args()
    import sys
    sys.argv = [
        "src.train",
        "--train_csv", args.train_csv,
        "--val_csv", args.val_csv,
        "--test_csv", args.test_csv,
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--device", args.device,
        "--out_dir", args.out_dir,
    ]
    train_main()


if __name__ == "__main__":
    main()


