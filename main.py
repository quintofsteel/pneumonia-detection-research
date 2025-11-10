"""
Project entrypoint to showcase a full demo and reproduce results.

Subcommands:
  - prepare:  RSNA DICOM→PNG + metadata/CSV, matching the notebook
  - train:    Train FusionModel on exported CSVs
  - eval:     Evaluate a trained model on test split and write metrics
  - export:   Copy figures/metrics into thesis folder
  - all:      Run prepare → train → eval → export

Quick demo example (PowerShell/CMD):
  python main.py all \
    --rsna_dir "/kaggle/input/rsna-pneumonia-detection-challenge" \
    --out_dir "chapter4_outputs/rsna_prepared" \
    --epochs 10 --batch_size 32 --lr 1e-4
"""

import argparse
from pathlib import Path
import sys

from scripts import prepare_rsna as prep
from scripts import export_to_thesis as export_mod

from src.train import main as train_main
from src import config
from src.data_loading import make_dataloaders
from src.model_architecture import FusionModel
from src.evaluate import evaluate_model

import pandas as pd
import torch


def cmd_prepare(rsna_dir: str, out_dir: str, limit: int) -> None:
    rsna_root = Path(rsna_dir)
    out_root = Path(out_dir)
    dicom_dir = rsna_root / "stage_2_train_images"
    png_dir = out_root / "rsna_png_images"

    out_root.mkdir(parents=True, exist_ok=True)

    print("[prepare] Extracting DICOM metadata...")
    meta_df = prep.extract_metadata(dicom_dir, limit)
    (out_root / "rsna_metadata.csv").write_text(meta_df.to_csv(index=False))

    print("[prepare] Converting DICOM → PNG...")
    n_png = prep.convert_dicoms(dicom_dir, png_dir, limit)
    print(f"[prepare] Converted {n_png} PNGs → {png_dir}")

    print("[prepare] Building classification CSV...")
    cls_df = prep.build_classification_csv(rsna_root, meta_df, png_dir)
    (out_root / "rsna_classification.csv").write_text(cls_df.to_csv(index=False))

    print("[prepare] Creating splits (80/10/10)...")
    splits = prep.stratified_split(cls_df, seed=42)
    for name, df in splits.items():
        (out_root / f"rsna_{name}.csv").write_text(df.to_csv(index=False))

    print("[prepare] Done →", out_root)


def cmd_train(train_csv: str, val_csv: str, test_csv: str, epochs: int, batch_size: int, lr: float, device: str, out_dir: str) -> None:
    # Reuse src.train CLI by setting sys.argv
    sys.argv = [
        "src.train",
        "--train_csv", train_csv,
        "--val_csv", val_csv,
        "--test_csv", test_csv,
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--lr", str(lr),
        "--device", device,
        "--out_dir", out_dir,
    ]
    train_main()


def cmd_eval(test_csv: str, device: str) -> None:
    device_t = torch.device(device if torch.cuda.is_available() else "cpu")
    df_test = pd.read_csv(test_csv)
    _, _, test_loader = make_dataloaders(df_test, df_test.iloc[0:0], df_test)
    model = FusionModel(metadata_dim=len(config.METADATA_FEATURES), pretrained_image=True).to(device_t)
    stats = evaluate_model(model, test_loader, device=device_t)
    print("[eval] Test metrics:", stats["metrics"])


def cmd_export(output_root: str, thesis_dir: str) -> None:
    sys.argv = [
        "export_to_thesis",
        "--output_root", output_root,
        "--thesis_dir", thesis_dir,
    ]
    export_mod.main()


def parse_args():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    # prepare
    sp = sub.add_parser("prepare")
    sp.add_argument("--rsna_dir", type=str, required=True)
    sp.add_argument("--out_dir", type=str, default=str(config.OUTPUT_ROOT / "rsna_prepared"))
    sp.add_argument("--limit", type=int, default=0)

    # train
    st = sub.add_parser("train")
    st.add_argument("--train_csv", type=str, required=True)
    st.add_argument("--val_csv", type=str, required=True)
    st.add_argument("--test_csv", type=str, required=True)
    st.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    st.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    st.add_argument("--lr", type=float, default=config.LR)
    st.add_argument("--device", type=str, default=config.DEVICE)
    st.add_argument("--out_dir", type=str, default=str(config.OUTPUT_ROOT))

    # eval
    se = sub.add_parser("eval")
    se.add_argument("--test_csv", type=str, required=True)
    se.add_argument("--device", type=str, default=config.DEVICE)

    # export
    sx = sub.add_parser("export")
    sx.add_argument("--output_root", type=str, default=str(config.OUTPUT_ROOT))
    sx.add_argument("--thesis_dir", type=str, default="thesis")

    # all
    sa = sub.add_parser("all")
    sa.add_argument("--rsna_dir", type=str, required=True)
    sa.add_argument("--out_dir", type=str, default=str(config.OUTPUT_ROOT / "rsna_prepared"))
    sa.add_argument("--limit", type=int, default=0)
    sa.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    sa.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    sa.add_argument("--lr", type=float, default=config.LR)
    sa.add_argument("--device", type=str, default=config.DEVICE)
    sa.add_argument("--export_to_thesis", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.cmd == "prepare":
        cmd_prepare(args.rsna_dir, args.out_dir, args.limit)
    elif args.cmd == "train":
        cmd_train(args.train_csv, args.val_csv, args.test_csv, args.epochs, args.batch_size, args.lr, args.device, args.out_dir)
    elif args.cmd == "eval":
        cmd_eval(args.test_csv, args.device)
    elif args.cmd == "export":
        cmd_export(args.output_root, args.thesis_dir)
    elif args.cmd == "all":
        prep_out = Path(args.out_dir)
        cmd_prepare(args.rsna_dir, args.out_dir, args.limit)
        train_csv = str(prep_out / "rsna_train.csv")
        val_csv = str(prep_out / "rsna_val.csv")
        test_csv = str(prep_out / "rsna_test.csv")
        cmd_train(train_csv, val_csv, test_csv, args.epochs, args.batch_size, args.lr, args.device, str(config.OUTPUT_ROOT))
        cmd_eval(test_csv, args.device)
        if args.export_to_thesis:
            cmd_export(str(config.OUTPUT_ROOT), "thesis")
    else:
        raise ValueError(f"Unknown cmd: {args.cmd}")


if __name__ == "__main__":
    main()




