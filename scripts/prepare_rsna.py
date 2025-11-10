"""
Prepare RSNA Pneumonia dataset to match the notebook pipeline:
 - Extract DICOM metadata (age, sex, view_position)
 - Convert DICOM to 8-bit PNG (grayscale)
 - Build classification CSV with columns: image_id, base_dir, label, age, sex, view_position
 - Create train/val/test CSV splits (80/10/10, stratified by label if possible)

Usage (PowerShell / bash):
  python scripts/prepare_rsna.py \
    --rsna_dir "/kaggle/input/rsna-pneumonia-detection-challenge" \
    --out_dir "chapter4_outputs/rsna_prepared" \
    --limit 0

Notes:
 - Sex mapping: {M:0, F:1}; ViewPosition mapping: {AP:0, PA:1}
 - Age kept numeric; NaNs preserved
 - Limit controls max number of DICOMs converted (0 = no limit).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rsna_dir", type=str, required=True, help="Path to RSNA challenge root with DICOMs and CSV labels")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for PNGs and CSVs")
    parser.add_argument("--limit", type=int, default=0, help="Max DICOMs to process (0 = all)")
    parser.add_argument("--png_dirname", type=str, default="rsna_png_images", help="Subfolder name for converted PNGs")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def safe_convert_to_png(dicom, output_path: Path) -> bool:
    try:
        pixel_array = dicom.pixel_array
        pixel_array = pixel_array - np.min(pixel_array)
        denom = np.max(pixel_array)
        if denom > 0:
            pixel_array = (pixel_array / denom) * 255.0
        pixel_array = pixel_array.astype(np.uint8)
        Image.fromarray(pixel_array).convert("L").save(output_path, "PNG")
        return True
    except Exception:
        # On failure, write a black placeholder to keep row alignment deterministic
        Image.fromarray(np.zeros((224, 224), dtype=np.uint8)).save(output_path, "PNG")
        return False


def extract_metadata(dicom_dir: Path, limit: int) -> pd.DataFrame:
    import pydicom  # defer import

    rows: List[Dict] = []
    filenames = sorted([f for f in os.listdir(dicom_dir) if f.endswith(".dcm")])
    processed = 0
    for fname in filenames:
        if limit and processed >= limit:
            break
        patient_id = fname[:-4]
        fp = dicom_dir / fname
        try:
            ds = pydicom.dcmread(str(fp), force=True)
            age = getattr(ds, "PatientAge", None)
            sex = getattr(ds, "PatientSex", None)
            view = getattr(ds, "ViewPosition", None)

            if isinstance(age, str):
                age = age.replace("Y", "")
                try:
                    age = int(age)
                except ValueError:
                    age = None

            rows.append({
                "patientId": patient_id,
                "age": age,
                "sex": sex,
                "view_position": view,
            })
            processed += 1
        except Exception:
            continue
    return pd.DataFrame(rows)


def convert_dicoms(dicom_dir: Path, png_dir: Path, limit: int) -> int:
    import pydicom

    png_dir.mkdir(parents=True, exist_ok=True)
    filenames = sorted([f for f in os.listdir(dicom_dir) if f.endswith(".dcm")])
    processed = 0
    for fname in filenames:
        if limit and processed >= limit:
            break
        patient_id = fname[:-4]
        fp = dicom_dir / fname
        out_fp = png_dir / f"{patient_id}.png"
        try:
            ds = pydicom.dcmread(str(fp), force=True)
            if safe_convert_to_png(ds, out_fp):
                processed += 1
        except Exception:
            continue
    return processed


def build_classification_csv(rsna_root: Path, metadata_df: pd.DataFrame, png_dir: Path) -> pd.DataFrame:
    labels_csv = rsna_root / "stage_2_train_labels.csv"
    if not labels_csv.exists():
        raise FileNotFoundError(f"Missing labels CSV: {labels_csv}")
    rsna_df = pd.read_csv(labels_csv)

    # Aggregate labels to image level (patientId): label = max(Target)
    rsna_df = rsna_df.rename(columns={"Target": "label"})
    rsna_agg = rsna_df.groupby("patientId").agg({"label": "max"}).reset_index()

    # Merge with metadata (left on aggregated labels for classification rows)
    merged = pd.merge(rsna_agg, metadata_df, on="patientId", how="left")

    # Encode metadata as in notebook
    merged["sex"] = merged["sex"].map({"M": 0, "F": 1}).astype("Int64")
    merged["view_position"] = merged["view_position"].map({"AP": 0, "PA": 1}).astype("Int64")

    # Define image paths
    merged["image_id"] = merged["patientId"].astype(str) + ".png"
    merged["base_dir"] = str(png_dir)

    # Select and order columns
    out = merged[["image_id", "base_dir", "label", "age", "sex", "view_position"]].copy()
    # Ensure numeric types where appropriate
    out["label"] = out["label"].astype(int)
    return out


def stratified_split(df: pd.DataFrame, seed: int) -> Dict[str, pd.DataFrame]:
    # Simple 80/10/10 split; stratify by label if both classes present
    rng = np.random.default_rng(seed)
    if df["label"].nunique() > 1:
        # Stratify
        train_idx, val_idx, test_idx = [], [], []
        for y, sub in df.groupby("label"):
            idx = rng.permutation(sub.index.values)
            n = len(idx)
            n_train = int(0.8 * n)
            n_val = int(0.1 * n)
            train_idx.extend(idx[:n_train])
            val_idx.extend(idx[n_train:n_train + n_val])
            test_idx.extend(idx[n_train + n_val:])
        return {
            "train": df.loc[train_idx].reset_index(drop=True),
            "val": df.loc[val_idx].reset_index(drop=True),
            "test": df.loc[test_idx].reset_index(drop=True),
        }
    else:
        idx = rng.permutation(df.index.values)
        n = len(idx)
        n_train = int(0.8 * n)
        n_val = int(0.1 * n)
        return {
            "train": df.loc[idx[:n_train]].reset_index(drop=True),
            "val": df.loc[idx[n_train:n_train + n_val]].reset_index(drop=True),
            "test": df.loc[idx[n_train + n_val:]].reset_index(drop=True),
        }


def main():
    args = parse_args()
    rsna_root = Path(args.rsna_dir)
    out_root = Path(args.out_dir)
    dicom_dir = rsna_root / "stage_2_train_images"
    png_dir = out_root / args.png_dirname

    out_root.mkdir(parents=True, exist_ok=True)

    print("[1/4] Extracting DICOM metadata...")
    meta_df = extract_metadata(dicom_dir, args.limit)
    meta_df.to_csv(out_root / "rsna_metadata.csv", index=False)

    print("[2/4] Converting DICOM → PNG...")
    n_png = convert_dicoms(dicom_dir, png_dir, args.limit)
    print(f"Converted {n_png} PNGs to {png_dir}")

    print("[3/4] Building classification CSV...")
    cls_df = build_classification_csv(rsna_root, meta_df, png_dir)
    cls_df.to_csv(out_root / "rsna_classification.csv", index=False)

    print("[4/4] Creating splits (80/10/10)...")
    splits = stratified_split(cls_df, args.seed)
    for name, split_df in splits.items():
        split_df.to_csv(out_root / f"rsna_{name}.csv", index=False)

    print("Done. Outputs in:", out_root)


if __name__ == "__main__":
    main()


