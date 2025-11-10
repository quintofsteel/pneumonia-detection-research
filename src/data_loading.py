"""
Data loading, preprocessing, and DataLoader builders.

Key expectations:
- Processed combined CSV must contain columns:
  - image_id, base_dir, label (0/1), and metadata columns listed in config.METADATA_FEATURES
- base_dir should be the absolute or relative path to the folder containing image file names in image_id
"""

import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.data._utils.collate import default_collate

from . import config

# Simple transform used by default
DEFAULT_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


class PneumoniaDataset(Dataset):
    """
    Dataset expects a dataframe with columns:
      - image_id : filename (may include subpaths relative to base_dir)
      - base_dir : directory where image_id can be found
      - label : 0 or 1
      - metadata columns (config.METADATA_FEATURES)
    """

    def __init__(self, df: pd.DataFrame, transform=None, metadata_features: Optional[List[str]] = None):
        self.df = df.reset_index(drop=True).copy()
        self.transform = transform if transform is not None else DEFAULT_TRANSFORM
        self.metadata_features = metadata_features or config.METADATA_FEATURES

        # Ensure metadata columns exist; if not, create with NaNs
        for c in self.metadata_features:
            if c not in self.df.columns:
                self.df[c] = np.nan

    def __len__(self):
        return len(self.df)
    

    def _resolve_path(self, base_dir: str, image_id: str) -> str:
        """
        Determine actual file path. Accepts image_id which may include subdirectories.
        Tries common image extensions if direct path not found.
        """
        candidate = os.path.join(base_dir, image_id)
        if os.path.exists(candidate):
            return candidate

        # Try typical extensions
        for ext in [".png", ".jpg", ".jpeg", ".dcm"]:
            if os.path.exists(candidate + ext):
                return candidate + ext

        # If image_id contains leading prefix that duplicates base_dir, strip it
        # e.g., image_id: "CheXpert-v1.0-small/train/..." and base_dir already points to /kaggle/input/chexpert
        if os.path.exists(os.path.join(base_dir, os.path.basename(image_id))):
            return os.path.join(base_dir, os.path.basename(image_id))

        raise FileNotFoundError(f"Image not found: base_dir={base_dir}, image_id={image_id}")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        base_dir = str(row.get("base_dir", ""))
        image_id = str(row["image_id"])

        try:
            img_path = self._resolve_path(base_dir, image_id)
        except FileNotFoundError:
            # Return None to allow filtering by custom collate
            return None

        # Load image (handle DICOM minimally if encountered)
        try:
            if img_path.lower().endswith(".dcm"):
                # Optional: pydicom reading can be added if needed
                import pydicom

                dicom = pydicom.dcmread(img_path)
                image = Image.fromarray(dicom.pixel_array).convert("RGB")
            else:
                image = Image.open(img_path).convert("RGB")
        except Exception:
            return None

        if self.transform:
            image = self.transform(image)

        # Metadata vector - ensure numeric dtype
        meta = []
        for f in self.metadata_features:
            v = row.get(f, np.nan)
            if pd.isna(v):
                meta.append(0.0)  # impute zero; upstream pipeline should have better handling
            else:
                meta.append(float(v))
        meta = torch.tensor(meta, dtype=torch.float32)

        label = torch.tensor(float(row["label"]), dtype=torch.float32)

        return image, meta, label, row.to_dict()



class PneumoniaDatasetMeta(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir # Note: root_dir is often not needed if df['image_path'] is already full path
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def _encode_meta(self, row):
        # Normalize age, use pre-encoded sex and view_position
        age = 0.0 if pd.isna(row.get('age')) else float(row.get('age')) / 100.0
        # Directly use the numerical value for sex (0 or 1)
        sex = float(row.get('sex', 0.0))
        # Directly use the numerical value for view_position (0 or 1)
        view = float(row.get('view_position', 0.0))
        return torch.tensor([age, sex, view], dtype=torch.float32)


    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        full_path = row['image_path'] # Use the full path

        # or None if loading failed.
        image_array = XRayProcessor.process_image(full_path, "") # Pass empty root_dir as full_path is used

        if image_array is None:
            # Skip missing or corrupt images by returning None
            print(f"[Warning] Skipping file due to processing error: {full_path}")
            return None

        image_tensor = torch.from_numpy(image_array).float() # Ensure float32

        # Ensure 3 channels if it's a 1-channel grayscale image
        if image_tensor.shape[0] == 1:
             image_tensor = image_tensor.repeat(3, 1, 1) # Shape changes from [1, H, W] to [3, H, W]

        # ---- Metadata vector ----
        meta = torch.tensor([
            row['age'] / 100.0 if not np.isnan(row['age']) else 0.5,
            row['sex'],
            row['view_position']
        ], dtype=torch.float32)

        label = torch.tensor(row['label'], dtype=torch.float32)

        group = {
            "sex": int(row['sex']),
            "view": row['view_position'],
        }

        # Re-apply the transform to the image tensor here:
        if self.transform:
             image_tensor = self.transform(image_tensor)


        return image_tensor, meta, label, group
    

def custom_collate(batch):
    """
    Filters out None samples returned by dataset (missing files or corrupt)
    and uses default_collate to assemble tensors.
    Returns tuple (images, metas, labels, rows) or raises ValueError if empty.
    """
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return default_collate(batch)


def make_dataloaders(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, batch_size=None, num_workers=None):
    batch_size = batch_size or config.BATCH_SIZE
    num_workers = num_workers if num_workers is not None else config.NUM_WORKERS

    train_ds = PneumoniaDataset(train_df)
    val_ds = PneumoniaDataset(val_df)
    test_ds = PneumoniaDataset(test_df)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=custom_collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=custom_collate)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=custom_collate)

    return train_loader, val_loader, test_loader
