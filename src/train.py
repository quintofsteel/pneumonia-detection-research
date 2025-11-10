"""
Training script (entrypoint). Usage (example):
python -m src.train --config src/config.py --epochs 10 --batch_size 32
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from tqdm import tqdm

from . import config
from .utils import set_seed, ensure_dirs, save_checkpoint
from .data_loading import make_dataloaders
from .model_architecture import FusionModel

from .evaluate import evaluate_model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, required=True, help="CSV with train rows (image_id, base_dir, label, metadata)")
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LR)
    parser.add_argument("--device", type=str, default=config.DEVICE)
    parser.add_argument("--metadata-dim", type=int, default=len(config.METADATA_FEATURES))
    parser.add_argument("--out_dir", type=str, default=str(config.OUTPUT_ROOT))
    return parser.parse_args()


def load_csv(path):
    df = pd.read_csv(path)
    # Important: ensure required columns exist
    required = ["image_id", "base_dir", "label"] + config.METADATA_FEATURES
    for c in required:
        if c not in df.columns:
            df[c] = pd.NA
    return df


def train_loop(model, optimizer, criterion, train_loader, device):
    model.train()
    running_loss = 0.0
    n = 0
    for batch in train_loader:
        if batch is None:
            continue
        images, metas, labels, _ = batch
        images = images.to(device)
        metas = metas.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        probs, logits = model(images, metas)
        loss = criterion(probs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        n += images.size(0)
    return running_loss / max(1, n)


def main():
    args = parse_args()
    set_seed()
    ensure_dirs()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_df = load_csv(args.train_csv)
    val_df = load_csv(args.val_csv)
    test_df = load_csv(args.test_csv)

    train_loader, val_loader, test_loader = make_dataloaders(train_df, val_df, test_df, batch_size=args.batch_size)

    model = FusionModel(metadata_dim=args.metadata_dim, pretrained_image=True)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=config.WEIGHT_DECAY)
    # Use BCELoss with probs; alternatively use BCEWithLogitsLoss if returning logits
    criterion = nn.BCELoss()

    best_auc = -1.0
    epochs_no_improve = 0

    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss = train_loop(model, optimizer, criterion, train_loader, device)
        # Evaluate on validation
        val_stats = evaluate_model(model, val_loader, device=device)
        val_auc = val_stats["metrics"]["auc"]

        elapsed = time.time() - t0
        print(f"Epoch {epoch+1}/{args.epochs} | train_loss: {train_loss:.4f} | val_auc: {val_auc:.4f} | time: {elapsed:.1f}s")

        # checkpoint
        ckpt_name = f"epoch{epoch+1:02d}_valauc{val_auc:.4f}.pt"
        save_checkpoint({
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_auc": val_auc,
        }, filename=ckpt_name)

        # early stopping logic
        if val_auc > best_auc:
            best_auc = val_auc
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping after {epoch+1} epochs.")
                break

    # Final evaluation on test set
    print("Running final evaluation on test set...")
    test_stats = evaluate_model(model, test_loader, device=device)
    print("Test metrics:", test_stats["metrics"])
    # Save final metrics
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    import json
    with open(Path(args.out_dir) / "test_metrics.json", "w") as f:
        json.dump(test_stats["metrics"], f, indent=2)


if __name__ == "__main__":
    main()
