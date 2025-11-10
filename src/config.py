"""
Central configuration for experiments.
Edit paths and hyperparameters here or override via CLI flags in train.py
"""

from pathlib import Path
import os

# Dataset roots (aligned with notebook defaults but overridable via env vars)
DATA_ROOT = Path(os.getenv("DATA_ROOT", "/kaggle/input"))
NIH_PATH = Path(os.getenv("NIH_PATH", str(DATA_ROOT / "data")))
CHEXPERT_PATH = Path(os.getenv("CHEXPERT_PATH", str(DATA_ROOT / "chexpert")))
RSNA_PATH = Path(os.getenv("RSNA_PATH", str(DATA_ROOT / "rsna-pneumonia-detection-challenge")))

# Output and experiment folders (default to notebook's chapter4_outputs)
OUTPUT_ROOT = Path(os.getenv("OUTPUT_ROOT", "chapter4_outputs"))
CHECKPOINT_DIR = OUTPUT_ROOT / "checkpoints"
RESULTS_DIR = OUTPUT_ROOT / "results"

# Model / training hyperparameters
SEED = 42
BATCH_SIZE = 32
NUM_EPOCHS = 10
LR = 1e-4
WEIGHT_DECAY = 1e-5
DEVICE = "cuda"  
NUM_WORKERS = 4

# Data / preprocessing
IMAGE_SIZE = 224
METADATA_FEATURES = ["age", "sex", "view_position"]  
CLASS_NAMES = ["no_pneumonia", "pneumonia"]

# Data prep defaults
MAX_FILES_TO_PROCESS = int(os.getenv("MAX_FILES_TO_PROCESS", "0"))  # 0 means no limit

# Checkpointing
SAVE_EVERY_N_EPOCHS = 1
EARLY_STOPPING_PATIENCE = 5

# Bootstrap settings for evaluation
BOOTSTRAP_SAMPLES = 1000
ALPHA = 0.05
