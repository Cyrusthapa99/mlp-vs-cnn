import os, sys, json, numpy as np, torch
from data_prep import *
torch.set_num_threads(1)
# Kaggle layout:  DATA/<ClassId>/*.png , labels.csv , TEST/*.png   (override with environment variables)
ROOTS = os.environ.get("DATA_DIRS", "DATA").split(",")          # comma-separated folders that contain <ClassId>/ sub-folders
LABELS = os.environ.get("LABELS_CSV", "labels.csv")
TEST_DIR = os.environ.get("TEST_DIR", "TEST")                    # supplied TEST folder (robustness check only)
def get_data():
    df = group_split(dedupe(build_metadata(ROOTS, LABELS)), seed=42)
    cache = "cache_64.npz"
    if os.path.exists(cache): X = np.load(cache)["X"]
    else: X = load_array(df.path.tolist(), 64); np.savez_compressed(cache, X=X)
    X = torch.tensor(X).permute(0, 3, 1, 2).float() / 255; y = torch.tensor(df.cls.values)
    sp = {k: np.where(df.split.values == k)[0] for k in ["train", "val", "test"]}
    return df, X, y, sp
