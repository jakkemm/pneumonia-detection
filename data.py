import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from torch.utils.data import Dataset, Subset
from torchvision import datasets

from model import create_transforms

BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "data" / "chest_xray"
OUTPUT_DIR = BASE_DIR / "outputs"
SPLIT_DIR = BASE_DIR / "splits"
ORIGINAL_SPLIT = SPLIT_DIR / "original.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SPLIT_DIR.mkdir(parents=True, exist_ok=True)

_IM_RE = re.compile(r"IM-(\d+)")
_PERSON_RE = re.compile(r"^(person\d+)_")

SPLITS = ("train", "val", "test")


class ManifestDataset(Dataset):
    def __init__(self, df, transform):
        self.paths = df["path"].astype(str).tolist()
        self.labels = df["label"].astype(int).tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        with Image.open(self.paths[idx]) as img:
            image = img.convert("RGB")

        image = self.transform(image)
        return image, self.labels[idx]

def _patient_id(path):
    """
    person99_bacteria_474   -> "person99"
    NORMAL2-IM-0381-0001    -> "IM-0381"
    IM-0011-0001-0002       -> "IM-0011"
    IM-0115-0001            -> "IM-0115"
    """
    stem = Path(path).stem

    m = _PERSON_RE.match(stem)
    if m:
        return m.group(1)
    m = _IM_RE.search(stem)
    if m:
        return f"IM-{m.group(1)}"
    raise ValueError(f"Cannot extract patient ID from {path}")

def _build_original_split_df():
    rows = []
    for s in SPLITS:
        ds = datasets.ImageFolder(DATA_DIR / s)
        rows += [(str(Path(p).relative_to(BASE_DIR)), y, _patient_id(p), s, s) for p, y in ds.samples]
    
    df = pd.DataFrame(rows, columns=["path", "label", "patient", "origin", "split"])
    path = SPLIT_DIR / "original.csv"
    df.to_csv(path, index=False)
    return path

def _build_original_regrouped_split_df(seed=42):
    path = _build_original_split_df()
    df = pd.read_csv(path)
    
    out = df.copy()
    out["split"] = np.where(out.origin == "test", "test", "train")
    dev = out[out.origin != "test"]
    
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    _, val_pos = next(splitter.split(dev, dev.label, dev.patient))
    
    out.loc[dev.index[val_pos], "split"] = "val"
    path = SPLIT_DIR / f"original_regrouped_s{seed}.csv"
    out.to_csv(path, index=False)
    return path

def _build_pooled_split_df(seed=42):
    path = _build_original_split_df()
    df = pd.read_csv(path)
    
    splitter = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
    folds = [i for _, i in splitter.split(df, df.label, df.patient)]
    
    out = df.copy()
    out["split"] = "train"
    out.loc[folds[0], "split"] = "test"
    out.loc[folds[1], "split"] = "val"
    
    path = SPLIT_DIR / f"pooled_s{seed}.csv"
    out.to_csv(path, index=False)
    return path

def build_manifest(mode, seed):    
    if mode == "original":
        return _build_original_split_df()
    if mode == "original_regrouped":
        return _build_original_split_df(seed)
    if mode == "pool":
        return _build_pooled_split_df(seed)

def load_split(manifest, split, transform):
    df = pd.read_csv(manifest)
    df = df[df.split == split]
    return ManifestDataset(df, transform)

def prepare_datasets(manifest):
    train_tf, eval_tf = create_transforms()
    train_ds = load_split(manifest, split="train", transform=train_tf)
    val_ds = load_split(manifest, split="val", transform=eval_tf)
    test_ds = load_split(manifest, split="test", transform=eval_tf)
    return train_ds, val_ds, test_ds

def shrink_train(train_ds, frac, seed=42):
    indices = np.arange(len(train_ds))
    labels = np.array(train_ds.labels)

    subset_idx, _ = train_test_split(indices, train_size=frac, stratify=labels, random_state=seed)
    train_ds = Subset(train_ds, subset_idx)

    return train_ds

if __name__ == "__main__":
    _build_original_regrouped_split_df(42)