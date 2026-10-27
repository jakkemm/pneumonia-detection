import re
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset
from torchvision import datasets

BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "data" / "chest_xray"
OUTPUT_DIR = BASE_DIR / "outputs"
SPLIT_DIR = BASE_DIR / "splits"

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
        rows += [(str(Path(p).relative_to(BASE_DIR)), y, _patient_id(p), s) for p, y in ds.samples]
    
    df = pd.DataFrame(rows, columns=["path", "label", "patient", "split"])
    df.to_csv(SPLIT_DIR / "original.csv", index=False)
    return df

def _build_pooled_split_df(seed=42):
    df = _build_original_split_df()
    
    splitter = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
    folds = [i for _, i in splitter.split(df, df.label, df.patient)]
    
    out = df.copy()
    out["split"] = "train"
    out.loc[folds[0], "split"] = "test"
    out.loc[folds[1], "split"] = "val"
    
    out.to_csv(SPLIT_DIR / f"pooled_s{seed}.csv", index=False)

def load_split(manifest, split, transform, frac=1.0, seed=42):
    df = pd.read_csv(manifest)
    df = df[df.split == split]

    if frac < 1.0:
        df = (
            df.groupby("label", group_keys=False)
            .sample(frac=frac, random_state=seed)
            .reset_index(drop=True)
        )

    return ManifestDataset(df, transform)

if __name__ == "__main__":
    _build_pooled_split_df()