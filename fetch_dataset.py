from pathlib import Path

import kagglehub

from utils import _build_original_split_df

DATA_DIR = Path.cwd() / "data"
SPLITS_DIR = Path.cwd() / "splits"
DATA_DIR.mkdir(parents=True, exist_ok=True)
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    path = kagglehub.dataset_download(
        handle="paultimothymooney/chest-xray-pneumonia",
        output_dir=str(DATA_DIR),
        force_download=True,
    )

    print("Path to dataset files:", path)
    
    _build_original_split_df()
    print("Built a manifest for original train-val-test split.")