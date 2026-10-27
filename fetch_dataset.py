from pathlib import Path

import kagglehub

DATA_DIR = Path.cwd() / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    path = kagglehub.dataset_download(
        handle="paultimothymooney/chest-xray-pneumonia",
        output_dir=str(DATA_DIR),
        force_download=True,
    )

    print("Path to dataset files:", path)