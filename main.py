import argparse

from data import OUTPUT_DIR, build_manifest, prepare_datasets
from engine import train_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train the pneumonia detection CNN")
    parser.add_argument("--epochs", type=int, default=100, help="Maximum number of training epochs")
    parser.add_argument("--patience", type=int, default=10, help="Epochs without validation improvement before early stopping")
    parser.add_argument("--seed", type=int, default=42, help="Seed number for reproducibility")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["original", "original_regrouped", "pool"],
        default="pool",
        help="Dataset split mode: 'original' (original splits), "
            "'original_regrouped' (80/20 train-val split), "
            "'pool' (pool all data and create new splits).",
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    manifest = build_manifest(args.mode, args.seed)

    file_prefix = f"{args.mode}_s{args.seed}"
    train_path = OUTPUT_DIR / f"{file_prefix}_epochs{args.epochs}_patience{args.patience}.pth"

    train, val, test = prepare_datasets(manifest)
    
    results = train_model(
        train, val, test,
        path=train_path,
        seed=args.seed,
        epochs=args.epochs,
        patience=args.patience,
    )
