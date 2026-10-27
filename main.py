import argparse

from train import train_full
from tune import tune_full
from utils import (
    _build_original_split_df,
    _build_pooled_split_df,
    prepare_datasets,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train the pneumonia detection CNN")
    parser.add_argument("--epochs", type=int, default=100, help="Maximum number of training epochs")
    parser.add_argument("--patience", type=int, default=10, help="Epochs without validation improvement before early stopping")
    parser.add_argument("--epochs_tuning", type=int, default=5, help="Number of epochs in hyperparameter tuning")
    parser.add_argument("--n_trials", type=int, default=25, help="Number of configurations to test for in hyperparameter tuning")
    parser.add_argument("--seed", type=int, default=42, help="Seed number for reproducibility")
    parser.add_argument("--run_original", type=bool, default=False, help="Use original train-val-split")
    
    return parser.parse_args()

def do_tune(manifest, epochs, n_trials, seed):
    sweep = {
        "fc_units": [64, 128, 256, 512],
        "learning_rates": [1e-3, 5e-4],
        "batch_sizes": [16, 32],
        "dropouts": [0.0, 0.3, 0.5],
        "activation": ["relu", "leakyrelu"]
    }

    train_ds, val_ds = prepare_datasets(manifest, train_frac=0.1, seed=seed)
    return tune_full(sweep, train_ds, val_ds, epochs, n_trials, seed)

def do_train(config, manifest, epochs, patience):
    train_ds, val_ds = prepare_datasets(manifest)
    train_full(config, train_ds, val_ds, epochs, patience)

if __name__ == "__main__":
    args = parse_args()
    
    if args.run_original:
        manifest = _build_original_split_df()
    else:
        manifest = _build_pooled_split_df(seed=args.seed)

    tune_results = do_tune(
        manifest, 
        epochs=args.epochs, 
        n_trials=args.n_trials, 
        seed=args.seed
    )

    # Use config that obtained best results
    config = tune_results[0]
    do_train(
        config,
        manifest,
        epochs=args.epochs,
        patience=args.patience
    )
