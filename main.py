import argparse
import json

from sklearn.model_selection import ParameterSampler
from tqdm import tqdm

from data import (
    OUTPUT_DIR,
    _build_original_split_df,
    _build_pooled_split_df,
    prepare_datasets,
)
from engine import train_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train the pneumonia detection CNN")
    parser.add_argument("--epochs", type=int, default=100, help="Maximum number of training epochs")
    parser.add_argument("--patience", type=int, default=10, help="Epochs without validation improvement before early stopping")
    parser.add_argument("--epochs_tuning", type=int, default=5, help="Number of epochs in hyperparameter tuning")
    parser.add_argument("--n_trials", type=int, default=25, help="Number of configurations to test for in hyperparameter tuning")
    parser.add_argument("--seed", type=int, default=42, help="Seed number for reproducibility")
    parser.add_argument("--run_original", type=bool, default=False, help="Use original train-val-split")
    
    return parser.parse_args()

def do_tune(tr, val, ts, epochs, patience, n_trials, seed, filename):
    sweep = {
        "fc_units": [64, 128, 256, 512],
        "learning_rates": [1e-3, 5e-4],
        "batch_sizes": [16, 32],
        "dropouts": [0.0, 0.3, 0.5],
        "activation": ["relu", "leakyrelu"]
    }

    sampler = list(ParameterSampler(sweep, n_iter=n_trials, random_state=seed))    
    results = []

    for config in tqdm(sampler, desc="Tuning"):
        checkpoint = train_model(
            config,
            train_ds=tr, 
            val_ds=val, 
            test_ds=ts, 
            filename=filename, 
            epochs=epochs, 
            patience=patience
        )
        
        results.append({
            "config": config,
            "accuracy": checkpoint["best_val_acc"],
        })
    
    results.sort(key=lambda x: x["accuracy"], reverse=True)

    with open(OUTPUT_DIR / filename, "w") as f:
        json.dump(results, f, indent=4)

    return results

if __name__ == "__main__":
    args = parse_args()
    
    if args.run_original:
        manifest = _build_original_split_df()
        file_prefix = "original"
    else:
        manifest = _build_pooled_split_df(seed=args.seed)
        file_prefix = f"s{args.seed}"

    tune_fname = f"{file_prefix}_epochs{args.epochs_tuning}_ntrials{args.n_trials}.json"
    train_fname = f"{file_prefix}_epochs{args.epochs}_patience{args.patience}.pth"

    train, val, test = prepare_datasets(manifest)

    tune_results = do_tune(
        train, val, test,
        epochs_tuning=args.epochs_tuning,
        patience=args.patience,
        n_trials=args.n_trials,
        seed=args.seed,
        filename=tune_fname
    )

    # Use config that obtained best results
    config = tune_results[0]
    results = train_model(
        config, train, val, test,
        filename=train_fname,
        epochs=args.epochs,
        patience=args.patience,
        evaluate_test=True
    )
