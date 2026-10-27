import argparse
import json

from sklearn.model_selection import ParameterSampler
from tqdm import tqdm

from data import (
    OUTPUT_DIR,
    build_manifest,
    prepare_datasets,
    shrink_train,
)
from engine import train_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train the pneumonia detection CNN")
    parser.add_argument("--epochs", type=int, default=100, help="Maximum number of training epochs")
    parser.add_argument("--patience", type=int, default=10, help="Epochs without validation improvement before early stopping")
    parser.add_argument("--epochs_tuning", type=int, default=10, help="Number of epochs in hyperparameter tuning")
    parser.add_argument("--n_trials", type=int, default=20, help="Number of configurations to test for in hyperparameter tuning")
    parser.add_argument("--seed", type=int, default=42, help="Seed number for reproducibility")
    parser.add_argument(
        "--mode",
        type=str,
        default="pool",
        help="Dataset split mode: 'original' (original splits), "
            "'original_regrouped' (80/20 train-val split), "
            "'pool' (pool all data and create new splits).",
    )
    
    return parser.parse_args()

def do_tune(tr, val, ts, epochs, patience, n_trials, seed, path):
    sweep = {
        "fc_units": [64, 128, 256, 512],
        "learning_rates": [1e-3, 5e-4],
        "batch_sizes": [16, 32],
        "dropouts": [0.0, 0.3, 0.5],
        "activation": ["relu", "leakyrelu"]
    }
    sampler = list(ParameterSampler(sweep, n_iter=n_trials, random_state=seed))

    tr = shrink_train(tr, frac=0.3, seed=seed)
    results = []

    for config in tqdm(sampler, desc="Tuning"):
        checkpoint = train_model(
            config,
            train_ds=tr,
            val_ds=val, 
            test_ds=ts,
            path=path,
            seed=seed,
            epochs=epochs,
            patience=patience,
            save_checkpoint=False
        )

        accuracy = checkpoint["best_val_acc"]
        tqdm.write(
            f"Accuracy: {accuracy:.2%} | "
            f"fc_units: {config['fc_units']} | "
            f"lr: {config['learning_rates']} | "
            f"batch_size: {config['batch_sizes']} | "
            f"dropout: {config['dropouts']} | "
            f"activation: {config['activation']}"
        )
        
        results.append({
            "config": config,
            "accuracy": accuracy,
        })
    
    results.sort(key=lambda x: x["accuracy"], reverse=True)

    with open(path, "w") as f:
        json.dump(results, f, indent=4)

    return results

if __name__ == "__main__":
    args = parse_args()
    
    manifest = build_manifest(args.mode, args.seed)
    file_prefix = f"{args.mode}_s{args.seed}" if args.mode != "original" else "original"

    tune_path = OUTPUT_DIR / f"sweep_{file_prefix}_epochs{args.epochs_tuning}_ntrials{args.n_trials}.json"
    train_path = OUTPUT_DIR / f"{file_prefix}_epochs{args.epochs}_patience{args.patience}.pth"

    train, val, test = prepare_datasets(manifest)

    tune_results = do_tune(
        train, val, test,
        epochs=args.epochs_tuning,
        patience=args.patience,
        n_trials=args.n_trials,
        seed=args.seed,
        path=tune_path
    )

    # Use config that obtained best results
    config = tune_results[0]["config"]
    results = train_model(
        config, train, val, test,
        path=train_path,
        seed=args.seed,
        epochs=args.epochs,
        patience=args.patience,
        evaluate_test=True
    )
