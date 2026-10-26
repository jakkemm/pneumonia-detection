import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from tqdm import tqdm

from model import CNN, DEVICE, create_transforms, generate_class_weights

OUTPUT_DIR = Path.cwd() / "outputs"
DATA_DIR = Path.cwd() / "data" / "chest_xray"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

torch.manual_seed(42)
np.random.seed(42)


def load_config():
    sweep_results_file = OUTPUT_DIR / "sweep_results.json"

    if not sweep_results_file.exists():
        raise FileNotFoundError("No `sweep_results.json` file found. Make sure to run `tune.py` first.")

    with open(sweep_results_file) as f:
        configs = json.load(f)
    
    best_config = configs[0]["config"]
    return best_config

def prepare_datasets():
    train_tf, val_test_tf = create_transforms()
    
    train_full = datasets.ImageFolder(DATA_DIR / "train", transform=train_tf)
    val_full = datasets.ImageFolder(DATA_DIR / "train", transform=val_test_tf)

    labels = np.array(train_full.targets)
    indices = np.arange(len(labels))

    # Splitting the original training data: 80% train, 20% validation
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(splitter.split(indices, labels))

    train_dataset = Subset(train_full, train_idx)
    val_dataset = Subset(val_full, val_idx)

    return train_dataset, val_dataset

def train_model(config, train_dataset, val_dataset, class_weights, epochs=100, patience=10):
    lr = config["learning_rates"]
    bs = config["batch_sizes"]
    do = config["dropouts"]
    fc = config["fc_units"]
    ac = config["activation"]

    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False)

    model = CNN(
        fc_units=fc, 
        activation=ac, 
        dropout=do
    ).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_balanced_acc": []
    }

    best_val_bal_acc = -1.0
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in tqdm(range(1, epochs+1)):
        # --------------- TRAINING ---------------
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * labels.size(0)
            _, preds = outputs.max(1)
            train_total += labels.size(0)
            train_correct += preds.eq(labels).sum().item()

        epoch_loss = running_loss / len(train_dataset)
        epoch_train_acc = train_correct / train_total

        # --------------- VALIDATION ---------------
        model.eval()
        val_labels = []
        val_preds = []
        val_loss_sum = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs = model(images)
                
                val_loss_sum += criterion(outputs, labels).item() * labels.size(0)
                
                _, preds = outputs.max(1)
                val_labels.extend(labels.cpu().numpy())
                val_preds.extend(preds.cpu().numpy())
        
        epoch_val_bal_acc = balanced_accuracy_score(val_labels, val_preds)
        epoch_val_loss = val_loss_sum / len(val_dataset)
        
        history["train_loss"].append(epoch_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_balanced_acc"].append(epoch_val_bal_acc)

        # Save checkpoint in case something goes wrong
        if epoch_val_bal_acc > best_val_bal_acc:
            best_val_bal_acc = epoch_val_bal_acc
            epochs_without_improvement = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_balanced_acc': epoch_val_bal_acc,
                'config': config
            }
            torch.save(checkpoint, OUTPUT_DIR / "best_model_epoch.pth")

        else:
            epochs_without_improvement += 1

        tqdm.write(
            f"Epoch [{epoch}/{epochs}] - "
            f"Loss: {epoch_loss:.4f} | "
            f"Train Acc: {epoch_train_acc:.4f} | "
            f"Balanced Val Acc: {epoch_val_bal_acc:.4f} | "
            f"Epochs without improvement: {epochs_without_improvement}"
        )

        if epochs_without_improvement >= patience:
            print(
                f"Early stopping at epoch {epoch}. "
                f"Best epoch: {best_epoch}, "
                f"best validation balanced accuracy: {best_val_bal_acc:.4f}"
            )
            break


    return model, optimizer, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the pneumonia detection CNN")
    parser.add_argument("--epochs", type=int, default=100, help="Maximum number of training epochs")
    parser.add_argument("--patience", type=int, default=10, help="Epochs without validation improvement before early stopping")
    args = parser.parse_args()

    config = load_config()
    train_ds, val_ds = prepare_datasets()
    class_weights = generate_class_weights(train_ds)
    
    model, optimizer, history = train_model(
        config,
        train_ds,
        val_ds,
        class_weights,
        epochs=args.epochs,
        patience=args.patience
    )
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
        'config': config
    }
    torch.save(checkpoint, OUTPUT_DIR / "model_final.pth")
