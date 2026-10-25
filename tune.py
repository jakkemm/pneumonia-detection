import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import ParameterSampler, StratifiedShuffleSplit
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from tqdm import tqdm

from model import CNN, create_transforms, generate_class_weights

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
DATA_DIR = Path.cwd() / "data" / "chest_xray"
OUTPUT_DIR = Path.cwd() / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

torch.manual_seed(42)
np.random.seed(42)

def train_for_config(config, train_dataset, val_dataset, class_weights, epochs=5):
    lr = config["learning_rates"]
    bs = config["batch_sizes"]
    do = config["dropouts"]
    fc = config["fc_units"]
    ac = config["activation"]

    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False, num_workers=2)

    model = CNN(
        fc_units=fc,
        activation=ac,
        dropout=do,
    ).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # --------------- TRAINING ---------------
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        
        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    # --------------- VALIDATION ---------------
    model.eval()
    val_labels = []
    val_preds = []
    val_probs = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            outputs = model(images)

            probs = torch.softmax(outputs, dim=1)
            _, preds = outputs.max(dim=1)

            val_labels.extend(labels.cpu().numpy())
            val_preds.extend(preds.cpu().numpy())
            val_probs.extend(probs[:, 1].cpu().numpy())

    bal_acc = balanced_accuracy_score(val_labels, val_preds)
    roc_auc = roc_auc_score(val_labels, val_probs)

    return bal_acc, roc_auc

def prepare_datasets():
    train_tf, val_tf = create_transforms()
    
    train_full = datasets.ImageFolder(DATA_DIR / "train", transform=train_tf)
    val_full = datasets.ImageFolder(DATA_DIR / "train", transform=val_tf)

    labels = np.array(train_full.targets)
    indices = np.arange(len(labels))

    # Splitting the original training data: 80% train, 20% validation
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(splitter.split(indices, labels))

    # Taking 10% of the 80% training partition for the random search
    subset_splitter = StratifiedShuffleSplit(n_splits=1, train_size=0.1, random_state=42)
    subset_pos, _ = next(subset_splitter.split(train_idx, labels[train_idx]))
    search_train_idx = train_idx[subset_pos]

    train_dataset = Subset(train_full, search_train_idx)
    val_dataset = Subset(val_full, val_idx)

    return train_dataset, val_dataset

def sample_config(space):
    return {k: np.random.choice(v) for k, v in space.items()}


if __name__ == "__main__":
    # Hyperparameters to sweep
    sweep = {
        "fc_units": [64, 128, 256, 512],
        "learning_rates": [1e-3, 5e-4],
        "batch_sizes": [16, 32],
        "dropouts": [0.0, 0.3, 0.5],
        "activation": ["relu", "leakyrelu"]
    }

    n_trials = 25
    epochs = 5

    sampler = list(ParameterSampler(sweep, n_iter=n_trials, random_state=42))
    
    train_dataset, val_dataset = prepare_datasets()
    class_weights = generate_class_weights(train_dataset)

    results = []
    
    for i, config in enumerate(tqdm(sampler, desc="Tuning"), start=1):
        bal_acc, roc_auc = train_for_config(config, train_dataset, val_dataset, class_weights, epochs=epochs)
        
        results.append({
            "config": config,
            "balanced_accuracy": bal_acc,
            "roc_auc": roc_auc,
        })
    
    results.sort(key=lambda x: (x["balanced_accuracy"], x["roc_auc"]), reverse=True)

    with open(OUTPUT_DIR / "sweep_results.json", "w") as f:
        json.dump(results, f, indent=4)
