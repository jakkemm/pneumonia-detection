import json

import numpy as np
import torch
from model import CNN, create_transforms, generate_class_weights
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import ParameterSampler, StratifiedShuffleSplit
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from tqdm import tqdm

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

torch.manual_seed(42)
np.random.seed(42)

def reduce_dataset(dataset, fraction, seed=42):
    indices = np.arange(len(dataset))
    labels = np.array([s[1] for s in dataset.samples])
    sss = StratifiedShuffleSplit(n_splits=1, train_size=fraction, random_state=seed)
    train_idx, _ = next(sss.split(indices, labels))
    return Subset(dataset, train_idx)


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
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(images)
            
            _, preds = outputs.max(1)
            val_labels.extend(labels.cpu().numpy())
            val_preds.extend(preds.cpu().numpy())
    
    val_balacc = balanced_accuracy_score(val_labels, val_preds)
    return val_balacc

def prepare_datasets():
    train_tf, val_tf = create_transforms()
    
    train_dataset = datasets.ImageFolder("data/train", transform=train_tf)
    val_dataset = datasets.ImageFolder("data/val", transform=val_tf)

    # Reduce the training dataset for hyperparameter search
    train_dataset = reduce_dataset(train_dataset, fraction=0.1)
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
    sampler = list(ParameterSampler(sweep, n_iter=n_trials, random_state=42))
    
    train_dataset, val_dataset = prepare_datasets()
    class_weights = generate_class_weights(train_dataset)

    epochs = 5
    results = []
    
    for i, config in enumerate(tqdm(sampler, desc="Tuning")):
        val_accuracy = train_for_config(config, train_dataset, val_dataset, class_weights, epochs=epochs)
        
        print(f" Validation Accuracy: {val_accuracy}")
        results.append({
            "config": config,
            "val_accuracy": val_accuracy
        })
    
    results.sort(key=lambda x: x["val_accuracy"], reverse=True)
    top_configs = results[:10] if n_trials >= 10 else results

    with open("hp_configs.json", "w") as f:
        json.dump(top_configs, f, indent=4)
