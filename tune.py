import json

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import ParameterSampler
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import CNN, DEVICE
from utils import OUTPUT_DIR

torch.manual_seed(42)
np.random.seed(42)

def train_for_config(config, train_dataset, val_dataset, epochs=5):
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
    criterion = nn.CrossEntropyLoss()
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


def tune_full(sweep, train_ds, val_ds, epochs, n_trials, seed):
    sampler = list(ParameterSampler(sweep, n_iter=n_trials, random_state=seed))    
    results = []

    for config in tqdm(sampler, desc="Tuning"):
        bal_acc, roc_auc = train_for_config(config, train_ds, val_ds, epochs=epochs)
        
        results.append({
            "config": config,
            "balanced_accuracy": bal_acc,
            "roc_auc": roc_auc,
        })
    
    results.sort(key=lambda x: (x["balanced_accuracy"], x["roc_auc"]), reverse=True)

    with open(OUTPUT_DIR / "sweep_results.json", "w") as f:
        json.dump(results, f, indent=4)

    return results
