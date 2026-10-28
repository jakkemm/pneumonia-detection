import copy
import random

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import CNN, DEVICE


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)

            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = outputs.argmax(dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    y_true = np.asarray(all_labels)
    y_pred = np.asarray(all_preds)
    y_prob = np.asarray(all_probs)

    loss = total_loss / len(loader.dataset)
    accuracy = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)

    return loss, accuracy, auc, y_true, y_pred, y_prob

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    train_labels = []
    train_preds = []

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        _, preds = outputs.max(1)
        train_labels.extend(labels.cpu().numpy())
        train_preds.extend(preds.cpu().numpy())

    loss = running_loss / len(loader.dataset)
    acc = accuracy_score(train_labels, train_preds)
    return loss, acc


def train_model(config, train_ds, val_ds, test_ds, path, seed, epochs=100, patience=10, evaluate_test=False, save_checkpoint=True):
    assert not evaluate_test or save_checkpoint, "When evaluating test dataset, checkpoint needs to be saved."
    set_seed(seed)

    lr = config["learning_rates"]
    bs = config["batch_sizes"]
    do = config["dropouts"]
    fc = config["fc_units"]
    ac = config["activation"]

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False)

    model = CNN(
        fc_units=fc, 
        activation=ac, 
        dropout=do
    ).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = []

    best_val_loss = float("inf")
    epochs_no_improvement = 0

    for epoch in tqdm(range(1, epochs+1), desc="Training"):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, val_auc, _, _, _ = evaluate(model, val_loader, criterion)
        
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_auc": val_auc
        })
        
        if val_loss < best_val_loss:
            epochs_no_improvement = 0
            best_val_loss = val_loss
            
            checkpoint = {
                "config": config,
                "best_epoch": epoch,
                "val_at_best_epoch": {
                    "loss": val_loss,
                    "acc": val_acc,
                    "auc": val_auc,
                },
                "model_state_dict": copy.deepcopy(model.state_dict()),
                "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
                "history": history,
                "test_metrics": None,
            }
            if save_checkpoint:
                torch.save(checkpoint, path)
        else:
            epochs_no_improvement += 1

        tqdm.write(
            f"Epoch [{epoch}/{epochs}] - "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2%} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.2%} | "
            f"No improvement: {epochs_no_improvement}"
        )

        if epochs_no_improvement >= patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    if evaluate_test:
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

        _, test_acc, auc, y_true, y_pred, y_prob = evaluate(model, test_loader, criterion)

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp = cm[0]
        fn, tp = cm[1]

        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        precision = tp / (tp + fp)
        f1 = 2 * tp / (2 * tp + fp + fn)

        print(f"\nBest checkpoint: epoch {checkpoint['best_epoch']}")
        print(f"Test set ({len(y_true)} images) - Accuracy: {test_acc:.2%}")
        print("                 Predicted Normal  Predicted Pneumo")
        print(f"True Normal      {tn:5d}             {fp:5d}")
        print(f"True Pneumo      {fn:5d}             {tp:5d}")
        print(
            f"Sensitivity={sensitivity:.2%}  Specificity={specificity:.2%}  "
            f"AUC={auc:.4f}  Accuracy={test_acc:.2%}"
        )
        
        checkpoint["history"] = history
        checkpoint["test_metrics"] = {
            "accuracy": test_acc,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "roc_auc": auc,
            "precision": precision,
            "f1": f1,
            "confusion_matrix": cm.tolist(),
        }
        
        torch.save(checkpoint, path)
        np.savez(path.with_suffix(".npz"), y_true=y_true, y_pred=y_pred, y_prob=y_prob) 

    return checkpoint