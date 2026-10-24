import json
from pathlib import Path

import torch
from model import CNN, create_transforms, generate_class_weights
from sklearn.metrics import balanced_accuracy_score
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
OUTPUT_DIR = Path.cwd() / "outputs"


def load_config():
    with open(OUTPUT_DIR / "best_hyperparameters.json") as f:
        configs = json.load(f)
    
    best_config = configs[0]["config"]
    return best_config

def prepare_datasets():
    train_tf, val_test_tf = create_transforms()
    
    train_dataset = datasets.ImageFolder("data/train", transform=train_tf)
    val_dataset = datasets.ImageFolder("data/val", transform=val_test_tf)
    test_dataset = datasets.ImageFolder("data/test", transform=val_test_tf)
    return train_dataset, val_dataset, test_dataset

def train_model(config, train_dataset, val_dataset, test_dataset, class_weights, epochs=50):
    lr = config["learning_rates"]
    bs = config["batch_sizes"]
    do = config["dropouts"]
    fc = config["fc_units"]
    ac = config["activation"]

    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)

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
        "val_acc": []
    }
    best_val_acc = 0.0

    for epoch in tqdm(range(epochs)):
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
            
            running_loss += loss.item()
            _, preds = outputs.max(1)
            train_total += labels.size(0)
            train_correct += preds.eq(labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_train_acc = train_correct / train_total

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
        
        epoch_val_balacc = balanced_accuracy_score(val_labels, val_preds)
        
        history["train_loss"].append(epoch_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_acc"].append(epoch_val_balacc)

        # Save checkpoint in case something goes wrong
        if epoch_val_balacc > best_val_acc:
            best_val_acc = epoch_val_balacc
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history,
                'val_acc': epoch_val_balacc
            }
            filename = f"best_model_epoch_{epoch+1}_val_bal_acc_{epoch_val_balacc:.4f}.pth"
            torch.save(checkpoint, OUTPUT_DIR / filename)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.4f} | Train Acc: {epoch_train_acc:.4f} | Val Acc: {epoch_val_balacc:.4f}")

    # --------------- TEST ---------------
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, preds = outputs.max(1)
            test_total += labels.size(0)
            test_correct += preds.eq(labels).sum().item()
    
    test_acc = test_correct / test_total
    print(f"\nFinal Test Accuracy: {test_acc:.4f}")
    history["test_acc"] = test_acc

    return model, optimizer, history

if __name__ == "__main__":
    config = load_config()
    train_ds, val_ds, test_ds = prepare_datasets()
    class_weights = generate_class_weights(train_ds)
    
    epochs = 100
    
    model, optimizer, history = train_model(config, train_ds, val_ds, test_ds, class_weights, epochs=epochs)
    
    checkpoint = {
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history
    }
    torch.save(checkpoint, OUTPUT_DIR / "model_final.pth")
