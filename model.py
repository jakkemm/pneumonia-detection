import torch
from torch import nn
from torchvision import transforms

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

class CNN(nn.Module):
    def __init__(self, fc_units=128, activation="relu", dropout=0.0):
        super().__init__()
        
        if activation.lower() == "relu":
            act = nn.ReLU()
        elif activation.lower() == "leakyrelu":
            act = nn.LeakyReLU(0.1)
        else:
            raise ValueError("activation must be 'relu' or 'leakyrelu'")
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            act,
            nn.MaxPool2d(2),
            
            nn.Conv2d(16, 32, 3, padding=1),
            act,
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            act,
            nn.MaxPool2d(2)
        )
        
        layers = [
            nn.Flatten(),
            nn.LazyLinear(fc_units),
            act
        ]
        if dropout != 0.0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(fc_units, 2))
        
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def create_transforms():
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((128, 128)),
        # ----- train augmentations for improving the model fitting -----
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(5, translate=(0.02,0.02)),
        transforms.ColorJitter(brightness=0.05, contrast=0.05),
        # ---------------------------------------------------------------
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # For val and test sets we don't want to augment, since this is for model testing
    val_test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    return train_transform, val_test_transform
