# BASIC PROTOTYPE 
import torch
import torch.nn as nn

class BirdCNN(nn.Module):
    def __init__(self, num_classes=28):
        super(BirdCNN, self).__init__()
        self.net = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),  # → [16, 224, 313]
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 2
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 3
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(64 * 28 * 39, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)  # raw logits, no LogSoftmax
        )

    def forward(self, x):
        return self.net(x)
