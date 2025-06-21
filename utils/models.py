# BASIC PROTOTYPE 
import torch
import torch.nn as nn

class BirdCNN(nn.Module):
    def __init__(self, num_classes=29, dropout_p=0.3):
        super(BirdCNN, self).__init__()
        self.net = nn.Sequential(
            # Block 1: no early downsampling
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # [32, 313, 224]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),  # [32, 313, 224]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),                              # [32, 156, 112]

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # [64, 156, 112]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),                              # [64, 78, 56]

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # [128, 78, 56]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),                              # [128, 39, 28]

            nn.Flatten(),                                 # [128 * 39 * 28 = 139776]
            nn.Linear(128 * 39 * 28, 256),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.net(x)

