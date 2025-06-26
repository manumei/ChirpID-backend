import torch
import torch.nn as nn
import torch.nn.functional as F

class BirdCNN(nn.Module):
    def __init__(self, num_classes, dropout_p=0.3):
        super(BirdCNN, self).__init__()
        
        # Lighter initial conv - smaller kernel, less aggressive stride
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)  # Reduced channels
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, stride=2)  # Less aggressive pooling
        
        # Fewer, lighter residual blocks
        self.layer1 = self._make_layer(32, 64, 1, stride=1)   # Single block
        self.layer2 = self._make_layer(64, 128, 1, stride=2)  # Single block
        self.layer3 = self._make_layer(128, 256, 1, stride=2) # Single block
        
        # Add dropout in feature extraction
        self.feature_dropout = nn.Dropout2d(0.1)
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Smaller intermediate layer to reduce parameters
        self.fc1 = nn.Linear(256, 128)
        self.fc_dropout = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(128, num_classes)
        
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride, dropout_p=0.1))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1, dropout_p=0.1))
        return nn.Sequential(*layers)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = self.layer1(x)
        x = self.feature_dropout(x)
        x = self.layer2(x)
        x = self.feature_dropout(x)
        x = self.layer3(x)
        x = self.feature_dropout(x)
        
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        
        # Two-stage classifier
        x = F.relu(self.fc1(x))
        x = self.fc_dropout(x)
        x = self.fc2(x)

        return x
    
    def predict_proba(self, x):
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
            return probabilities
    
    def predict(self, x):
        with torch.no_grad():
            logits = self.forward(x)
            predictions = torch.argmax(logits, dim=1)
            return predictions


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout_p=0.0):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout_p) if dropout_p > 0 else nn.Identity()
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out