import torch
import torch.nn as nn
import torch.nn.functional as F

class BirdCNN(nn.Module):
    def __init__(self, num_classes, dropout_p=0.5):
        super(BirdCNN, self).__init__()
        
        # More gradual channel progression
        self.block1 = self._make_block(1, 32, dropout_p=0.2)      # [32, 313, 224]
        self.pool1 = nn.MaxPool2d(2, stride=2)                    # [32, 156, 112]
        
        self.block2 = self._make_block(32, 64, dropout_p=0.3)     # [64, 156, 112] 
        self.pool2 = nn.MaxPool2d(2, stride=2)                    # [64, 78, 56]
        
        self.block3 = self._make_block(64, 128, dropout_p=0.4)    # [128, 78, 56]
        self.pool3 = nn.MaxPool2d(2, stride=2)                    # [128, 39, 28]
        
        self.block4 = self._make_block(128, 256, dropout_p=0.4)   # [256, 39, 28]
        self.pool4 = nn.MaxPool2d(2, stride=2)                    # [256, 19, 14]
        
        # Global Average Pooling instead of massive linear layer
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))           # [256, 1, 1]
        
        # Smaller, more gradual classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),                                         # [256]
            nn.Dropout(dropout_p),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(128, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_block(self, in_channels, out_channels, dropout_p=0.0):
        """Consistent block structure with residual-like connections"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_p),  # Spatial dropout
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def _initialize_weights(self):
        """Proper weight initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.pool3(self.block3(x))
        x = self.pool4(self.block4(x))
        
        # Global average pooling eliminates the massive linear layer
        x = self.global_pool(x)
        x = self.classifier(x)
        
        return x

# Alternative: ResNet-style with skip connections
class BirdResNet(nn.Module):
    def __init__(self, num_classes, dropout_p=0.5):
        super(BirdResNet, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(3, stride=2, padding=1)
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(dropout_p)
        
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
