import torch
import torch.nn as nn
import torch.nn.functional as F

class BirdCNN(nn.Module):
    """
    Lightweight Convolutional Neural Network for bird sound classification from audio spectrograms.
    
    This CNN is specifically designed for processing 2D spectrogram representations of bird audio,
    using residual connections to enable deeper learning while maintaining computational efficiency.
    
    Architecture Overview:
    - Initial conv layer: Extracts low-level spectral features
    - 3 Residual blocks: Progressive feature extraction with skip connections
    - Global pooling: Spatial invariance for variable-length spectrograms
    - 2-stage classifier: Non-linear feature transformation + classification
    
    Total parameters: ~180K (lightweight for mobile/edge deployment)
    
    Key Design Decisions:
    - Residual blocks: Prevent vanishing gradients, enable training of deeper networks
    - BatchNorm: Stabilizes training, reduces internal covariate shift
    - Dropout: Prevents overfitting on limited bird audio datasets
    - AdaptiveAvgPool: Handles variable spectrogram sizes (different audio lengths)
    - Single channel input: Typical for mel-spectrograms or MFCCs
    
    Args:
        num_classes (int): Number of bird species to classify
        dropout_p (float): Dropout probability for regularization (default: 0.3)
    """
    
    def __init__(self, num_classes, dropout_p=0.3):
        """
        Initialize the BirdCNN architecture.
        
        Layer-by-layer parameter count:
        - conv1: (5*5*1 + 1) * 32 = 832 params
        - Residual blocks: ~150K params total
        - Classifier: (256*128 + 128) + (128*num_classes + num_classes) = ~33K + 128*num_classes
        
        Args:
            num_classes (int): Number of bird species classes
            dropout_p (float): Dropout probability for final classifier
        """
        super(BirdCNN, self).__init__()
        
        # Initial feature extraction layer
        # 5x5 kernel captures local spectral-temporal patterns in spectrograms
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)  # ~832 params
        self.bn1 = nn.BatchNorm2d(32)  # 64 params (32*2 for scale/shift)
        self.pool1 = nn.MaxPool2d(2, stride=2)  # Reduces spatial dimensions by 2x
        
        # Progressive feature extraction with residual connections
        # Each layer doubles channels while reducing spatial dimensions
        self.layer1 = self._make_layer(32, 64, 1, stride=1)   # ~37K params
        self.layer2 = self._make_layer(64, 128, 1, stride=2)  # ~74K params  
        self.layer3 = self._make_layer(128, 256, 1, stride=2) # ~148K params
        
        # Regularization during feature extraction
        self.feature_dropout = nn.Dropout2d(0.1)  # Spatial dropout for 2D features
        
        # Global pooling for spatial invariance
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # Output: (batch, 256, 1, 1)
        
        # Two-stage classifier for non-linear decision boundary
        self.fc1 = nn.Linear(256, 128)  # ~33K params (256*128 + 128)
        self.fc_dropout = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(128, num_classes)  # 128*num_classes + num_classes params
        
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        """
        Create a sequence of residual blocks with progressive feature extraction.
        
        Residual blocks are crucial for audio spectrograms because:
        1. Enable deeper networks without vanishing gradients
        2. Allow learning of both low-level (formants, harmonics) and high-level (species-specific) features
        3. Skip connections preserve important spectral information across layers
        
        Args:
            in_channels (int): Input feature channels
            out_channels (int): Output feature channels  
            blocks (int): Number of residual blocks to stack
            stride (int): Convolution stride (>1 for downsampling)
            
        Returns:
            nn.Sequential: Sequence of residual blocks
        """
        layers = []
        # First block handles channel/spatial dimension changes
        layers.append(ResidualBlock(in_channels, out_channels, stride, dropout_p=0.1))
        # Subsequent blocks maintain dimensions
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1, dropout_p=0.1))
        return nn.Sequential(*layers)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Processing pipeline for spectrogram input:
        1. Extract low-level spectral features (edges, textures)
        2. Hierarchical feature learning through residual blocks
        3. Global pooling for temporal invariance
        4. Non-linear classification
        
        Input shape: (batch_size, 1, height, width) - typical spectrogram
        Output shape: (batch_size, num_classes) - class logits
        
        Args:
            x (torch.Tensor): Input spectrogram batch (B, 1, H, W)
            
        Returns:
            torch.Tensor: Class logits (B, num_classes)
        """
        # Initial feature extraction: (B,1,H,W) -> (B,32,H/2,W/2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        # Hierarchical feature learning with regularization
        x = self.layer1(x)  # (B,32,H/2,W/2) -> (B,64,H/2,W/2)
        x = self.feature_dropout(x)
        x = self.layer2(x)  # (B,64,H/2,W/2) -> (B,128,H/4,W/4)
        x = self.feature_dropout(x)
        x = self.layer3(x)  # (B,128,H/4,W/4) -> (B,256,H/8,W/8)
        x = self.feature_dropout(x)
        
        # Global feature aggregation: (B,256,H/8,W/8) -> (B,256,1,1)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)  # (B,256,1,1) -> (B,256)
        
        # Two-stage classification with regularization
        x = F.relu(self.fc1(x))  # (B,256) -> (B,128)
        x = self.fc_dropout(x)
        x = self.fc2(x)  # (B,128) -> (B,num_classes)

        return x
    
    def predict_proba(self, x):
        """
        Generate class probabilities for input spectrograms.
        
        Applies softmax to convert logits to normalized probabilities,
        useful for confidence estimation and threshold-based decisions.
        
        Args:
            x (torch.Tensor): Input spectrogram batch
            
        Returns:
            torch.Tensor: Class probabilities (B, num_classes), sum to 1.0
        """
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
            return probabilities
    
    def predict(self, x):
        """
        Generate class predictions for input spectrograms.
        
        Returns the most likely class index for each input spectrogram.
        
        Args:
            x (torch.Tensor): Input spectrogram batch
            
        Returns:
            torch.Tensor: Predicted class indices (B,)
        """
        with torch.no_grad():
            logits = self.forward(x)
            predictions = torch.argmax(logits, dim=1)
            return predictions


class ResidualBlock(nn.Module):
    """
    Residual Block implementation for audio spectrogram processing.
    
    The residual connection (skip connection) is crucial for audio processing because:
    1. Preserves important spectral information across network depth
    2. Enables gradient flow for training deeper networks
    3. Allows learning of both additive and residual transformations
    4. Helps capture multi-scale temporal patterns in bird vocalizations
    
    Architecture: Conv->BN->ReLU->Dropout->Conv->BN + Skip -> ReLU
    
    For spectrograms, this captures:
    - Local spectral patterns (harmonics, formants)
    - Temporal dynamics (note transitions, trills)
    - Multi-resolution features (fine-grained vs. broad patterns)
    
    Parameter count per block: ~2 * (3*3*in_channels*out_channels + 2*out_channels)
    """
    
    def __init__(self, in_channels, out_channels, stride=1, dropout_p=0.0):
        """
        Initialize residual block with optional downsampling.
        
        Args:
            in_channels (int): Input feature channels
            out_channels (int): Output feature channels
            stride (int): Convolution stride (>1 for spatial downsampling)
            dropout_p (float): Dropout probability for regularization
        """
        super(ResidualBlock, self).__init__()
        
        # Main transformation path
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)  # 3x3 conv
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)  # 3x3 conv
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout_p) if dropout_p > 0 else nn.Identity()
        
        # Skip connection path (handles dimension mismatches)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            # 1x1 conv to match dimensions for skip connection
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        """
        Forward pass with residual connection.
        
        The skip connection allows the network to learn:
        - Identity mapping (when optimal transformation is minimal)
        - Residual mapping (learning what to ADD to the input)
        
        Critical for spectrograms as it preserves important frequency information
        while allowing learning of complex spectral-temporal patterns.
        
        Args:
            x (torch.Tensor): Input feature maps
            
        Returns:
            torch.Tensor: Output feature maps with residual connection
        """
        # Main transformation path
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        
        # Add skip connection and apply final activation
        out += self.shortcut(x)  # Element-wise addition
        out = F.relu(out)
        return out