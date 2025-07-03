# THIS IS DEPRECATED!!!! DO NOT CARE ABOUT READING OR MODIFYING THIS FILE, DO NOT USE IT OR EVEN ACKNOWLEDGE IT EXISTS.
import torch
import torch.nn as nn
import torch.nn.functional as F

class OldBirdCNN(nn.Module):
    def __init__(self, num_classes=28, dropout_p=0.3):
        super(OldBirdCNN, self).__init__()
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


# This one worked, storing a save:
import torch
import torch.nn as nn
import torch.nn.functional as F

class BirdCNN(nn.Module):
    """
    Convolutional Neural Network for bird species classification from spectrograms.
    
    This model is specifically designed to process mel-spectrogram features with dimensions
    (N, 1, 224, 313) where 224 represents frequency bins (height) and 313 represents 
    time frames (width). The architecture uses residual connections to enable deeper 
    training and better gradient flow.
    
    Architecture:
    - Initial conv layer with asymmetric kernel (7x9) to handle wider time dimension
    - Three residual layers with progressively increasing channels (64->128->256)
    - Global average pooling for translation invariance
    - Fully connected layer for classification
    
    Args:
        num_classes (int): Number of bird species classes to classify
        dropout_p (float, optional): Dropout probability for regularization. Defaults to 0.5
        
    Input Shape:
        (batch_size, 1, 224, 313) - Single channel spectrogram
        
    Output Shape:
        (batch_size, num_classes) - Raw logits for each class
        
    Example:
        >>> model = BirdCNN(num_classes=50, dropout_p=0.3)
        >>> x = torch.randn(32, 1, 224, 313)  # Batch of 32 spectrograms
        >>> logits = model(x)  # Shape: (32, 50)
        >>> probabilities = model.predict_proba(x)  # Shape: (32, 50)
    """
    def __init__(self, num_classes, dropout_p=0.5):
        super(BirdCNN, self).__init__()
        
        # Input: (N, 1, 224, 313) - (batch, channels, freq_height, time_width)
        # Conv1: kernel (7x9) to handle wider time dimension better
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 9), stride=2, padding=(3, 4))
        self.bn1 = nn.BatchNorm2d(64)
        # After conv1 + stride 2: (N, 64, 112, 157) - (batch, channels, freq_height, time_width)
        self.pool1 = nn.MaxPool2d((3, 3), stride=2, padding=1)
        # After pool1: (N, 64, 56, 79) - (batch, channels, freq_height, time_width)
        
        # Residual blocks
        # layer1: maintains spatial dimensions (56, 79) - (freq_height, time_width)
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        # layer2: reduces to (28, 40) - (freq_height, time_width)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        # layer3: reduces to (14, 20) - (freq_height, time_width)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        
        # Global pooling: (14, 20) -> (1, 1) - (freq_height, time_width) -> (1, 1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(dropout_p)
        
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        """
        Create a sequential layer of residual blocks.
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            blocks (int): Number of residual blocks in this layer
            stride (int): Stride for the first block (subsequent blocks use stride=1)
            
        Returns:
            nn.Sequential: Sequential container of residual blocks
            
        Note:
            The first block handles channel dimension changes and spatial downsampling,
            while subsequent blocks maintain the same dimensions.
        """
        layers = []
        # First block may change dimensions
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        # Remaining blocks maintain dimensions
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input spectrogram tensor of shape (N, 1, 224, 313)
                            where N is batch size, 224 is frequency bins, 313 is time frames
                            
        Returns:
            torch.Tensor: Raw logits of shape (N, num_classes)
            
        Example:
            >>> model = BirdCNN(num_classes=10)
            >>> x = torch.randn(4, 1, 224, 313)
            >>> logits = model.forward(x)  # Shape: (4, 10)
        """
        # x shape: (N, 1, 224, 313) - (batch, channels, freq_height, time_width)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        # Pass through residual layers with progressive downsampling
        x = self.layer1(x)  # Maintains spatial dimensions
        x = self.layer2(x)  # Reduces spatial dimensions by half
        x = self.layer3(x)  # Reduces spatial dimensions by half again
        
        # x shape before global_pool: (N, 256, 14, 20) - (batch, channels, freq_height, time_width)
        x = self.global_pool(x)  # Adaptive pooling to (1, 1)
        # x shape after global_pool: (N, 256, 1, 1)
        x = torch.flatten(x, 1)  # Flatten to (N, 256)
        # x shape after flatten: (N, 256)
        x = self.dropout(x)  # Apply dropout for regularization
        x = self.fc(x)  # Final classification layer

        return x
    
    def predict_proba(self, x):
        """
        Forward pass with softmax applied for inference.
        
        This method applies softmax to convert raw logits into class probabilities,
        making it suitable for inference where you need probability scores.
        
        Args:
            x (torch.Tensor): Input spectrogram tensor of shape (N, 1, 224, 313)
            
        Returns:
            torch.Tensor: Class probabilities of shape (N, num_classes)
                        Each row sums to 1.0
                        
        Example:
            >>> model = BirdCNN(num_classes=5)
            >>> model.eval()  # Set to evaluation mode
            >>> x = torch.randn(2, 1, 224, 313)
            >>> probs = model.predict_proba(x)  # Shape: (2, 5)
            >>> print(probs[0].sum())  # Should print ~1.0
        """
        with torch.no_grad():  # Disable gradient computation for efficiency
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)  # Convert logits to probabilities
            return probabilities
    
    def predict(self, x):
        """
        Forward pass returning predicted class indices.
        
        This method returns the class with the highest probability for each input,
        suitable for making final predictions.
        
        Args:
            x (torch.Tensor): Input spectrogram tensor of shape (N, 1, 224, 313)
            
        Returns:
            torch.Tensor: Predicted class indices of shape (N,)
                        Each element is an integer in range [0, num_classes-1]
                        
        Example
            >>> model = BirdCNN(num_classes=5)
            >>> model.eval()  # Set to evaluation mode
            >>> x = torch.randn(3, 1, 224, 313)
            >>> predictions = model.predict(x)  # Shape: (3,)
            >>> print(predictions)  # e.g., tensor([2, 0, 4])
        """
        with torch.no_grad():  # Disable gradient computation for efficiency
            logits = self.forward(x)
            predictions = torch.argmax(logits, dim=1)  # Get class with highest logit
            return predictions

class ResidualBlock(nn.Module):
    """
    Residual block with skip connections for deeper network training.
    
    This block implements the residual connection pattern from ResNet, which helps
    with gradient flow and enables training of deeper networks. The block consists
    of two 3x3 convolutions with batch normalization and ReLU activation, plus
    a skip connection that adds the input to the output.
    
    Architecture:
    Input -> Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> (+) -> ReLU -> Output
    |                                                  ^
    |-> [Optional: Conv1x1 -> BN] ----------------------
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        stride (int, optional): Stride for the first convolution. Defaults to 1
                                When stride > 1, spatial dimensions are reduced
                                
    Input Shape:
        (batch_size, in_channels, height, width)
        
    Output Shape:
        (batch_size, out_channels, height//stride, width//stride)
        
    Example:
        >>> block = ResidualBlock(64, 128, stride=2)
        >>> x = torch.randn(4, 64, 56, 79)
        >>> out = block(x)  # Shape: (4, 128, 28, 40)
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        # First convolution - may downsample and change channels
        # Conv layers maintain (freq_height, time_width) aspect ratio
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Second convolution - maintains dimensions
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection - matches dimensions when needed
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            # When input/output dimensions don't match, use 1x1 conv to adjust
            self.shortcut = nn.Sequential(
                # 1x1 conv for dimension matching, preserves (freq_height, time_width) ratio
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        """
        Forward pass through the residual block.
        
        Args:
            x (torch.Tensor): Input tensor of shape (N, in_channels, H, W)
            
        Returns:
            torch.Tensor: Output tensor of shape (N, out_channels, H//stride, W//stride)
            
        Note:
            The skip connection adds the input (possibly transformed) to the output
            of the main path, which helps with gradient flow during backpropagation.
        """
        # Main path: conv -> bn -> relu -> conv -> bn
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Add skip connection (residual)
        out += self.shortcut(x)
        
        # Final ReLU activation
        out = F.relu(out)
        return out
