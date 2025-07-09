import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Helper blocks and modules used by various architectures
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
        """
        Initialize the ResidualBlock.
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels  
            stride (int, optional): Stride for the first convolution. Defaults to 1
        """
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


# Helper blocks for CNN architectures
class BottleneckBlock(nn.Module):
    """
    Bottleneck residual block with 1x1, 3x3, 1x1 convolution pattern.
    
    This block is used in deeper ResNet architectures to reduce computational cost
    while maintaining representational capacity. It uses 1x1 convolutions to reduce
    and then expand the channel dimensions around a 3x3 convolution.
    
    Args:
        inplanes (int): Number of input channels
        planes (int): Number of intermediate channels (bottleneck width)
        expansion_planes (int): Number of output channels after expansion
        stride (int, optional): Stride for the 3x3 convolution. Defaults to 1
    """
    def __init__(self, inplanes, planes, expansion_planes, stride=1):
        """
        Initialize the BottleneckBlock.
        
        Args:
            inplanes (int): Number of input channels
            planes (int): Number of intermediate channels
            expansion_planes (int): Number of output channels  
            stride (int, optional): Stride for middle convolution. Defaults to 1
        """
        super(BottleneckBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, expansion_planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(expansion_planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != expansion_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes, expansion_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(expansion_planes)
            )
    
    def forward(self, x):
        """
        Forward pass through the bottleneck block.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor with skip connection applied
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class SeparableConvBlock(nn.Module):
    """
    Depthwise separable convolution block.
    
    Separates spatial and channel-wise convolutions for computational efficiency.
    First applies depthwise convolution (spatial filtering per channel) followed
    by pointwise convolution (channel mixing).
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        stride (int, optional): Stride for depthwise convolution. Defaults to 1
    """
    def __init__(self, in_channels, out_channels, stride=1):
        """
        Initialize the SeparableConvBlock.
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            stride (int, optional): Stride for depthwise conv. Defaults to 1
        """
        super(SeparableConvBlock, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        """
        Forward pass through separable convolution block.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor with residual connection if applicable
        """
        out = F.relu(self.bn1(self.depthwise(x)))
        out = F.relu(self.bn2(self.pointwise(out)), inplace=False)  # Remove inplace operation
        shortcut_out = self.shortcut(x)
        out = out + shortcut_out  # Avoid inplace addition
        return out


class AttentionBlock(nn.Module):
    """
    Channel attention mechanism using global average pooling.
    
    Computes channel-wise attention weights using global average pooling
    followed by a small fully connected network. Helps the model focus
    on important feature channels.
    
    Args:
        channels (int): Number of input channels
        reduction (int, optional): Reduction factor for bottleneck. Defaults to 16
    """
    def __init__(self, channels, reduction=16):
        """
        Initialize the AttentionBlock.
        
        Args:
            channels (int): Number of input channels
            reduction (int, optional): Channel reduction factor. Defaults to 16
        """
        super(AttentionBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=False),  # Changed from inplace=True to avoid gradient issues
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Apply channel attention to input tensor.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
            
        Returns:
            torch.Tensor: Attention-weighted tensor of same shape as input
        """
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block for channel attention.
    
    Implements the SE mechanism that adaptively recalibrates channel-wise
    feature responses by explicitly modelling interdependencies between channels.
    
    Args:
        channels (int): Number of input channels
        reduction (int, optional): Reduction ratio for squeeze operation. Defaults to 4
    """
    def __init__(self, channels, reduction=4):
        """
        Initialize the SEBlock.
        
        Args:
            channels (int): Number of input channels
            reduction (int, optional): Reduction ratio. Defaults to 4
        """
        super(SEBlock, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Apply squeeze-and-excitation attention.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: SE-weighted tensor
        """
        return x * self.se(x)


class MBConvBlock(nn.Module):
    """
    Mobile Inverted Bottleneck Convolution block (MBConv).
    
    Used in EfficientNet and MobileNet architectures. Expands channels,
    applies depthwise convolution, applies SE attention, then compresses
    channels back down.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        expand_ratio (int): Channel expansion ratio
        stride (int, optional): Stride for depthwise conv. Defaults to 1
    """
    def __init__(self, in_channels, out_channels, expand_ratio, stride=1):
        """
        Initialize the MBConvBlock.
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            expand_ratio (int): Channel expansion ratio
            stride (int, optional): Stride for conv. Defaults to 1
        """
        super(MBConvBlock, self).__init__()
        hidden_dim = in_channels * expand_ratio
        self.use_residual = stride == 1 and in_channels == out_channels
        
        layers = []
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(inplace=True)
            ])
        
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True)
        ])
        
        layers.append(SEBlock(hidden_dim))
        
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass through MBConv block.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor with residual connection if applicable
        """
        out = self.conv(x)
        if self.use_residual:
            out = out + x
        return out


class DenseBlock(nn.Module):
    """
    Dense block from DenseNet architecture.
    
    Each layer receives feature maps from all preceding layers as input,
    promoting feature reuse and improving gradient flow.
    
    Args:
        in_channels (int): Number of input channels
        growth_rate (int): Number of channels added by each layer
        num_layers (int): Number of dense layers in the block
    """
    def __init__(self, in_channels, growth_rate, num_layers):
        """
        Initialize the DenseBlock.
        
        Args:
            in_channels (int): Number of input channels
            growth_rate (int): Growth rate (channels added per layer)
            num_layers (int): Number of dense layers
        """
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(DenseLayer(in_channels + i * growth_rate, growth_rate))
    
    def forward(self, x):
        """
        Forward pass through dense block.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Concatenated features from all layers
        """
        features = [x]
        for layer in self.layers:
            new_feature = layer(torch.cat(features, 1))
            features.append(new_feature)
        return torch.cat(features, 1)


class DenseLayer(nn.Module):
    """
    Individual dense layer within a DenseBlock.
    
    Implements the bottleneck design with BN-ReLU-Conv(1x1)-BN-ReLU-Conv(3x3).
    
    Args:
        in_channels (int): Number of input channels
        growth_rate (int): Number of output channels (growth rate)
    """
    def __init__(self, in_channels, growth_rate):
        """
        Initialize the DenseLayer.
        
        Args:
            in_channels (int): Number of input channels
            growth_rate (int): Number of output channels
        """
        super(DenseLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, 4 * growth_rate, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, 3, padding=1, bias=False)
    
    def forward(self, x):
        """
        Forward pass through dense layer.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output feature map
        """
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        return out


class TransitionLayer(nn.Module):
    """
    Transition layer between dense blocks.
    
    Reduces spatial dimensions and optionally reduces channels to control
    model complexity between dense blocks.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
    """
    def __init__(self, in_channels, out_channels):
        """
        Initialize the TransitionLayer.
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
        """
        super(TransitionLayer, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.pool = nn.AvgPool2d(2, stride=2)
    
    def forward(self, x):
        """
        Forward pass through transition layer.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Downsampled output tensor
        """
        out = self.conv(F.relu(self.bn(x)))
        out = self.pool(out)
        return out


class InceptionBlock(nn.Module):
    """
    Inception block with parallel convolutions of different kernel sizes.
    
    Processes input through multiple parallel paths with different receptive
    fields and concatenates the results.
    
    Args:
        in_channels (int): Number of input channels
        ch1x1 (int): Channels for 1x1 conv path
        ch3x3red (int): Channels for 3x3 reduction path
        ch3x3 (int): Channels for 3x3 conv path
        ch5x5red (int): Channels for 5x5 reduction path  
        ch5x5 (int): Channels for 5x5 conv path
        pool_proj (int): Channels for pooling projection path
    """
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        """
        Initialize the InceptionBlock.
        
        Args:
            in_channels (int): Number of input channels
            ch1x1 (int): 1x1 conv output channels
            ch3x3red (int): 3x3 path reduction channels
            ch3x3 (int): 3x3 conv output channels
            ch5x5red (int): 5x5 path reduction channels
            ch5x5 (int): 5x5 conv output channels
            pool_proj (int): Pooling path projection channels
        """
        super(InceptionBlock, self).__init__()
        self.branch1 = nn.Conv2d(in_channels, ch1x1, kernel_size=1)
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3red, kernel_size=1),
            nn.Conv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5red, kernel_size=1),
            nn.Conv2d(ch5x5red, ch5x5, kernel_size=5, padding=2)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1)
        )
    
    def forward(self, x):
        """
        Forward pass through inception block.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Concatenated output from all branches
        """
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        return torch.cat([branch1, branch2, branch3, branch4], 1)


class WideResidualBlock(nn.Module):
    """
    Wide residual block with increased width and dropout.
    
    Wider layers with fewer blocks, includes dropout for regularization.
    Used in Wide ResNet architectures.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        stride (int, optional): Stride for first convolution. Defaults to 1
        dropout_p (float, optional): Dropout probability. Defaults to 0.0
    """
    def __init__(self, in_channels, out_channels, stride=1, dropout_p=0.0):
        """
        Initialize the WideResidualBlock.
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            stride (int, optional): Stride for conv. Defaults to 1
            dropout_p (float, optional): Dropout probability. Defaults to 0.0
        """
        super(WideResidualBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.dropout = nn.Dropout(dropout_p)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
    
    def forward(self, x):
        """
        Forward pass through wide residual block.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor with residual connection
        """
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)
        return out


class ShuffleNetUnit(nn.Module):
    """
    ShuffleNet unit with channel shuffle operation (simplified).
    
    Efficient building block that uses grouped convolutions and channel
    shuffling for computational efficiency.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        stride (int): Stride for convolution
        groups (int): Number of groups for grouped convolution
    """
    def __init__(self, in_channels, out_channels, stride, groups):
        """
        Initialize the ShuffleNetUnit.
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            stride (int): Convolution stride
            groups (int): Number of groups
        """
        super(ShuffleNetUnit, self).__init__()
        self.stride = stride
        self.groups = max(1, min(groups, in_channels, out_channels))  # Ensure valid groups
        
        # Simplified design to avoid complex group constraints
        mid_channels = out_channels // 4
        mid_channels = max(mid_channels, self.groups)  # Ensure at least groups channels
        
        # Use regular convolutions if groups cause issues
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, 
                              padding=1, groups=mid_channels, bias=False)  # Depthwise
        self.bn2 = nn.BatchNorm2d(mid_channels)
        
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(3, stride=stride, padding=1) if stride > 1 else nn.Identity(),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        """
        Forward pass through shuffle unit.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor with skip connection
        """
        residual = self.shortcut(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        
        if self.stride == 1 and x.shape == out.shape:
            out = out + residual
        else:
            out = torch.cat([out, residual], dim=1) if self.stride > 1 else out + residual
        
        return F.relu(out)


class RegNetBlock(nn.Module):
    """
    RegNet block with regular design principles.
    
    Simple and regular design with consistent width and depth patterns
    for predictable network scaling.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        stride (int, optional): Stride for convolution. Defaults to 1
        groups (int, optional): Number of groups. Defaults to 1
    """
    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        """
        Initialize the RegNetBlock.
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            stride (int, optional): Convolution stride. Defaults to 1
            groups (int, optional): Number of groups. Defaults to 1
        """
        super(RegNetBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, 
                                padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        """
        Forward pass through RegNet block.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor with residual connection
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# Main CNN Architectures for Bird Classification
class BirdCNN_v1(nn.Module):
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
        """
        Initialize the BirdCNN_v1 model.
        
        Args:
            num_classes (int): Number of bird species classes
            dropout_p (float, optional): Dropout probability. Defaults to 0.5
        """
        super(BirdCNN_v1, self).__init__()
        
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


# Keep the original BirdCNN for backward compatibility  
BirdCNN = BirdCNN_v1

class BirdCNN_v2(nn.Module):
    """
    VGG-inspired CNN with deeper layers and smaller filters.
    
    Uses 3x3 convolutions throughout and more layers for detailed feature extraction.
    Progressive channel expansion with max pooling for spatial downsampling.
    
    Args:
        num_classes (int): Number of bird species classes to classify
        dropout_p (float, optional): Dropout probability for regularization. Defaults to 0.5
    """
    def __init__(self, num_classes, dropout_p=0.5):
        """
        Initialize the BirdCNN_v2 model.
        
        Args:
            num_classes (int): Number of bird species classes
            dropout_p (float, optional): Dropout probability. Defaults to 0.5
        """
        super(BirdCNN_v2, self).__init__()
        
        # VGG-style architecture with 3x3 convs
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),  # 224x313 -> 112x156
            
            # Block 2  
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),  # 112x156 -> 56x78
            
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),  # 56x78 -> 28x39
            
            # Block 4
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),  # 28x39 -> 14x19
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout_p),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass through VGG-style network.
        
        Args:
            x (torch.Tensor): Input spectrogram tensor
            
        Returns:
            torch.Tensor: Classification logits
        """
        x = self.features(x)
        x = self.classifier(x)
        return x


class BirdCNN_v3(nn.Module):
    """
    ResNet-inspired deeper network with bottleneck blocks.
    
    Better gradient flow with more sophisticated residual connections using
    bottleneck design for computational efficiency in deeper networks.
    
    Args:
        num_classes (int): Number of bird species classes to classify
        dropout_p (float, optional): Dropout probability for regularization. Defaults to 0.5
    """
    def __init__(self, num_classes, dropout_p=0.5):
        """
        Initialize the BirdCNN_v3 model.
        
        Args:
            num_classes (int): Number of bird species classes
            dropout_p (float, optional): Dropout probability. Defaults to 0.5
        """
        super(BirdCNN_v3, self).__init__()
        
        # Initial conv layer
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(3, stride=2, padding=1)
        
        # ResNet layers with bottleneck blocks
        self.layer1 = self._make_layer(64, 64, 256, 3, stride=1)   # 56x78
        self.layer2 = self._make_layer(256, 128, 512, 4, stride=2) # 28x39
        self.layer3 = self._make_layer(512, 256, 1024, 6, stride=2) # 14x19
        self.layer4 = self._make_layer(1024, 512, 2048, 3, stride=2) # 7x9
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)
        self.dropout = nn.Dropout(dropout_p)
        
    def _make_layer(self, inplanes, planes, expansion_planes, blocks, stride):
        """
        Create a layer of bottleneck blocks.
        
        Args:
            inplanes (int): Input channels
            planes (int): Intermediate channels
            expansion_planes (int): Output channels
            blocks (int): Number of blocks
            stride (int): Stride for first block
            
        Returns:
            nn.Sequential: Layer of bottleneck blocks
        """
        layers = []
        layers.append(BottleneckBlock(inplanes, planes, expansion_planes, stride))
        for _ in range(1, blocks):
            layers.append(BottleneckBlock(expansion_planes, planes, expansion_planes, 1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass through ResNet-style network.
        
        Args:
            x (torch.Tensor): Input spectrogram tensor
            
        Returns:
            torch.Tensor: Classification logits
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class BirdCNN_v4(nn.Module):
    """
    PANN-inspired architecture with attention mechanisms.
    
    Uses separable convolutions and attention for audio-specific feature learning.
    Incorporates depthwise separable convolutions for efficiency and channel attention.
    
    Args:
        num_classes (int): Number of bird species classes to classify
        dropout_p (float, optional): Dropout probability for regularization. Defaults to 0.5
    """
    def __init__(self, num_classes, dropout_p=0.5):
        """
        Initialize the BirdCNN_v4 model.
        
        Args:
            num_classes (int): Number of bird species classes
            dropout_p (float, optional): Dropout probability. Defaults to 0.5
        """
        super(BirdCNN_v4, self).__init__()
        
        # Initial processing
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
        self.bn1 = nn.BatchNorm2d(64)
        
        # Depthwise separable conv blocks (PANN-style)
        self.block1 = SeparableConvBlock(64, 128, stride=2)    # 56x78
        self.block2 = SeparableConvBlock(128, 256, stride=2)   # 28x39
        self.block3 = SeparableConvBlock(256, 512, stride=2)   # 14x19
        self.block4 = SeparableConvBlock(512, 1024, stride=2)  # 7x9
        
        # Attention mechanism
        self.attention = AttentionBlock(1024)
        
        # Classification head
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=False),  # Changed from inplace=True to avoid gradient issues
            nn.Dropout(dropout_p),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass through PANN-style network with attention.
        
        Args:
            x (torch.Tensor): Input spectrogram tensor
            
        Returns:
            torch.Tensor: Classification logits
        """
        x = F.relu(self.bn1(self.conv1(x)))
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        
        x = self.attention(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class BirdCNN_v5(nn.Module):
    """
    EfficientNet-inspired mobile architecture.
    
    Uses inverted residual blocks with squeeze-and-excitation for efficient
    feature learning with reduced computational cost.
    
    Args:
        num_classes (int): Number of bird species classes to classify
        dropout_p (float, optional): Dropout probability for regularization. Defaults to 0.5
    """
    def __init__(self, num_classes, dropout_p=0.5):
        """
        Initialize the BirdCNN_v5 model.
        
        Args:
            num_classes (int): Number of bird species classes
            dropout_p (float, optional): Dropout probability. Defaults to 0.5
        """
        super(BirdCNN_v5, self).__init__()
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True)
        )
        
        # MBConv blocks (EfficientNet-style)
        self.blocks = nn.Sequential(
            MBConvBlock(32, 16, expand_ratio=1, stride=1),   # 112x156
            MBConvBlock(16, 24, expand_ratio=6, stride=2),   # 56x78
            MBConvBlock(24, 24, expand_ratio=6, stride=1),   
            MBConvBlock(24, 40, expand_ratio=6, stride=2),   # 28x39
            MBConvBlock(40, 40, expand_ratio=6, stride=1),
            MBConvBlock(40, 80, expand_ratio=6, stride=2),   # 14x19
            MBConvBlock(80, 80, expand_ratio=6, stride=1),
            MBConvBlock(80, 112, expand_ratio=6, stride=1),
            MBConvBlock(112, 192, expand_ratio=6, stride=2), # 7x9
            MBConvBlock(192, 192, expand_ratio=6, stride=1),
            MBConvBlock(192, 320, expand_ratio=6, stride=1),
        )
        
        # Head
        self.head = nn.Sequential(
            nn.Conv2d(320, 1280, 1),
            nn.BatchNorm2d(1280),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout_p),
            nn.Linear(1280, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass through EfficientNet-style network.
        
        Args:
            x (torch.Tensor): Input spectrogram tensor
            
        Returns:
            torch.Tensor: Classification logits
        """
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x


class BirdCNN_v6(nn.Module):
    """
    Lightweight CNN with asymmetric kernels for time-frequency analysis.
    
    Optimized for spectrogram shape (224x313) with time-frequency aware kernels
    that separately process frequency and temporal patterns before fusion.
    
    Args:
        num_classes (int): Number of bird species classes to classify
        dropout_p (float, optional): Dropout probability for regularization. Defaults to 0.5
    """
    def __init__(self, num_classes, dropout_p=0.5):
        """
        Initialize the BirdCNN_v6 model.
        
        Args:
            num_classes (int): Number of bird species classes
            dropout_p (float, optional): Dropout probability. Defaults to 0.5
        """
        super(BirdCNN_v6, self).__init__()
        
        # Asymmetric kernels for time-frequency separation
        self.freq_conv = nn.Conv2d(1, 32, kernel_size=(7, 1), padding=(3, 0))  # Frequency patterns
        self.time_conv = nn.Conv2d(1, 32, kernel_size=(1, 7), padding=(0, 3))  # Temporal patterns
        self.fuse_conv = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Progressive feature extraction
        self.conv_blocks = nn.Sequential(
            self._conv_block(64, 128, stride=2),    # 112x156
            self._conv_block(128, 256, stride=2),   # 56x78
            self._conv_block(256, 512, stride=2),   # 28x39
            self._conv_block(512, 512, stride=2),   # 14x19
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(256, num_classes)
        )
    
    def _conv_block(self, in_channels, out_channels, stride=1):
        """
        Create a convolutional block with two conv layers.
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            stride (int, optional): Stride for first conv. Defaults to 1
            
        Returns:
            nn.Sequential: Convolutional block
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        """
        Forward pass with parallel frequency and time processing.
        
        Args:
            x (torch.Tensor): Input spectrogram tensor
            
        Returns:
            torch.Tensor: Classification logits
        """
        # Parallel frequency and time processing
        freq_features = self.freq_conv(x)
        time_features = self.time_conv(x)
        x = torch.cat([freq_features, time_features], dim=1)
        
        x = F.relu(self.bn1(self.fuse_conv(x)))
        x = self.conv_blocks(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class BirdCNN_v7(nn.Module):
    """
    Dense network with dense blocks for feature reuse.
    
    Inspired by DenseNet for efficient parameter usage through feature
    concatenation and reuse across layers.
    
    Args:
        num_classes (int): Number of bird species classes to classify
        dropout_p (float, optional): Dropout probability for regularization. Defaults to 0.5
    """
    def __init__(self, num_classes, dropout_p=0.5):
        """
        Initialize the BirdCNN_v7 model.
        
        Args:
            num_classes (int): Number of bird species classes
            dropout_p (float, optional): Dropout probability. Defaults to 0.5
        """
        super(BirdCNN_v7, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(3, stride=2, padding=1)
        
        # Dense blocks
        self.dense1 = DenseBlock(64, growth_rate=32, num_layers=6)
        self.trans1 = TransitionLayer(64 + 6*32, 128)
        
        self.dense2 = DenseBlock(128, growth_rate=32, num_layers=12)
        self.trans2 = TransitionLayer(128 + 12*32, 256)
        
        self.dense3 = DenseBlock(256, growth_rate=32, num_layers=24)
        self.trans3 = TransitionLayer(256 + 24*32, 512)
        
        self.dense4 = DenseBlock(512, growth_rate=32, num_layers=16)
        
        # Final classification
        final_channels = 512 + 16*32
        self.bn_final = nn.BatchNorm2d(final_channels)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(final_channels, num_classes)
        self.dropout = nn.Dropout(dropout_p)
    
    def forward(self, x):
        """
        Forward pass through DenseNet-style network.
        
        Args:
            x (torch.Tensor): Input spectrogram tensor
            
        Returns:
            torch.Tensor: Classification logits
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = self.dense1(x)
        x = self.trans1(x)
        x = self.dense2(x)
        x = self.trans2(x)
        x = self.dense3(x)
        x = self.trans3(x)
        x = self.dense4(x)
        
        x = F.relu(self.bn_final(x))
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x


class BirdCNN_v8(nn.Module):
    """
    PANN-inspired architecture with pre-activation residual blocks and dual attention.
    
    Uses channel attention and spatial attention for audio-specific feature learning.
    Different from v4 which uses separable convolutions - this uses attention mechanisms
    with pre-activation residual blocks and frequency-aware processing.
    
    Args:
        num_classes (int): Number of bird species classes to classify
        dropout_p (float, optional): Dropout probability for regularization. Defaults to 0.5
    """
    def __init__(self, num_classes, dropout_p=0.5):
        """
        Initialize the BirdCNN_v8 model.
        
        Args:
            num_classes (int): Number of bird species classes
            dropout_p (float, optional): Dropout probability. Defaults to 0.5
        """
        super(BirdCNN_v8, self).__init__()
        
        # Initial processing - spectrogram-aware kernel
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
        self.bn1 = nn.BatchNorm2d(64)
        
        # Pre-activation residual blocks with progressive feature extraction
        self.block1 = self._make_pre_activation_block(64, 128, stride=2)    # 56x78
        self.block2 = self._make_pre_activation_block(128, 256, stride=2)   # 28x39
        self.block3 = self._make_pre_activation_block(256, 512, stride=2)   # 14x19
        self.block4 = self._make_pre_activation_block(512, 1024, stride=2)  # 7x9
        
        # Dual attention mechanisms (channel + spatial)
        self.channel_attention = self._make_channel_attention(1024)
        self.spatial_attention = self._make_spatial_attention()
        
        # Frequency-aware pooling
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))  # Pool frequency dimension
        self.temporal_conv = nn.Conv1d(1024, 1024, kernel_size=3, padding=1)
        self.temporal_bn = nn.BatchNorm1d(1024)
        
        # Classification head with attention pooling
        self.attention_pool = nn.Sequential(
            nn.Conv1d(1024, 1, kernel_size=1),
            nn.Softmax(dim=-1)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout_p),
            nn.Linear(512, num_classes)
        )
    
    def _make_pre_activation_block(self, in_channels, out_channels, stride=1):
        """
        Create pre-activation residual block (BN->ReLU->Conv).
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            stride (int, optional): Stride for conv. Defaults to 1
            
        Returns:
            nn.Sequential: Pre-activation block
        """
        return nn.Sequential(
            # Main path
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels, out_channels//2, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels//2),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels//2, out_channels//2, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels//2),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels//2, out_channels, kernel_size=1, bias=False),
            # Skip connection handled in forward pass
        )
    
    def _make_channel_attention(self, channels, reduction=16):
        """
        Create channel attention module.
        
        Args:
            channels (int): Number of channels
            reduction (int, optional): Reduction factor. Defaults to 16
            
        Returns:
            nn.Sequential: Channel attention module
        """
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=False),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    
    def _make_spatial_attention(self):
        """
        Create spatial attention module.
        
        Returns:
            nn.Sequential: Spatial attention module
        """
        return nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Forward pass with dual attention and frequency-aware processing.
        
        Args:
            x (torch.Tensor): Input spectrogram tensor
            
        Returns:
            torch.Tensor: Classification logits
        """
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Pre-activation residual blocks with skip connections
        identity = x
        x = self.block1(x)
        if identity.shape != x.shape:
            identity = F.interpolate(identity, size=x.shape[2:], mode='bilinear', align_corners=False)
            identity = F.pad(identity, (0, 0, 0, 0, 0, x.shape[1] - identity.shape[1]))
        x = x + identity
        
        identity = x
        x = self.block2(x)
        if identity.shape != x.shape:
            identity = F.interpolate(identity, size=x.shape[2:], mode='bilinear', align_corners=False)
            identity = F.pad(identity, (0, 0, 0, 0, 0, x.shape[1] - identity.shape[1]))
        x = x + identity
        
        identity = x
        x = self.block3(x)
        if identity.shape != x.shape:
            identity = F.interpolate(identity, size=x.shape[2:], mode='bilinear', align_corners=False)
            identity = F.pad(identity, (0, 0, 0, 0, 0, x.shape[1] - identity.shape[1]))
        x = x + identity
        
        identity = x
        x = self.block4(x)
        if identity.shape != x.shape:
            identity = F.interpolate(identity, size=x.shape[2:], mode='bilinear', align_corners=False)
            identity = F.pad(identity, (0, 0, 0, 0, 0, x.shape[1] - identity.shape[1]))
        x = x + identity
        
        # Apply dual attention
        # Channel attention
        ca_weights = self.channel_attention(x).unsqueeze(-1).unsqueeze(-1)
        x = x * ca_weights
        
        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_out, max_out], dim=1)
        sa_weights = self.spatial_attention(spatial_input)
        x = x * sa_weights
        
        # Frequency-aware processing
        x = self.freq_pool(x)  # Pool frequency dimension: (B, C, 1, T)
        x = x.squeeze(2)       # Remove frequency dimension: (B, C, T)
        x = F.relu(self.temporal_bn(self.temporal_conv(x)))
        
        # Attention-based temporal pooling
        attention_weights = self.attention_pool(x)  # (B, 1, T)
        x = torch.sum(x * attention_weights, dim=-1)  # (B, C)
        
        x = self.classifier(x)
        return x


class BirdCNN_v9(nn.Module):
    """
    Inception-inspired network with parallel convolutions.
    
    Uses parallel convolution paths with different kernel sizes to capture
    multi-scale features in spectrograms.
    
    Args:
        num_classes (int): Number of bird species classes to classify
        dropout_p (float, optional): Dropout probability for regularization. Defaults to 0.5
    """
    def __init__(self, num_classes, dropout_p=0.5):
        """
        Initialize the BirdCNN_v9 model.
        
        Args:
            num_classes (int): Number of bird species classes
            dropout_p (float, optional): Dropout probability. Defaults to 0.5
        """
        super(BirdCNN_v9, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, padding=1)
        
        self.inception1 = InceptionBlock(64, 64, 96, 128, 16, 32, 32)
        self.inception2 = InceptionBlock(256, 128, 128, 192, 32, 96, 64)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, padding=1)
        
        self.inception3 = InceptionBlock(480, 192, 96, 208, 16, 48, 64)
        self.inception4 = InceptionBlock(512, 160, 112, 224, 24, 64, 64)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_p)
        self.fc = nn.Linear(512, num_classes)
    
    def forward(self, x):
        """
        Forward pass through Inception-style network.
        
        Args:
            x (torch.Tensor): Input spectrogram tensor
            
        Returns:
            torch.Tensor: Classification logits
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool1(x)
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.maxpool2(x)
        x = self.inception3(x)
        x = self.inception4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class BirdCNN_v10(nn.Module):
    """
    Compact CNN optimized for limited data.
    
    Lightweight architecture with fewer parameters, designed for scenarios
    with limited training data or computational resources.
    
    Args:
        num_classes (int): Number of bird species classes to classify
        dropout_p (float, optional): Dropout probability for regularization. Defaults to 0.15 (la subi para testing porque antes se caia)
    """
    def __init__(self, num_classes, dropout_p=0.15):
        """
        Initialize the BirdCNN_v10 model.
        
        Args:
            num_classes (int): Number of bird species classes
            dropout_p (float, optional): Dropout probability. Defaults to 0.5
        """
        super(BirdCNN_v10, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_p),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass through compact network.
        
        Args:
            x (torch.Tensor): Input spectrogram tensor
            
        Returns:
            torch.Tensor: Classification logits
        """
        x = self.features(x)
        x = self.classifier(x)
        return x


class BirdCNN_v11(nn.Module):
    """
    Wide ResNet variant with wider layers but fewer residual blocks.
    
    Uses wider layers with increased capacity per layer while reducing
    the total depth of the network.
    
    Args:
        num_classes (int): Number of bird species classes to classify
        dropout_p (float, optional): Dropout probability for regularization. Defaults to 0.5
    """
    def __init__(self, num_classes, dropout_p=0.5):
        """
        Initialize the BirdCNN_v11 model.
        
        Args:
            num_classes (int): Number of bird species classes
            dropout_p (float, optional): Dropout probability. Defaults to 0.5
        """
        super(BirdCNN_v11, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        
        self.layer1 = self._make_wide_layer(64, 128, 2, stride=1)
        self.layer2 = self._make_wide_layer(128, 256, 2, stride=2)
        self.layer3 = self._make_wide_layer(256, 512, 2, stride=2)
        self.layer4 = self._make_wide_layer(512, 1024, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_p)
        self.fc = nn.Linear(1024, num_classes)
    
    def _make_wide_layer(self, in_channels, out_channels, blocks, stride):
        """
        Create a layer of wide residual blocks.
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            blocks (int): Number of blocks in layer
            stride (int): Stride for first block
            
        Returns:
            nn.Sequential: Layer of wide residual blocks
        """
        layers = []
        layers.append(WideResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(WideResidualBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass through Wide ResNet.
        
        Args:
            x (torch.Tensor): Input spectrogram tensor
            
        Returns:
            torch.Tensor: Classification logits
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class BirdCNN_v12(nn.Module):
    """
    Temporal-aware CNN with 1D convolutions along time axis.
    
    Separates frequency and temporal processing, using 2D convolutions
    for frequency analysis followed by 1D convolutions for temporal modeling.
    
    Args:
        num_classes (int): Number of bird species classes to classify
        dropout_p (float, optional): Dropout probability for regularization. Defaults to 0.5
    """
    def __init__(self, num_classes, dropout_p=0.5):
        """
        Initialize the BirdCNN_v12 model.
        
        Args:
            num_classes (int): Number of bird species classes
            dropout_p (float, optional): Dropout probability. Defaults to 0.5
        """
        super(BirdCNN_v12, self).__init__()
        
        self.freq_conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=(5, 1), stride=(2, 1), padding=(2, 0)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(256*28, 512, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 1024, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Conv1d(1024, 1024, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
        )
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_p),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass with separate frequency and temporal processing.
        
        Args:
            x (torch.Tensor): Input spectrogram tensor
            
        Returns:
            torch.Tensor: Classification logits
        """
        x = self.freq_conv(x)
        N, C, H, W = x.shape
        x = x.view(N, C*H, W) # tensor reshaping
        x = self.temporal_conv(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x


class BirdCNN_v13(nn.Module):
    """
    ShuffleNet-inspired efficient architecture (simplified).
    
    Efficient network design using simplified blocks instead of complex
    group convolutions for better compatibility and training stability.
    
    Args:
        num_classes (int): Number of bird species classes to classify
        dropout_p (float, optional): Dropout probability for regularization. Defaults to 0.5
    """
    def __init__(self, num_classes, dropout_p=0.5):
        """
        Initialize the BirdCNN_v13 model.
        
        Args:
            num_classes (int): Number of bird species classes
            dropout_p (float, optional): Dropout probability. Defaults to 0.5
        """
        super(BirdCNN_v13, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 24, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(24)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        
        # Simplified stages without group convolutions to avoid issues
        self.stage2 = self._make_simple_stage(24, 116, 4, stride=2)
        self.stage3 = self._make_simple_stage(116, 232, 8, stride=2)
        self.stage4 = self._make_simple_stage(232, 464, 4, stride=2)
        
        self.conv5 = nn.Conv2d(464, 1024, kernel_size=1)
        self.bn5 = nn.BatchNorm2d(1024)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_p)
        self.fc = nn.Linear(1024, num_classes)
    
    def _make_simple_stage(self, in_channels, out_channels, num_blocks, stride):
        """
        Create stage with simple residual blocks instead of ShuffleNet units.
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            num_blocks (int): Number of blocks in stage
            stride (int): Stride for first block
            
        Returns:
            nn.Sequential: Stage of simple blocks
        """
        layers = []
        # First block with stride
        layers.append(self._simple_block(in_channels, out_channels, stride))
        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(self._simple_block(out_channels, out_channels, 1))
        return nn.Sequential(*layers)
    
    def _simple_block(self, in_channels, out_channels, stride):
        """
        Simple residual block.
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            stride (int): Convolution stride
            
        Returns:
            nn.Sequential: Simple residual block
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels//2, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels//2, out_channels//2, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels//2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        """
        Forward pass through simplified ShuffleNet-style network.
        
        Args:
            x (torch.Tensor): Input spectrogram tensor
            
        Returns:
            torch.Tensor: Classification logits
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class BirdCNN_v14(nn.Module):
    """
    RegNet-inspired architecture with regular design.
    
    Uses regular design principles with consistent width and depth
    scaling for predictable network behavior and performance.
    
    Args:
        num_classes (int): Number of bird species classes to classify
        dropout_p (float, optional): Dropout probability for regularization. Defaults to 0.5
    """
    def __init__(self, num_classes, dropout_p=0.5):
        """
        Initialize the BirdCNN_v14 model.
        
        Args:
            num_classes (int): Number of bird species classes
            dropout_p (float, optional): Dropout probability. Defaults to 0.5
        """
        super(BirdCNN_v14, self).__init__()
        
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.stage1 = self._make_stage(32, 64, depth=2, stride=2)
        self.stage2 = self._make_stage(64, 128, depth=4, stride=2)
        self.stage3 = self._make_stage(128, 256, depth=8, stride=2)
        self.stage4 = self._make_stage(256, 512, depth=2, stride=2)
        
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout_p),
            nn.Linear(512, num_classes)
        )
    
    def _make_stage(self, in_channels, out_channels, depth, stride):
        """
        Create a stage of RegNet blocks.
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            depth (int): Number of blocks in stage
            stride (int): Stride for first block
            
        Returns:
            nn.Sequential: Stage of RegNet blocks
        """
        layers = []
        layers.append(RegNetBlock(in_channels, out_channels, stride=stride))
        for _ in range(1, depth):
            layers.append(RegNetBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass through RegNet-style network.
        
        Args:
            x (torch.Tensor): Input spectrogram tensor
            
        Returns:
            torch.Tensor: Classification logits
        """
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.head(x)
        return x


class BirdCNN_v15(nn.Module):
    """
    Frequency-aware CNN with explicit frequency band processing.
    
    Explicitly processes different frequency bands (low, mid, high) with
    separate pathways before fusion for frequency-aware feature learning.
    
    Args:
        num_classes (int): Number of bird species classes to classify
        dropout_p (float, optional): Dropout probability for regularization. Defaults to 0.5
    """
    def __init__(self, num_classes, dropout_p=0.5):
        """
        Initialize the BirdCNN_v15 model.
        
        Args:
            num_classes (int): Number of bird species classes
            dropout_p (float, optional): Dropout probability. Defaults to 0.5
        """
        super(BirdCNN_v15, self).__init__()
        
        self.low_freq_path = self._make_freq_path(1, 128)
        self.mid_freq_path = self._make_freq_path(1, 128)
        self.high_freq_path = self._make_freq_path(1, 128)
        
        self.cross_band_conv = nn.Conv2d(384, 512, kernel_size=3, padding=1)
        self.cross_band_bn = nn.BatchNorm2d(512)
        
        self.temporal_layers = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=(1, 7), stride=(1, 2), padding=(0, 3)),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2)),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_p),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(512, num_classes)
        )
    
    def _make_freq_path(self, in_channels, out_channels):
        """
        Create frequency processing pathway.
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            
        Returns:
            nn.Sequential: Frequency processing pathway
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        """
        Forward pass with explicit frequency band processing.
        
        Args:
            x (torch.Tensor): Input spectrogram tensor
            
        Returns:
            torch.Tensor: Classification logits
        """
        low_band = x[:, :, :75, :]
        mid_band = x[:, :, 75:150, :]
        high_band = x[:, :, 150:, :]
        
        low_features = self.low_freq_path(low_band)
        mid_features = self.mid_freq_path(mid_band)  
        high_features = self.high_freq_path(high_band)
        
        target_size = low_features.shape[2:]
        mid_features = F.interpolate(mid_features, size=target_size, mode='bilinear', align_corners=False)
        high_features = F.interpolate(high_features, size=target_size, mode='bilinear', align_corners=False)
        
        x = torch.cat([low_features, mid_features, high_features], dim=1)
        x = F.relu(self.cross_band_bn(self.cross_band_conv(x)))
        x = self.temporal_layers(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x


class BirdCNN_v16(nn.Module):
    """
    Hybrid CNN-RNN architecture.
    
    Combines convolutional feature extraction with LSTM temporal modeling
    for capturing both local spectral patterns and long-term temporal dependencies.
    
    Args:
        num_classes (int): Number of bird species classes to classify
        dropout_p (float, optional): Dropout probability for regularization. Defaults to 0.5
    """
    def __init__(self, num_classes, dropout_p=0.5):
        """
        Initialize the BirdCNN_v16 model.
        
        Args:
            num_classes (int): Number of bird species classes
            dropout_p (float, optional): Dropout probability. Defaults to 0.5
        """
        super(BirdCNN_v16, self).__init__()
        
        self.conv_features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))
        
        self.lstm = nn.LSTM(
            input_size=512, 
            hidden_size=256, 
            num_layers=2, 
            batch_first=True,
            dropout=dropout_p if dropout_p > 0 else 0,
            bidirectional=True
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass through hybrid CNN-RNN network.
        
        Args:
            x (torch.Tensor): Input spectrogram tensor
            
        Returns:
            torch.Tensor: Classification logits
        """
        x = self.conv_features(x)
        x = self.freq_pool(x)
        x = x.squeeze(2)
        x = x.transpose(1, 2)
        
        lstm_out, (h_n, c_n) = self.lstm(x)
        x = lstm_out[:, -1, :]
        x = self.classifier(x)
        return x


class BirdCNN_v17(nn.Module):
    """
    v7 con mas dropout y menos params
    """
    def __init__(self, num_classes, dropout_p=0.6):
        """
        Initialize the BirdCNN_v7 model.
        
        Args:
            num_classes (int): Number of bird species classes
            dropout_p (float, optional): Dropout probability. Defaults to 0.5
        """
        super(BirdCNN_v7, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(3, stride=2, padding=1)
        
        # Dense blocks
        self.dense1 = DenseBlock(64, growth_rate=32, num_layers=6)
        self.trans1 = TransitionLayer(64 + 6*32, 128)
        
        self.dense2 = DenseBlock(128, growth_rate=32, num_layers=12)
        self.trans2 = TransitionLayer(128 + 12*32, 256)
        
        self.dense3 = DenseBlock(256, growth_rate=32, num_layers=24)
        self.trans3 = TransitionLayer(256 + 24*32, 512)
        
        self.dense4 = DenseBlock(512, growth_rate=32, num_layers=16)
        
        # Final classification
        final_channels = 512 + 16*32
        self.bn_final = nn.BatchNorm2d(final_channels)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(final_channels, num_classes)
        self.dropout = nn.Dropout(dropout_p)
    
    def forward(self, x):
        """
        Forward pass through DenseNet-style network.
        
        Args:
            x (torch.Tensor): Input spectrogram tensor
            
        Returns:
            torch.Tensor: Classification logits
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = self.dense1(x)
        x = self.trans1(x)
        x = self.dense2(x)
        x = self.trans2(x)
        x = self.dense3(x)
        x = self.trans3(x)
        x = self.dense4(x)
        
        x = F.relu(self.bn_final(x))
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x


class BirdCNN_v18(nn.Module):
    """
    v7 con leaky relu
    """
    def __init__(self, num_classes, dropout_p=0.5):
        """
        Initialize the BirdCNN_v18 model.
        
        Args:
            num_classes (int): Number of bird species classes
            dropout_p (float, optional): Dropout probability. Defaults to 0.5
        """
        super(BirdCNN_v18, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(3, stride=2, padding=1)
        
        # Dense blocks
        self.dense1 = DenseBlock(64, growth_rate=32, num_layers=6)
        self.trans1 = TransitionLayer(64 + 6*32, 128)
        
        self.dense2 = DenseBlock(128, growth_rate=32, num_layers=12)
        self.trans2 = TransitionLayer(128 + 12*32, 256)
        
        self.dense3 = DenseBlock(256, growth_rate=32, num_layers=24)
        self.trans3 = TransitionLayer(256 + 24*32, 512)
        
        self.dense4 = DenseBlock(512, growth_rate=32, num_layers=16)
        
        # Final classification
        final_channels = 512 + 16*32
        self.bn_final = nn.BatchNorm2d(final_channels)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(final_channels, num_classes)
        self.dropout = nn.Dropout(dropout_p)
    
    def forward(self, x):
        """
        Forward pass through DenseNet-style network.
        
        Args:
            x (torch.Tensor): Input spectrogram tensor
            
        Returns:
            torch.Tensor: Classification logits
        """
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = self.dense1(x)
        x = self.trans1(x)
        x = self.dense2(x)
        x = self.trans2(x)
        x = self.dense3(x)
        x = self.trans3(x)
        x = self.dense4(x)
        
        x = F.leaky_relu(self.bn_final(x))
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x


class BirdCNN_v19(nn.Module):
    """
    v5 con mas dropout y menos params
    """
    def __init__(self, num_classes, dropout_p=0.5):
        """
        Initialize the BirdCNN_v5 model.
        
        Args:
            num_classes (int): Number of bird species classes
            dropout_p (float, optional): Dropout probability. Defaults to 0.5
        """
        super(BirdCNN_v5, self).__init__()
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True)
        )
        
        # MBConv blocks (EfficientNet-style)
        self.blocks = nn.Sequential(
            MBConvBlock(32, 16, expand_ratio=1, stride=1),   # 112x156
            MBConvBlock(16, 24, expand_ratio=6, stride=2),   # 56x78
            MBConvBlock(24, 24, expand_ratio=6, stride=1),   
            MBConvBlock(24, 40, expand_ratio=6, stride=2),   # 28x39
            MBConvBlock(40, 40, expand_ratio=6, stride=1),
            MBConvBlock(40, 80, expand_ratio=6, stride=2),   # 14x19
            MBConvBlock(80, 80, expand_ratio=6, stride=1),
            MBConvBlock(80, 112, expand_ratio=6, stride=1),
            MBConvBlock(112, 192, expand_ratio=6, stride=2), # 7x9
            MBConvBlock(192, 192, expand_ratio=6, stride=1),
            MBConvBlock(192, 320, expand_ratio=6, stride=1),
        )
        
        # Head
        self.head = nn.Sequential(
            nn.Conv2d(320, 1280, 1),
            nn.BatchNorm2d(1280),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout_p),
            nn.Linear(1280, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass through EfficientNet-style network.
        
        Args:
            x (torch.Tensor): Input spectrogram tensor
            
        Returns:
            torch.Tensor: Classification logits
        """
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x


class BirdCNN_v20(nn.Module):
    """
    v5 con GELU, dropout in stem/head, residuals in MBConvBlock
    """
    
    def __init__(self, num_classes, dropout_p=0.5):
        super(BirdCNN_v20, self).__init__()

        # Stem with GELU activation
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU()
        )
        
        # MBConv blocks sequence (uses your unchanged MBConvBlock with SEBlock)
        self.blocks = nn.Sequential(
            MBConvBlock(32, 16, expand_ratio=1, stride=1),
            MBConvBlock(16, 24, expand_ratio=6, stride=2),
            MBConvBlock(24, 24, expand_ratio=6, stride=1),
            MBConvBlock(24, 40, expand_ratio=6, stride=2),
            MBConvBlock(40, 40, expand_ratio=6, stride=1),
            MBConvBlock(40, 80, expand_ratio=6, stride=2),
            MBConvBlock(80, 80, expand_ratio=6, stride=1),
            MBConvBlock(80, 112, expand_ratio=6, stride=1),
            MBConvBlock(112, 192, expand_ratio=6, stride=2),
            MBConvBlock(192, 192, expand_ratio=6, stride=1),
            MBConvBlock(192, 320, expand_ratio=6, stride=1),
        )

        # Head with GELU activation and dropout
        self.head = nn.Sequential(
            nn.Conv2d(320, 1280, 1),
            nn.BatchNorm2d(1280),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout_p),
            nn.Linear(1280, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x