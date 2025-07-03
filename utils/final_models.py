import torch
import torch.nn as nn
import torch.nn.functional as F

# Aux Classes
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


# Models

# V5
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

# add versions here

# V5 Variants
class BirdCNN_v5b(nn.Module):
    """
    EfficientNet-inspired mobile architecture - Variant B.
    
    Similar to v5 but with reduced initial channels and modified expand ratios
    for slightly more efficient computation.
    
    Args:
        num_classes (int): Number of bird species classes to classify
        dropout_p (float, optional): Dropout probability for regularization. Defaults to 0.4
    """
    def __init__(self, num_classes, dropout_p=0.4):
        super(BirdCNN_v5b, self).__init__()
        
        # Stem - reduced channels
        self.stem = nn.Sequential(
            nn.Conv2d(1, 24, 3, stride=2, padding=1),
            nn.BatchNorm2d(24),
            nn.SiLU(inplace=True)
        )
        
        # MBConv blocks with modified expand ratios
        self.blocks = nn.Sequential(
            MBConvBlock(24, 16, expand_ratio=1, stride=1),
            MBConvBlock(16, 24, expand_ratio=4, stride=2),   # reduced from 6
            MBConvBlock(24, 24, expand_ratio=4, stride=1),   
            MBConvBlock(24, 40, expand_ratio=4, stride=2),   
            MBConvBlock(40, 40, expand_ratio=4, stride=1),
            MBConvBlock(40, 80, expand_ratio=6, stride=2),   
            MBConvBlock(80, 80, expand_ratio=6, stride=1),
            MBConvBlock(80, 112, expand_ratio=6, stride=1),
            MBConvBlock(112, 192, expand_ratio=6, stride=2), 
            MBConvBlock(192, 192, expand_ratio=6, stride=1),
            MBConvBlock(192, 320, expand_ratio=6, stride=1),
        )
        
        # Head - reduced final channels
        self.head = nn.Sequential(
            nn.Conv2d(320, 960, 1),  # reduced from 1280
            nn.BatchNorm2d(960),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout_p),
            nn.Linear(960, num_classes)
        )
    
    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x


class BirdCNN_v5c(nn.Module):
    """
    EfficientNet-inspired mobile architecture - Variant C.
    
    Similar to v5 but with increased initial channels and higher expand ratios
    for potentially better feature learning.
    
    Args:
        num_classes (int): Number of bird species classes to classify
        dropout_p (float, optional): Dropout probability for regularization. Defaults to 0.6
    """
    def __init__(self, num_classes, dropout_p=0.6):
        super(BirdCNN_v5c, self).__init__()
        
        # Stem - increased channels
        self.stem = nn.Sequential(
            nn.Conv2d(1, 40, 3, stride=2, padding=1),
            nn.BatchNorm2d(40),
            nn.SiLU(inplace=True)
        )
        
        # MBConv blocks with higher expand ratios
        self.blocks = nn.Sequential(
            MBConvBlock(40, 20, expand_ratio=1, stride=1),
            MBConvBlock(20, 32, expand_ratio=8, stride=2),   # increased from 6
            MBConvBlock(32, 32, expand_ratio=8, stride=1),   
            MBConvBlock(32, 48, expand_ratio=8, stride=2),   
            MBConvBlock(48, 48, expand_ratio=8, stride=1),
            MBConvBlock(48, 96, expand_ratio=8, stride=2),   
            MBConvBlock(96, 96, expand_ratio=8, stride=1),
            MBConvBlock(96, 128, expand_ratio=8, stride=1),
            MBConvBlock(128, 224, expand_ratio=8, stride=2), 
            MBConvBlock(224, 224, expand_ratio=8, stride=1),
            MBConvBlock(224, 384, expand_ratio=8, stride=1),
        )
        
        # Head - increased final channels
        self.head = nn.Sequential(
            nn.Conv2d(384, 1536, 1),  # increased from 1280
            nn.BatchNorm2d(1536),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout_p),
            nn.Linear(1536, num_classes)
        )
    
    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x


class BirdCNN_v5d(nn.Module):
    """
    EfficientNet-inspired mobile architecture - Variant D.
    
    Similar to v5 but with mixed expand ratios and intermediate channel counts
    for balanced efficiency and performance.
    
    Args:
        num_classes (int): Number of bird species classes to classify
        dropout_p (float, optional): Dropout probability for regularization. Defaults to 0.3
    """
    def __init__(self, num_classes, dropout_p=0.3):
        super(BirdCNN_v5d, self).__init__()
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(1, 28, 3, stride=2, padding=1),
            nn.BatchNorm2d(28),
            nn.SiLU(inplace=True)
        )
        
        # MBConv blocks with mixed expand ratios
        self.blocks = nn.Sequential(
            MBConvBlock(28, 14, expand_ratio=1, stride=1),
            MBConvBlock(14, 20, expand_ratio=3, stride=2),   # varied ratios
            MBConvBlock(20, 20, expand_ratio=5, stride=1),   
            MBConvBlock(20, 36, expand_ratio=7, stride=2),   
            MBConvBlock(36, 36, expand_ratio=5, stride=1),
            MBConvBlock(36, 72, expand_ratio=6, stride=2),   
            MBConvBlock(72, 72, expand_ratio=6, stride=1),
            MBConvBlock(72, 104, expand_ratio=6, stride=1),
            MBConvBlock(104, 176, expand_ratio=7, stride=2), 
            MBConvBlock(176, 176, expand_ratio=7, stride=1),
            MBConvBlock(176, 288, expand_ratio=7, stride=1),
        )
        
        # Head
        self.head = nn.Sequential(
            nn.Conv2d(288, 1152, 1),  # 4x the input channels
            nn.BatchNorm2d(1152),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout_p),
            nn.Linear(1152, num_classes)
        )
    
    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x


# V7
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

# add versions here

# V7 Variants
class BirdCNN_v7b(nn.Module):
    """
    Dense network with dense blocks for feature reuse - Variant B.
    
    Similar to v7 but with reduced growth rate and fewer layers per block
    for more efficient computation.
    
    Args:
        num_classes (int): Number of bird species classes to classify
        dropout_p (float, optional): Dropout probability for regularization. Defaults to 0.4
    """
    def __init__(self, num_classes, dropout_p=0.4):
        super(BirdCNN_v7b, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(1, 48, kernel_size=7, stride=2, padding=3)  # reduced from 64
        self.bn1 = nn.BatchNorm2d(48)
        self.pool1 = nn.MaxPool2d(3, stride=2, padding=1)
        
        # Dense blocks with reduced growth rate and layers
        growth_rate = 24  # reduced from 32
        self.dense1 = DenseBlock(48, growth_rate=growth_rate, num_layers=4)  # reduced from 6
        self.trans1 = TransitionLayer(48 + 4*growth_rate, 96)
        
        self.dense2 = DenseBlock(96, growth_rate=growth_rate, num_layers=8)  # reduced from 12
        self.trans2 = TransitionLayer(96 + 8*growth_rate, 192)
        
        self.dense3 = DenseBlock(192, growth_rate=growth_rate, num_layers=18)  # reduced from 24
        self.trans3 = TransitionLayer(192 + 18*growth_rate, 384)
        
        self.dense4 = DenseBlock(384, growth_rate=growth_rate, num_layers=12)  # reduced from 16
        
        # Final classification
        final_channels = 384 + 12*growth_rate
        self.bn_final = nn.BatchNorm2d(final_channels)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(final_channels, num_classes)
        self.dropout = nn.Dropout(dropout_p)
    
    def forward(self, x):
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


class BirdCNN_v7c(nn.Module):
    """
    Dense network with dense blocks for feature reuse - Variant C.
    
    Similar to v7 but with increased growth rate and more layers per block
    for potentially better feature learning capacity.
    
    Args:
        num_classes (int): Number of bird species classes to classify
        dropout_p (float, optional): Dropout probability for regularization. Defaults to 0.6
    """
    def __init__(self, num_classes, dropout_p=0.6):
        super(BirdCNN_v7c, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(1, 80, kernel_size=7, stride=2, padding=3)  # increased from 64
        self.bn1 = nn.BatchNorm2d(80)
        self.pool1 = nn.MaxPool2d(3, stride=2, padding=1)
        
        # Dense blocks with increased growth rate and layers
        growth_rate = 40  # increased from 32
        self.dense1 = DenseBlock(80, growth_rate=growth_rate, num_layers=8)  # increased from 6
        self.trans1 = TransitionLayer(80 + 8*growth_rate, 160)
        
        self.dense2 = DenseBlock(160, growth_rate=growth_rate, num_layers=16)  # increased from 12
        self.trans2 = TransitionLayer(160 + 16*growth_rate, 320)
        
        self.dense3 = DenseBlock(320, growth_rate=growth_rate, num_layers=32)  # increased from 24
        self.trans3 = TransitionLayer(320 + 32*growth_rate, 640)
        
        self.dense4 = DenseBlock(640, growth_rate=growth_rate, num_layers=20)  # increased from 16
        
        # Final classification
        final_channels = 640 + 20*growth_rate
        self.bn_final = nn.BatchNorm2d(final_channels)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(final_channels, num_classes)
        self.dropout = nn.Dropout(dropout_p)
    
    def forward(self, x):
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


class BirdCNN_v7d(nn.Module):
    """
    Dense network with dense blocks for feature reuse - Variant D.
    
    Similar to v7 but with modified transition layer compression and
    different layer distribution across blocks.
    
    Args:
        num_classes (int): Number of bird species classes to classify
        dropout_p (float, optional): Dropout probability for regularization. Defaults to 0.3
    """
    def __init__(self, num_classes, dropout_p=0.3):
        super(BirdCNN_v7d, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(1, 56, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(56)
        self.pool1 = nn.MaxPool2d(3, stride=2, padding=1)
        
        # Dense blocks with different layer distribution
        growth_rate = 28
        self.dense1 = DenseBlock(56, growth_rate=growth_rate, num_layers=5)
        self.trans1 = TransitionLayer(56 + 5*growth_rate, 112)  # different compression
        
        self.dense2 = DenseBlock(112, growth_rate=growth_rate, num_layers=10)
        self.trans2 = TransitionLayer(112 + 10*growth_rate, 224)
        
        self.dense3 = DenseBlock(224, growth_rate=growth_rate, num_layers=20)
        self.trans3 = TransitionLayer(224 + 20*growth_rate, 448)
        
        self.dense4 = DenseBlock(448, growth_rate=growth_rate, num_layers=14)
        
        # Final classification
        final_channels = 448 + 14*growth_rate
        self.bn_final = nn.BatchNorm2d(final_channels)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(final_channels, num_classes)
        self.dropout = nn.Dropout(dropout_p)
    
    def forward(self, x):
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


class BirdCNN_v7e(nn.Module):
    """
    Dense network with dense blocks for feature reuse - Variant E.
    
    Similar to v7 but with intermediate growth rate and balanced
    layer counts for optimal efficiency/performance trade-off.
    
    Args:
        num_classes (int): Number of bird species classes to classify
        dropout_p (float, optional): Dropout probability for regularization. Defaults to 0.45
    """
    def __init__(self, num_classes, dropout_p=0.45):
        super(BirdCNN_v7e, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(1, 72, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(72)
        self.pool1 = nn.MaxPool2d(3, stride=2, padding=1)
        
        # Dense blocks with balanced parameters
        growth_rate = 36
        self.dense1 = DenseBlock(72, growth_rate=growth_rate, num_layers=7)
        self.trans1 = TransitionLayer(72 + 7*growth_rate, 144)
        
        self.dense2 = DenseBlock(144, growth_rate=growth_rate, num_layers=14)
        self.trans2 = TransitionLayer(144 + 14*growth_rate, 288)
        
        self.dense3 = DenseBlock(288, growth_rate=growth_rate, num_layers=28)
        self.trans3 = TransitionLayer(288 + 28*growth_rate, 576)
        
        self.dense4 = DenseBlock(576, growth_rate=growth_rate, num_layers=18)
        
        # Final classification
        final_channels = 576 + 18*growth_rate
        self.bn_final = nn.BatchNorm2d(final_channels)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(final_channels, num_classes)
        self.dropout = nn.Dropout(dropout_p)
    
    def forward(self, x):
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

# add versions here