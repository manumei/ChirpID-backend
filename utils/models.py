# BASIC PROTOTYPE 

# TODO DECIRLE A CHATCGT QUE HAGA UNA CLASE DE MODELO DE PYTORCH USANDO LA SIGUIENTE ARQUITECTURA:
nn.Sequential(
    # Block 1
    nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),  # output: [16, 128, 128]
    nn.BatchNorm2d(16),
    nn.ReLU(),
    nn.MaxPool2d(2),  # [16, 64, 64]

    # Block 2
    nn.Conv2d(16, 32, kernel_size=3, padding=1),  # [32, 64, 64]
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.MaxPool2d(2),  # [32, 32, 32]

    # Block 3
    nn.Conv2d(32, 64, kernel_size=3, padding=1),  # [64, 32, 32]
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(2),  # [64, 16, 16]

    nn.Flatten(),  # [64 * 16 * 16]
    nn.Linear(64 * 16 * 16, 128),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(128, 28),  # 28 classes
    nn.LogSoftmax(dim=1)
)
