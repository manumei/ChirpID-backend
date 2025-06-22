import os
import sys
import torch
import torch.nn as nn
import numpy as np

# Add the parent directory to sys.path to import utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from utils import util, models

# Temporary class definition to match the saved model weights
class BirdCNN(nn.Module):
    def __init__(self, num_classes=28, dropout_p=0.3):
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

def process_audio(file_path):
    """
    Process audio file using the trained CNN model for bird identification.
    
    Args:
        file_path (str): Path to the audio file
        
    Returns:
        int: Predicted class_id for the bird species
    """
    try:
        # Step 1: Process audio file to get spectrogram matrices
        # The audio_process function handles:
        # - Loading audio with librosa
        # - Segmenting into 5-second chunks with RMS filtering
        # - Optional noise reduction
        # - Converting to mel spectrograms
        matrices = util.audio_process(
            file_path, 
            reduce_noise=True, 
            sr=32000, 
            segment_sec=5.0, 
            frame_len=2048, 
            hop_len=512, 
            mels=224, 
            nfft=2048, 
            thresh=0.75
        )
        
        if not matrices or len(matrices) == 0:
            print(f"No valid audio segments found in {file_path}")
            return None
            
        # Step 2: Load the trained CNN model
        model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'bird_cnn.pth')
        
        if not os.path.exists(model_path):
            print(f"Model file not found at {model_path}")
            return None
              # Determine number of classes (based on notebooks, appears to be around 31-33)
        # You should update this with the exact number from your papersheet
        num_classes = 28  # Updated to match the saved model
        
        # Initialize model using the local BirdCNN class
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = BirdCNN(num_classes=num_classes)  # Use local class
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        # Step 3: Prepare matrices for prediction
        # Convert matrices to tensors and add batch and channel dimensions
        tensors = []
        for matrix in matrices:
            # matrix shape should be (224, 313) - (height, width)
            # Convert to tensor and normalize to [0, 1]
            tensor = torch.tensor(matrix, dtype=torch.float32) / 255.0
            # Add channel dimension: (1, 224, 313)
            tensor = tensor.unsqueeze(0)
            tensors.append(tensor)
        
        # Stack all segments into a batch
        batch_tensor = torch.stack(tensors).to(device)  # Shape: (num_segments, 1, 224, 313)
        
        # Step 4: Get predictions
        with torch.no_grad():
            outputs = model(batch_tensor)
            # Get probabilities for each segment
            probabilities = torch.softmax(outputs, dim=1)
            
            # Aggregate predictions across all segments (mean)
            avg_probabilities = torch.mean(probabilities, dim=0)
            
            # Get the predicted class
            predicted_class = torch.argmax(avg_probabilities).item()
            
        print(f"Processed {len(matrices)} segments from {file_path}")
        print(f"Predicted class: {predicted_class}")
        
        return predicted_class
        
    except Exception as e:
        print(f"Error processing audio file {file_path}: {str(e)}")
        return None
