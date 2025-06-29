import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from utils.oldmodels import OldBirdCNN
from utils.data_processing import audio_process

def load_model_weights(model_class, model_path, num_classes=27, device=None):
    """
    Load the CNN model with pretrained weights.
    
    Args:
        model_path (str): Path to the .pth model weights file
        num_classes (int): Number of classes (default: 27)
        device (torch.device): Device to load the model on
    
    Returns:
        torch.nn.Module: Loaded model ready for inference
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize the model
    model = model_class(num_classes=num_classes)
    
    # Load the weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    
    # Set to evaluation mode and move to device
    model.eval()
    model.to(device)
    
    return model, device

def matrices_to_tensor(matrices, device):
    """
    Convert list of matrices to a tensor batch ready for CNN inference.
    
    Args:
        matrices (list): List of 2D numpy arrays (grayscale spectrogram matrices)
        device (torch.device): Device to move tensor to
    
    Returns:
        torch.Tensor: Tensor of shape (batch_size, 1, height, width)
    """
    if not matrices:
        raise ValueError("Input matrices list is empty. Cannot convert to tensor.")
    
    # Convert to numpy array and add channel dimension
    batch = np.array(matrices)  # (batch_size, height, width)
    batch = np.expand_dims(batch, axis=1)  # (batch_size, 1, height, width)
    
    # Convert to tensor and move to device
    tensor = torch.from_numpy(batch).float().to(device)
    
    return tensor

def perform_audio_inference(audio_path, model_class, model_path, reduce_noise=True):
    """
    Perform inference on an audio file using a trained CNN model.
    
    This function processes an audio file by:
    1. Extracting grayscale log-mel-spectrogram matrices for each usable segment
    2. Loading the pretrained CNN model
    3. Running inference on each segment
    4. Returning the average softmax probabilities across all segments
    
    Args:
        audio_path (str): Path to the audio file to analyze
        model_class (type): Class of the CNN model to use (e.g., OldBirdCNN)
        model_path (str): Path to the .pth model weights file
        reduce_noise (bool): Whether to apply noise reduction to audio segments
    
    Returns:
        list: Average softmax probabilities for each of the 27 classes (indices 0-26)
        
    Raises:
        FileNotFoundError: If audio_path or model_path don't exist
        ValueError: If no usable segments are extracted from the audio
    """
    
    # Validate input files
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights file not found: {model_path}")
    
    print(f"Starting inference for: {audio_path}")
    
    # Step 1: Extract matrices from audio using audio_process
    matrices = audio_process(audio_path, reduce_noise=reduce_noise)
    
    if not matrices:
        raise ValueError(f"No usable segments extracted from audio file: {audio_path}")
    
    print(f"Extracted {len(matrices)} segments for inference")
    
    # Step 2: Load the model
    model, device = load_model_weights(model_class, model_path, num_classes=27)
    print(f"Model loaded on device: {device}")
    
    # Step 3: Convert matrices to tensor batch
    input_tensor = matrices_to_tensor(matrices, device)
    
    # Step 4: Perform inference
    all_probabilities = []
    
    with torch.no_grad():
        # Run inference on the batch
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)  # Shape: (num_segments, 27)
        
        # Convert to numpy for easier handling
        probabilities_np = probabilities.cpu().numpy()
        all_probabilities.extend(probabilities_np)
    
    # Step 5: Calculate average probabilities across all segments
    all_probabilities = np.array(all_probabilities)  # Shape: (num_segments, 27)
    average_probabilities = np.mean(all_probabilities, axis=0)  # Shape: (27,)
    
    print(f"Inference completed. Processed {len(all_probabilities)} segments")
    print(f"Average probabilities calculated for 27 classes")
    
    # Return as list for classes 0-26
    return average_probabilities.tolist()

def predict_bird(audio_path, model_class, model_path, mapping_csv, reduce_noise=True):
    """
    Predict bird species from audio file and display results with confidence.
    
    Args:
        audio_path (str): Path to the audio file to analyze
        model_class (type): Class of the CNN model to use (e.g., OldBirdCNN)
        model_path (str): Path to the .pth model weights file
        mapping_csv (str): Path to CSV file with columns: class_id, scientific_name, common_name
        reduce_noise (bool): Whether to apply noise reduction to audio segments
    
    Returns:
        dict: Dictionary containing predicted class_id, common_name, scientific_name, and confidence
        
    Raises:
        FileNotFoundError: If mapping_csv doesn't exist
        ValueError: If mapping CSV doesn't have required columns
    """
    
    # Validate mapping CSV file
    if not os.path.exists(mapping_csv):
        raise FileNotFoundError(f"Mapping CSV file not found: {mapping_csv}")
    
    # Load mapping CSV
    try:
        mapping_df = pd.read_csv(mapping_csv)
        required_columns = ['class_id', 'scientific_name', 'common_name']
        if not all(col in mapping_df.columns for col in required_columns):
            raise ValueError(f"Mapping CSV must contain columns: {required_columns}")
    except Exception as e:
        raise ValueError(f"Error reading mapping CSV: {e}")
    
    # Get average probabilities from inference
    average_probabilities = perform_audio_inference(audio_path, model_class, model_path, reduce_noise)
    
    # Find the class with highest probability
    predicted_class_id = np.argmax(average_probabilities)
    confidence = average_probabilities[predicted_class_id]
    confidence_pct = confidence * 100
    
    # Get bird information from mapping
    bird_info = mapping_df[mapping_df['class_id'] == predicted_class_id]
    
    if bird_info.empty:
        raise ValueError(f"Class ID {predicted_class_id} not found in mapping CSV")
    
    common_name = bird_info.iloc[0]['common_name']
    scientific_name = bird_info.iloc[0]['scientific_name']
    
    # # Print results
    # print(f"Predicted Bird is {common_name} ({scientific_name}), confidence of {confidence_pct:.2f}%")
    
    # Return structured results
    return {
        'class_id': int(predicted_class_id),
        'common_name': common_name,
        'scientific_name': scientific_name,
        'confidence': float(confidence),
        'confidence_prct': float(confidence_pct)
    }
