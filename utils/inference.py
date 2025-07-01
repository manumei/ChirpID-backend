import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from utils.oldmodels import OldBirdCNN
from utils.data_processing import audio_process

NUM_CLASSES = 33  # Default number of classes, can be overridden

def load_model_weights(model_class, model_path, num_classes=NUM_CLASSES, device=None):
    """
    Load the CNN model with pretrained weights.
    
    Args:
        model_path (str): Path to the .pth model weights file
        num_classes (int): Number of classes (default: NUM_CLASSES)
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

def matrices_to_tensor(segment_matrix, device):
    """
    Convert a single segment matrix to a tensor ready for CNN inference.
    
    Args:
        segment_matrix (numpy.ndarray): Single 2D numpy array (grayscale spectrogram segment)
        device (torch.device): Device to move tensor to
    
    Returns:
        torch.Tensor: Tensor of shape (1, 1, height, width)
    """
    if segment_matrix is None or segment_matrix.size == 0:
        raise ValueError("Input segment matrix is empty or None. Cannot convert to tensor.")
    
    # Add batch and channel dimensions
    matrix = np.expand_dims(segment_matrix, axis=0)  # (1, height, width)
    matrix = np.expand_dims(matrix, axis=0)  # (1, 1, height, width)
    
    # Convert to tensor and move to device
    tensor = torch.from_numpy(matrix).float().to(device)
    
    return tensor

def perform_audio_inference(audio_path, model_class, model_path, reduce_noise=True):
    """
    Perform inference on an audio file using a trained CNN model.
    
    This function processes an audio file by:
    1. Extracting grayscale log-mel-spectrogram segment_matrices for each usable segment
    2. Loading the pretrained CNN model
    3. Running inference on each segment individually
    4. Returning the average softmax probabilities across all segments
    
    Args:
        audio_path (str): Path to the audio file to analyze
        model_class (type): Class of the CNN model to use (e.g., OldBirdCNN)
        model_path (str): Path to the .pth model weights file
        reduce_noise (bool): Whether to apply noise reduction to audio segments
    
    Returns:
        list: Average softmax probabilities for each of the NUM_CLASSES classes (indices 0-26)
        
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
    
    # Step 1: Extract segment_matrices from audio using audio_process
    segment_matrices = audio_process(audio_path, reduce_noise=reduce_noise)
    
    if not segment_matrices:
        raise ValueError(f"No usable segments extracted from audio file: {audio_path}")
    
    print(f"Extracted {len(segment_matrices)} segments for inference")
    
    # Step 2: Load the model
    model, device = load_model_weights(model_class, model_path, num_classes=NUM_CLASSES)
    print(f"Model loaded on device: {device}")
    
    # Step 3 & 4: Process each segment individually and perform inference
    all_probabilities = []
    
    with torch.no_grad():
        for i, segment_matrix in enumerate(segment_matrices):
            # Convert single segment to tensor
            input_tensor = matrices_to_tensor(segment_matrix, device)
            
            # Run inference on the single segment
            logits = model(input_tensor)
            probabilities = torch.softmax(logits, dim=1)  # Shape: (1, NUM_CLASSES)
            
            # Convert to numpy and store
            probabilities_np = probabilities.cpu().numpy()[0]  # Shape: (NUM_CLASSES,)
            all_probabilities.append(probabilities_np)
    
    # Step 5: Calculate average probabilities across all segments
    all_probabilities = np.array(all_probabilities)  # Shape: (num_segments, NUM_CLASSES)
    average_probabilities = np.mean(all_probabilities, axis=0)  # Shape: (NUM_CLASSES,)
    
    print(f"Inference completed. Processed {len(all_probabilities)} segments")
    print(f"Average probabilities calculated for NUM_CLASSES classes")
    
    # Return as list for classes 0-26
    return average_probabilities.tolist()

# # This function is only for fancy front-end reasons, to be ignored in Model Testing, don't call it in ModelTesting.ipynb, the CSV mappings are slow and inefficient.
# def predict_bird(audio_path, model_class, model_weights_path, mapping_csv, reduce_noise=True):
#     """
#     Predict bird species from audio file and display results with confidence.
    
#     Args:
#         audio_path (str): Path to the audio file to analyze
#         model_class (type): Class of the CNN model to use (e.g., OldBirdCNN)
#         model_weights_path (str): Path to the .pth model weights file
#         mapping_csv (str): Path to CSV file with columns: class_id, scientific_name, common_name
#         reduce_noise (bool): Whether to apply noise reduction to audio segments
    
#     Returns:
#         dict: Dictionary containing predicted class_id, common_name, scientific_name, and confidence
        
#     Raises:
#         FileNotFoundError: If mapping_csv doesn't exist
#         ValueError: If mapping CSV doesn't have required columns
#     """
    
#     # Validate mapping CSV file
#     if not os.path.exists(mapping_csv):
#         raise FileNotFoundError(f"Mapping CSV file not found: {mapping_csv}")
    
#     # Load mapping CSV
#     try:
#         mapping_df = pd.read_csv(mapping_csv)
#         required_columns = ['class_id', 'scientific_name', 'common_name']
#         if not all(col in mapping_df.columns for col in required_columns):
#             raise ValueError(f"Mapping CSV must contain columns: {required_columns}")
#     except Exception as e:
#         raise ValueError(f"Error reading mapping CSV: {e}")
    
#     # Get average probabilities from inference
#     average_probabilities = perform_audio_inference(audio_path, model_class, model_weights_path, reduce_noise)
    
#     # Find the class with highest probability
#     predicted_class_id = np.argmax(average_probabilities)
#     confidence = average_probabilities[predicted_class_id]
#     confidence_pct = confidence * 100
    
#     # Get bird information from mapping
#     bird_info = mapping_df[mapping_df['class_id'] == predicted_class_id]
    
#     if bird_info.empty:
#         raise ValueError(f"Class ID {predicted_class_id} not found in mapping CSV")
    
#     common_name = bird_info.iloc[0]['common_name']
#     scientific_name = bird_info.iloc[0]['scientific_name']
    
#     # # Print results
#     # print(f"Predicted Bird is {common_name} ({scientific_name}), confidence of {confidence_pct:.2f}%")
    
#     # Return structured results
#     return {
#         'class_id': int(predicted_class_id),
#         'common_name': common_name,
#         'scientific_name': scientific_name,
#         'confidence': float(confidence),
#         'confidence_prct': float(confidence_pct)
#     }
