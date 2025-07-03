import os
import sys
import torch
import numpy as np
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
    
    # Load the checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Check if checkpoint is a full model or state dict
    if isinstance(checkpoint, torch.nn.Module):
        # If it's a full model, use it directly
        model = checkpoint
        # Ensure it matches the expected class
        if not isinstance(model, model_class):
            raise ValueError(f"Warning: Loaded model type {type(model)} doesn't match expected {model_class}")
    else:
        # If it's a state dict, load it into a new model instance
        model = model_class(num_classes=num_classes)
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

def vectors_to_tensor(segment_vector, device):
    """
    Convert a single segment vector to a tensor ready for FCNN inference.
    
    Args:
        segment_vector (numpy.ndarray): Single 1D numpy array (flattened spectrogram segment)
        device (torch.device): Device to move tensor to
    
    Returns:
        torch.Tensor: Tensor of shape (1, vector_length)
    """
    if segment_vector is None or segment_vector.size == 0:
        raise ValueError("Input segment vector is empty or None. Cannot convert to tensor.")
    
    # Add batch dimension
    vector = np.expand_dims(segment_vector, axis=0)  # (1, vector_length)
    
    # Convert to tensor and move to device
    tensor = torch.from_numpy(vector).float().to(device)
    
    return tensor

def perform_audio_inference(audio_path, model_class, model_path):
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
    
    # print(f"Starting inference for: {audio_path}")
    
    # Step 1: Extract segment_matrices from audio using audio_process
    segment_matrices = audio_process(audio_path)
    
    if not segment_matrices:
        raise ValueError(f"No usable segments extracted from audio file: {audio_path}")
    
    # print(f"Extracted {len(segment_matrices)} segments for inference")
    
    # Step 2: Load the model
    model, device = load_model_weights(model_class, model_path, num_classes=NUM_CLASSES)
    # print(f"Model loaded on device: {device}")
    
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
    
    # print(f"Inference completed. Processed {len(all_probabilities)} segments")
    # print(f"Average probabilities calculated for NUM_CLASSES classes")
    
    # Return as list for classes 0-26
    return average_probabilities.tolist()

def perform_audio_inference_fcnn(audio_path, model_class, model_path):
    """
    Perform inference on an audio file using a trained FCNN model.
    
    This function processes an audio file by:
    1. Extracting grayscale log-mel-spectrogram segment_matrices for each usable segment
    2. Flattening each matrix into a vector for FCNN input
    3. Loading the pretrained FCNN model
    4. Running inference on each flattened segment individually
    5. Returning the average softmax probabilities across all segments
    
    Args:
        audio_path (str): Path to the audio file to analyze
        model_class (type): Class of the FCNN model to use
        model_path (str): Path to the .pth model weights file
    
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
    
    # print(f"Starting FCNN inference for: {audio_path}")
    
    # Step 1: Extract segment_matrices from audio using audio_process
    segment_matrices = audio_process(audio_path)
    
    if not segment_matrices:
        raise ValueError(f"No usable segments extracted from audio file: {audio_path}")
    
    # print(f"Extracted {len(segment_matrices)} segments for FCNN inference")
    
    # Step 2: Load the model
    model, device = load_model_weights(model_class, model_path, num_classes=NUM_CLASSES)
    # print(f"FCNN model loaded on device: {device}")
    
    # Step 3 & 4: Process each segment individually and perform inference
    all_probabilities = []
    
    with torch.no_grad():
        for i, segment_matrix in enumerate(segment_matrices):
            # Flatten the matrix into a vector for FCNN
            segment_vector = segment_matrix.flatten()
            
            # Convert single segment vector to tensor
            input_tensor = vectors_to_tensor(segment_vector, device)
            
            # Run inference on the single flattened segment
            logits = model(input_tensor)
            probabilities = torch.softmax(logits, dim=1)  # Shape: (1, NUM_CLASSES)
            
            # Convert to numpy and store
            probabilities_np = probabilities.cpu().numpy()[0]  # Shape: (NUM_CLASSES,)
            all_probabilities.append(probabilities_np)
    
    # Step 5: Calculate average probabilities across all segments
    all_probabilities = np.array(all_probabilities)  # Shape: (num_segments, NUM_CLASSES)
    average_probabilities = np.mean(all_probabilities, axis=0)  # Shape: (NUM_CLASSES,)
    
    # print(f"FCNN inference completed. Processed {len(all_probabilities)} segments")
    # print(f"Average probabilities calculated for NUM_CLASSES classes")
    
    # Return as list for classes 0-26
    return average_probabilities.tolist()

def infer_model_direct(audio_path, model):
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
    
    Returns:
        list: Average softmax probabilities for each of the NUM_CLASSES classes (indices 0-26)
        
    Raises:
        FileNotFoundError: If audio_path or model_path don't exist
        ValueError: If no usable segments are extracted from the audio
    """
    
    # print(f"Starting inference for: {audio_path}")
    
    # Step 1: Extract segment_matrices from audio using audio_process
    segment_matrices = audio_process(audio_path)
    
    if not segment_matrices:
        raise ValueError(f"No usable segments extracted from audio file: {audio_path}")
    
    # Step 3 & 4: Process each segment individually and perform inference
    all_probabilities = []
    
    with torch.no_grad():
        for i, segment_matrix in enumerate(segment_matrices):
            # Convert single segment to tensor
            input_tensor = matrices_to_tensor(segment_matrix, device='cuda')
            
            # Run inference on the single segment
            logits = model(input_tensor)
            probabilities = torch.softmax(logits, dim=1)  # Shape: (1, NUM_CLASSES)
            
            # Convert to numpy and store
            probabilities_np = probabilities.cpu().numpy()[0]  # Shape: (NUM_CLASSES,)
            all_probabilities.append(probabilities_np)
    
    # Step 5: Calculate average probabilities across all segments
    all_probabilities = np.array(all_probabilities)  # Shape: (num_segments, NUM_CLASSES)
    average_probabilities = np.mean(all_probabilities, axis=0)  # Shape: (NUM_CLASSES,)
    
    # print(f"Inference completed. Processed {len(all_probabilities)} segments")
    # print(f"Average probabilities calculated for NUM_CLASSES classes")
    
    # Return as list for classes 0-26
    return average_probabilities.tolist()
