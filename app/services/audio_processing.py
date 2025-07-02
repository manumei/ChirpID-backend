import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add the parent directory to sys.path to import utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.oldmodels import OldBirdCNN
from utils.data_processing import audio_process

def load_model_weights(model_class, model_path, num_classes=28, device=None):
    """
    Load the CNN model with pretrained weights.
    
    Args:
        model_path (str): Path to the .pth model weights file
        num_classes (int): Number of classes (default: 28)
        device (torch.device): Device to load the model on
    
    Returns:
        torch.nn.Module: Loaded model ready for inference
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize the model
    model = model_class(num_classes=num_classes)
    
    # Load the weights
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
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

def plot_segment_probabilities(all_probabilities, save_path=None):
    """
    Plot probabilities for each segment across all classes. Creates one plot per segment.
    
    Args:
        all_probabilities (numpy.ndarray): Array of shape (num_segments, 28) containing probabilities
        save_path (str, optional): Directory path to save the plots. If None, displays the plots.
    """
    num_segments = all_probabilities.shape[0]
    num_classes = all_probabilities.shape[1]
    
    for segment_idx in range(num_segments):
        plt.figure(figsize=(12, 6))
        
        # Get probabilities for this segment
        segment_probs = all_probabilities[segment_idx]
        class_indices = np.arange(num_classes)
        
        # Create bar plot
        bars = plt.bar(class_indices, segment_probs, color='skyblue', alpha=0.7)
        
        # Highlight the highest probability
        max_prob_idx = np.argmax(segment_probs)
        bars[max_prob_idx].set_color('orange')
        
        plt.title(f'Segment {segment_idx + 1} - Class Probabilities')
        plt.xlabel('Bird Class ID')
        plt.ylabel('Probability')
        plt.xticks(class_indices)
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        
        # Add probability values on top of bars for highest probabilities
        for i, prob in enumerate(segment_probs):
            if prob > 0.1:  # Only show labels for probabilities > 10%
                plt.text(i, prob + 0.01, f'{prob:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            plot_filename = os.path.join(save_path, f'segment_{segment_idx + 1}_probabilities.png')
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"Plot for segment {segment_idx + 1} saved to: {plot_filename}")
        else:
            plt.show()
        
        plt.close()
    
    print(f"Created {num_segments} probability plots")

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
        list: Average softmax probabilities for each of the 28 classes (indices 0-26)
        
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
    model, device = load_model_weights(model_class, model_path, num_classes=28)
    print(f"Model loaded on device: {device}")
    
    # Step 3 & 4: Process each segment individually and perform inference
    all_probabilities = []
    
    with torch.no_grad():
        for i, segment_matrix in enumerate(segment_matrices):
            # Convert single segment to tensor
            input_tensor = matrices_to_tensor(segment_matrix, device)
            
            # Run inference on the single segment
            logits = model(input_tensor)
            probabilities = torch.softmax(logits, dim=1)  # Shape: (1, 28)
            
            # Convert to numpy and store
            probabilities_np = probabilities.cpu().numpy()[0]  # Shape: (28,)
            all_probabilities.append(probabilities_np)
    
    # Step 5: Calculate average probabilities across all segments
    all_probabilities = np.array(all_probabilities)  # Shape: (num_segments, 28)
    average_probabilities = np.mean(all_probabilities, axis=0)  # Shape: (28,)
    
    print(f"Inference completed. Processed {len(all_probabilities)} segments")
    print(f"Average probabilities calculated for 28 classes")
    
    # Plot segment probabilities
    plot_segment_probabilities(all_probabilities)
    
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
        'species': common_name,
        'scientific_name': scientific_name,
        'confidence': float(confidence),
    }

def inferencia_prueba(path):
    """
    Test function to demonstrate inference on a sample audio file.
    
    Args:
        path (str): Path to the audio file to analyze
    
    Returns:
        dict: Inference results
    """
    model_class = OldBirdCNN
    repo_root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_path = os.path.join(repo_root_path, 'models', 'bird_cnn.pth')
    mapping_csv = os.path.join(repo_root_path, 'database', 'meta', 'class_mapping.csv')
    
    return predict_bird(path, model_class, model_path, mapping_csv, reduce_noise=True)

path = os.path.join('database', 'audio', 'dev', 'XC73767.ogg')
if __name__ == "__main__":
    # Example usage
    result = inferencia_prueba(path)
    print(result)