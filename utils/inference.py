import os
import sys
import torch
import numpy as np
import logging
from utils.data_processing import audio_process

# Configure logging for this module
logger = logging.getLogger(__name__)

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
    logger.info("Step 2: Loading model...")
    logger.info(f"Loading model weights from: {model_path}")
    logger.info(f"Model class: {model_class.__name__}")
    logger.info(f"Number of classes: {num_classes}")
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info(f"Using device: {device}")
    
    # Check model file exists and get info
    if not os.path.exists(model_path):
        logger.error(f"Model file does not exist: {model_path}")
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model_size = os.path.getsize(model_path)
    logger.info(f"Model file size: {model_size} bytes ({model_size / (1024*1024):.2f} MB)")
    
    try:
        # Add extensive diagnostics for model file
        logger.info("=== MODEL FILE DIAGNOSTICS ===")
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"Model path exists: {os.path.exists(model_path)}")
        logger.info(f"Model path is file: {os.path.isfile(model_path)}")
        logger.info(f"Model file readable: {os.access(model_path, os.R_OK)}")
        
        # Read first few bytes to check file integrity
        try:
            with open(model_path, 'rb') as f:
                first_bytes = f.read(16)
                logger.info(f"First 16 bytes of model file: {first_bytes.hex()}")
                
                # Check if it looks like a valid PyTorch file (starts with PK for zip format)
                if first_bytes.startswith(b'PK'):
                    logger.info("File appears to be a valid ZIP/PyTorch format")
                else:
                    logger.warning("File does not appear to be in standard PyTorch format")
        except Exception as read_error:
            logger.error(f"Cannot read model file: {read_error}")
        
        # Load the checkpoint with additional error handling
        logger.info("Loading PyTorch checkpoint...")
        checkpoint = None
        load_method_used = None
        
        # Try multiple loading methods for maximum compatibility
        # PyTorch 2.6+ changed default weights_only from False to True
        # We need to explicitly set weights_only=False for older model files
        loading_methods = [
            ("explicit_weights_false", lambda: torch.load(model_path, map_location=device, weights_only=False)),
            ("with_pickle_module", lambda: torch.load(model_path, map_location=device, weights_only=False, pickle_module=__import__('pickle'))),
            ("legacy_explicit", lambda: torch.load(model_path, map_location=device, weights_only=False))
        ]
        
        last_error = None
        for method_name, load_func in loading_methods:
            try:
                logger.info(f"Trying loading method: {method_name}")
                checkpoint = load_func()
                load_method_used = method_name
                logger.info(f"Successfully loaded with method: {method_name}")
                break
            except Exception as method_error:
                logger.warning(f"Loading method '{method_name}' failed: {method_error}")
                last_error = method_error
                continue
        
        if checkpoint is None:
            logger.error("All loading methods failed!")
            if last_error:
                raise last_error
            else:
                raise RuntimeError("Unable to load model with any method")
        
        logger.info(f"Checkpoint loaded successfully with method '{load_method_used}'. Type: {type(checkpoint)}")
        
        # Check if checkpoint is a full model or state dict
        if isinstance(checkpoint, torch.nn.Module):
            logger.info("Checkpoint contains a full model")
            # If it's a full model, use it directly
            model = checkpoint
            # Ensure it matches the expected class
            if not isinstance(model, model_class):
                logger.warning(f"Loaded model type {type(model)} doesn't match expected {model_class}")
                raise ValueError(f"Warning: Loaded model type {type(model)} doesn't match expected {model_class}")
        else:
            logger.info("Checkpoint contains state dict, creating new model instance")
            # If it's a state dict, load it into a new model instance
            model = model_class(num_classes=num_classes)
            logger.info(f"Created model instance: {type(model)}")
            model.load_state_dict(checkpoint)
            logger.info("State dict loaded successfully")
        
        # Set to evaluation mode and move to device
        model.eval()
        model.to(device)
        logger.info("Model set to evaluation mode and moved to device")
        logger.info(f"Model loaded successfully on device: {device}")
        return model, device
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}", exc_info=True)
        # Check if it's the specific "load key 'v'" error
        if "load key 'v'" in str(e):
            logger.error("This error typically indicates a corrupted or incompatible model file")
            logger.error("The model file may have been saved with a different PyTorch version or is corrupted")
            logger.error("Common causes:")
            logger.error("1. Model file was corrupted during transfer to server")
            logger.error("2. PyTorch version mismatch between training and inference environments")
            logger.error("3. Model file was created with an incompatible Python/PyTorch version")
            logger.error("4. File system corruption or permissions issue")
            logger.error(f"Current PyTorch version: {torch.__version__}")
        elif "Weights only load failed" in str(e) or "WeightsUnpickler error" in str(e):
            logger.error("PyTorch 2.6+ weights_only compatibility issue detected")
            logger.error("This happens when loading models saved with older PyTorch versions")
            logger.error("The model file contains objects that require weights_only=False")
            logger.error("This is safe if you trust the model file source (which you should for your own models)")
            logger.error(f"Current PyTorch version: {torch.__version__}")
            logger.error("Solution: Ensure all torch.load() calls use weights_only=False explicitly")
        raise

def eval_on_model(segment_matrices, model, device):
    # Step 3 & 4: Process each segment individually and perform inference
    logger.info("Step 3: Processing segments and running inference...")
    all_probabilities = []
    
    with torch.no_grad():
        for i, segment_matrix in enumerate(segment_matrices):
            logger.debug(f"Processing segment {i+1}/{len(segment_matrices)}")
            
            # Convert single segment to tensor
            input_tensor = matrices_to_tensor(segment_matrix, device)
            logger.debug(f"Segment {i+1} tensor shape: {input_tensor.shape}")
            
            # Run inference on the single segment
            logits = model(input_tensor)
            probabilities = torch.softmax(logits, dim=1)  # Shape: (1, NUM_CLASSES)
            
            # Convert to numpy and store
            probabilities_np = probabilities.cpu().numpy()[0]  # Shape: (NUM_CLASSES,)
            all_probabilities.append(probabilities_np)
            
            if i == 0:  # Log details for first segment
                logger.info(f"First segment logits shape: {logits.shape}")
                logger.info(f"First segment probabilities shape: {probabilities.shape}")
                logger.info(f"First segment max probability: {np.max(probabilities_np):.4f}")
    return all_probabilities

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

def validate_paths(audio_path, model_path):
    if not os.path.exists(audio_path):
        logger.error(f"Audio file not found: {audio_path}")
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if not os.path.exists(model_path):
        logger.error(f"Model weights file not found: {model_path}")
        raise FileNotFoundError(f"Model weights file not found: {model_path}")
    logger.info(f"Audio file size: {os.path.getsize(audio_path)} bytes")

def validate_segments(segment_matrices, audio_path):
    if not segment_matrices:
        logger.error(f"No usable segments extracted from audio file: {audio_path}")
        raise ValueError(f"No usable segments extracted from audio file: {audio_path}")
    
    logger.info(f"Successfully extracted {len(segment_matrices)} segments for inference")

def success_logs(probs, avg_probs):
    logger.info(f"Inference completed successfully!")
    logger.info(f"Processed {len(probs)} segments")
    logger.info(f"Final average probabilities calculated for {len(avg_probs)} classes")
    logger.info(f"Max final probability: {np.max(avg_probs):.4f}")
    logger.info(f"Predicted class: {np.argmax(avg_probs)}")

def get_avg_probabilities(all_probabilities):
    logger.info("Step 4: Calculating average probabilities...")
    all_probabilities = np.array(all_probabilities)  # Shape: (num_segments, NUM_CLASSES)
    average_probabilities = np.mean(all_probabilities, axis=0)  # Shape: (NUM_CLASSES,)
    
    success_logs(all_probabilities, average_probabilities)
    return average_probabilities.tolist()

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

    # First Validate input files
    validate_paths(audio_path, model_path)
    
    try:
        # Step 1: Extract segment_matrices from audio using audio_process
        segment_matrices = audio_process(audio_path)
        validate_segments(segment_matrices, audio_path)
        
        # Step 2: Load the model
        model, device = load_model_weights(model_class, model_path, num_classes=NUM_CLASSES)
        
        # Step 3: Evaluate with Model
        all_probabilities = eval_on_model(segment_matrices, model, device)
        
        # Step 5: Calculate average probabilities across all segments
        avg_probs_list = get_avg_probabilities(all_probabilities)
        
        return avg_probs_list
        
    except Exception as e:
        logger.error(f"Error during audio inference: {str(e)}", exc_info=True)
        raise e

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
