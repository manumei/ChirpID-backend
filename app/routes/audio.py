from flask import Blueprint, request, jsonify
import os
import uuid
import logging
from datetime import datetime
from werkzeug.utils import secure_filename
import sys
import numpy as np
import pandas as pd
import pathlib


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

audio_bp = Blueprint("audio", __name__)

ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac', 'm4a'}

def allowed_file(filename):
    return '.' in filename and \
            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_unique_filename(original_filename):
    """
    Generate a unique filename to prevent conflicts.
    Format: YYYYMMDD_HHMMSS_UUID_originalname.ext
    """
    # Get current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Generate short UUID (first 8 characters)
    unique_id = str(uuid.uuid4())[:8]
    
    # Secure the original filename
    safe_filename = secure_filename(original_filename or "recording.wav")
    
    # Split filename and extension
    name, ext = os.path.splitext(safe_filename)
    
    # Create unique filename
    unique_filename = f"{timestamp}_{unique_id}_{name}{ext}"
    
    return unique_filename

@audio_bp.route("/upload", methods=["POST"])
def upload_audio():
    try:
        client_ip = request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr)
        logger.info(f"Upload request received from {client_ip}")
        logger.info(f"Request headers: {dict(request.headers)}")
        logger.info(f"Files in request: {list(request.files.keys())}")
        
        if "file" not in request.files:
            logger.warning("No file part in request")
            return jsonify({"error": "No file part"}), 400

        file = request.files["file"]
        if file.filename == "":
            logger.warning("No selected file")
            return jsonify({"error": "No selected file"}), 400

        if not allowed_file(file.filename):
            logger.warning(f"File type not allowed: {file.filename}")
            return jsonify({"error": "File type not allowed. Supported formats: wav, mp3, ogg, flac, m4a"}), 400

        # Generate unique filename to prevent conflicts
        original_filename = file.filename or "recording.wav"
        unique_filename = generate_unique_filename(original_filename)
        
        uploads_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "uploads")
        if not os.path.exists(uploads_dir):
            os.makedirs(uploads_dir)
        save_path = os.path.join(uploads_dir, unique_filename)
        
        logger.info(f"Saving file: {original_filename} -> {unique_filename}")
        logger.info(f"Save path: {save_path}")
        
        file.save(save_path)
        
        logger.info("Processing audio...")
        logger.info(f"Starting audio processing for file: {save_path}")
        logger.info(f"File exists: {os.path.exists(save_path)}")
        logger.info(f"File size: {os.path.getsize(save_path) if os.path.exists(save_path) else 'N/A'} bytes")
        
        try:
            result = process_audio(save_path)
            logger.info("Audio processing completed successfully")
        except Exception as processing_error:
            logger.error(f"Audio processing failed: {str(processing_error)}", exc_info=True)
            raise
        response_data = {
            "success": True,
            "message": "Audio uploaded and processed successfully",
            "result": result,
            "original_filename": original_filename,
            "unique_filename": unique_filename,
            "upload_id": str(uuid.uuid4())
        }
        logger.info(f"Sending response: {response_data}")
        return jsonify(response_data)
        
    except Exception as e:
        error_message = str(e)
        logger.error(f"Error processing upload: {error_message}", exc_info=True)
        
        # Provide more specific error messages based on the error type
        if "load key 'v'" in error_message:
            user_message = "Model file is corrupted or incompatible. Please contact the administrator."
        elif "No such file or directory" in error_message or "not found" in error_message.lower():
            user_message = "Required model or mapping files are missing. Please contact the administrator."
        elif "No usable segments extracted" in error_message:
            user_message = "The audio file could not be processed. Please ensure it's a valid audio file with bird sounds."
        elif "File type not allowed" in error_message:
            user_message = error_message  # Already user-friendly
        elif "File size" in error_message:
            user_message = error_message  # Already user-friendly
        else:
            user_message = f"Audio processing failed: {error_message}"
        
        return jsonify({"error": user_message}), 500

@audio_bp.route("/files", methods=["GET"])
def list_uploaded_files():
    """List all uploaded audio files with their metadata"""
    try:
        uploads_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "uploads")
        if not os.path.exists(uploads_dir):
            return jsonify({"files": []})
        
        files = []
        for filename in os.listdir(uploads_dir):
            if allowed_file(filename):
                file_path = os.path.join(uploads_dir, filename)
                file_stat = os.stat(file_path)
                files.append({
                    "filename": filename,
                    "size": file_stat.st_size,
                    "created": datetime.fromtimestamp(file_stat.st_ctime).isoformat(),
                    "modified": datetime.fromtimestamp(file_stat.st_mtime).isoformat()
                })
        
        # Sort by creation time (newest first)
        files.sort(key=lambda x: x["created"], reverse=True)
        
        return jsonify({
            "success": True,
            "count": len(files),
            "files": files
        })
        
    except Exception as e:
        logger.error(f"Error listing files: {str(e)}", exc_info=True)
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@audio_bp.route("/cleanup", methods=["POST"])
def cleanup_old_files():
    """Remove uploaded files older than specified days (default: 1)"""
    try:
        days = request.json.get("days", 1) if request.json else 1
        uploads_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "uploads")
        
        if not os.path.exists(uploads_dir):
            return jsonify({"message": "No uploads directory found", "deleted_count": 0})
        
        cutoff_time = datetime.now().timestamp() - (days * 24 * 60 * 60)
        deleted_files = []
        
        for filename in os.listdir(uploads_dir):
            file_path = os.path.join(uploads_dir, filename)
            if os.path.isfile(file_path) and os.path.getctime(file_path) < cutoff_time:
                try:
                    os.remove(file_path)
                    deleted_files.append(filename)
                    logger.info(f"Deleted old file: {filename}")
                except Exception as e:
                    logger.warning(f"Failed to delete {filename}: {e}")
        
        logger.info(f"Cleanup completed. Deleted {len(deleted_files)} files older than {days} days")
        return jsonify({
            "success": True,
            "message": f"Cleaned up files older than {days} days",
            "deleted_count": len(deleted_files),
            "deleted_files": deleted_files
        })
        
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}", exc_info=True)
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


# FUNCTIONS FOR AUDIO PROCESSING AND INFERENCE
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils.inference import perform_audio_inference
from utils.final_models import BirdCNN_v5c

def predict_bird(audio_path, model_class, model_path, mapping_csv):
    """
    Predict bird species from audio file and display results with confidence.
    
    Args:
        audio_path (str): Path to the audio file to analyze
        model_class (type): Class of the CNN model to use (e.g., OldBirdCNN)
        model_path (str): Path to the .pth model weights file
        mapping_csv (str): Path to CSV file with columns: class_id, scientific_name, common_name
    
    Returns:
        dict: Dictionary containing predicted class_id, common_name, scientific_name, and confidence
        
    Raises:
        FileNotFoundError: If mapping_csv doesn't exist
        ValueError: If mapping CSV doesn't have required columns
    """
    logger.info(f"Starting predict_bird with audio_path: {audio_path}")
    logger.info(f"Model class: {model_class.__name__}")
    logger.info(f"Model path: {model_path}")
    logger.info(f"Mapping CSV: {mapping_csv}")
    
    # Validate mapping CSV file
    if not os.path.exists(mapping_csv):
        logger.error(f"Mapping CSV file not found: {mapping_csv}")
        raise FileNotFoundError(f"Mapping CSV file not found: {mapping_csv}")
    
    # Load mapping CSV
    try:
        logger.info("Loading mapping CSV...")
        mapping_df = pd.read_csv(mapping_csv)
        logger.info(f"Mapping CSV loaded successfully. Shape: {mapping_df.shape}")
        logger.info(f"Mapping CSV columns: {list(mapping_df.columns)}")
        
        required_columns = ['class_id', 'scientific_name', 'common_name']
        if not all(col in mapping_df.columns for col in required_columns):
            logger.error(f"Mapping CSV missing required columns. Has: {list(mapping_df.columns)}, Needs: {required_columns}")
            raise ValueError(f"Mapping CSV must contain columns: {required_columns}")
        
        logger.info(f"Number of classes in mapping: {len(mapping_df)}")
        
    except Exception as e:
        logger.error(f"Error reading mapping CSV: {e}", exc_info=True)
        raise ValueError(f"Error reading mapping CSV: {e}")
    
    # Get average probabilities from inference
    try:
        logger.info("Starting audio inference...")
        logger.info(f"Calling perform_audio_inference with:")
        logger.info(f"  - audio_path: {audio_path}")
        logger.info(f"  - model_class: {model_class}")
        logger.info(f"  - model_path: {model_path}")
        
        average_probabilities = perform_audio_inference(audio_path, model_class, model_path)
        logger.info(f"Audio inference completed. Probabilities shape: {average_probabilities.shape if hasattr(average_probabilities, 'shape') else len(average_probabilities)}")
        logger.info(f"Max probability: {np.max(average_probabilities):.4f}")
        
    except Exception as e:
        logger.error(f"Error during audio inference: {str(e)}", exc_info=True)
        raise
    
    # Find the class with highest probability
    predicted_class_id = np.argmax(average_probabilities)
    confidence = average_probabilities[predicted_class_id]
    
    logger.info(f"Predicted class ID: {predicted_class_id}")
    logger.info(f"Confidence: {confidence:.4f}")
    
    # Get bird information from mapping
    bird_info = mapping_df[mapping_df['class_id'] == predicted_class_id]
    
    if bird_info.empty:
        logger.error(f"Class ID {predicted_class_id} not found in mapping CSV")
        logger.info(f"Available class IDs in mapping: {sorted(mapping_df['class_id'].unique())}")
        raise ValueError(f"Class ID {predicted_class_id} not found in mapping CSV")
    
    common_name = bird_info.iloc[0]['common_name']
    scientific_name = bird_info.iloc[0]['scientific_name']
    
    logger.info(f"Prediction result: {common_name} ({scientific_name}) with confidence {confidence:.4f}")
    
    # Return structured results
    return {
        'species': common_name,
        'scientific_name': scientific_name,
        'confidence': float(confidence),
    }

def load_files():
    logger.info("Loading model and mapping files...")
    repo_root_path = pathlib.Path(__file__).resolve().parent.parent.parent
    model_path = repo_root_path / 'models' / 'bird_cnn.pth'
    mapping_csv = repo_root_path / 'mapping' / 'class_mapping.csv'
    model_path = str(model_path)
    mapping_path = str(mapping_csv)
    
    logger.info(f"Repository root path: {repo_root_path}")
    logger.info(f"Model path: {model_path}")
    logger.info(f"Mapping path: {mapping_path}")
    logger.info(f"Model file exists: {os.path.exists(model_path)}")
    logger.info(f"Mapping file exists: {os.path.exists(mapping_path)}")
    
    if os.path.exists(model_path):
        logger.info(f"Model file size: {os.path.getsize(model_path)} bytes")
    if os.path.exists(mapping_path):
        logger.info(f"Mapping file size: {os.path.getsize(mapping_path)} bytes")
    
    return model_path, mapping_path

def process_audio(path):
    """
    Test function to demonstrate inference on a sample audio file.
    
    Args:
        path (str): Path to the audio file to analyze
    
    Returns:
        dict: Inference results
    """
    logger.info(f"Starting process_audio for file: {path}")
    logger.info(f"Input file exists: {os.path.exists(path)}")
    
    if os.path.exists(path):
        logger.info(f"Input file size: {os.path.getsize(path)} bytes")
    
    try:
        model_class = BirdCNN_v5c
        logger.info(f"Using model class: {model_class.__name__}")
        
        model_path, mapping_path = load_files()
        
        logger.info("Calling predict_bird function...")
        result = predict_bird(path, model_class, model_path, mapping_path)
        logger.info(f"predict_bird completed successfully: {result}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error in process_audio: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    # Example usage
    audio_file = "../../database/audio/dev/XC114492.ogg"
    try:
        result = process_audio(audio_file)
        print("Inference Result:", result)
    except Exception as e:
        print(f"Error during inference: {e}")
