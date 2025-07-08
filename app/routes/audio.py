from flask import Blueprint, request, jsonify
import os
import numpy as np
import pandas as pd
import uuid
import logging
from datetime import datetime
from werkzeug.utils import secure_filename
import sys
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
            user_message = (
                "Model file is corrupted or incompatible. This usually indicates:\n"
                "• PyTorch version mismatch between training and inference environments\n"
                "• Model file corruption during server deployment\n"
                "• Incompatible model file format\n"
                "Please contact the administrator to re-deploy the model file."
            )
        elif "No such file or directory" in error_message or "not found" in error_message.lower():
            user_message = "Required model or mapping files are missing. Please contact the administrator."
        elif "No usable segments extracted" in error_message:
            user_message = "The audio file could not be processed. Please ensure it's a valid audio file with bird sounds."
        elif "File type not allowed" in error_message:
            user_message = error_message  # Already user-friendly
        elif "File size" in error_message:
            user_message = error_message  # Already user-friendly
        elif "corrupted" in error_message.lower() or "incompatible" in error_message.lower():
            user_message = "Model file appears to be corrupted or incompatible. Please contact the administrator."
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
        file_size = os.path.getsize(model_path)
        logger.info(f"Model file size: {file_size} bytes ({file_size / (1024*1024):.2f} MB)")
        
        # Additional file integrity checks
        try:
            with open(model_path, 'rb') as f:
                first_bytes = f.read(16)
                logger.info(f"Model file first 16 bytes: {first_bytes.hex()}")
                if not first_bytes.startswith(b'PK'):
                    logger.warning("Model file does not appear to be in standard PyTorch format")
        except Exception as e:
            logger.error(f"Cannot read model file for integrity check: {e}")
    else:
        logger.error(f"Model file does not exist at: {model_path}")
        
    if os.path.exists(mapping_path):
        logger.info(f"Mapping file size: {os.path.getsize(mapping_path)} bytes")
    else:
        logger.error(f"Mapping file does not exist at: {mapping_path}")
    
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
    try:
        average_probabilities = perform_audio_inference(audio_path, model_class, model_path)
        
    except Exception as e:
        raise
    
    # Find the class with highest probability
    predicted_class_id = np.argmax(average_probabilities)
    confidence = average_probabilities[predicted_class_id]
    
    # Get bird information from mapping
    bird_info = mapping_df[mapping_df['class_id'] == predicted_class_id]
    
    if bird_info.empty:
        raise ValueError(f"Class ID {predicted_class_id} not found in mapping CSV")
    
    common_name = bird_info.iloc[0]['common_name']
    scientific_name = bird_info.iloc[0]['scientific_name']
    
    # Return structured results
    return {
        'species': common_name,
        'scientific_name': scientific_name,
        'confidence': float(confidence),
    }

@audio_bp.route("/health", methods=["GET"])
def health_check():
    """
    Health check endpoint that verifies model and mapping files are accessible.
    This helps diagnose issues before actual audio processing.
    """
    try:
        logger.info("Starting health check...")
        
        # Check if files exist and are accessible
        model_path, mapping_path = load_files()
        
        health_status = {
            "status": "healthy",
            "checks": {
                "model_file_exists": os.path.exists(model_path),
                "mapping_file_exists": os.path.exists(mapping_path),
                "model_file_readable": False,
                "mapping_file_readable": False,
                "model_file_size": 0,
                "mapping_file_size": 0,
                "pytorch_available": True,
                "model_load_test": "not_tested"
            },
            "pytorch_version": None,
            "system_info": {
                "python_version": sys.version,
                "platform": sys.platform
            }
        }
        
        # Check PyTorch
        try:
            import torch
            health_status["pytorch_version"] = torch.__version__
            health_status["checks"]["pytorch_available"] = True
        except ImportError as e:
            health_status["checks"]["pytorch_available"] = False
            health_status["errors"] = health_status.get("errors", [])
            health_status["errors"].append(f"PyTorch not available: {e}")
        
        # Check model file accessibility
        if health_status["checks"]["model_file_exists"]:
            try:
                health_status["checks"]["model_file_size"] = os.path.getsize(model_path)
                health_status["checks"]["model_file_readable"] = os.access(model_path, os.R_OK)
                
                # Test file format
                with open(model_path, 'rb') as f:
                    first_bytes = f.read(16)
                    health_status["checks"]["model_file_format"] = "pytorch" if first_bytes.startswith(b'PK') else "unknown"
                    
            except Exception as e:
                health_status["checks"]["model_file_readable"] = False
                health_status["errors"] = health_status.get("errors", [])
                health_status["errors"].append(f"Cannot read model file: {e}")
        
        # Check mapping file accessibility
        if health_status["checks"]["mapping_file_exists"]:
            try:
                health_status["checks"]["mapping_file_size"] = os.path.getsize(mapping_path)
                health_status["checks"]["mapping_file_readable"] = os.access(mapping_path, os.R_OK)
                
                # Test mapping file content
                import pandas as pd
                mapping_df = pd.read_csv(mapping_path)
                required_columns = ['class_id', 'scientific_name', 'common_name']
                health_status["checks"]["mapping_file_valid"] = all(col in mapping_df.columns for col in required_columns)
                health_status["checks"]["mapping_file_rows"] = len(mapping_df)
                
            except Exception as e:
                health_status["checks"]["mapping_file_readable"] = False
                health_status["errors"] = health_status.get("errors", [])
                health_status["errors"].append(f"Cannot read mapping file: {e}")
        
        # Test model loading (only if all prerequisites are met)
        if (health_status["checks"]["model_file_exists"] and 
            health_status["checks"]["model_file_readable"] and 
            health_status["checks"]["pytorch_available"]):
            try:
                logger.info("Testing model loading...")
                import torch
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                
                # Try to load just the checkpoint without creating model instance
                checkpoint = torch.load(model_path, map_location=device, weights_only=False)
                health_status["checks"]["model_load_test"] = "success"
                health_status["checks"]["model_type"] = str(type(checkpoint))
                
            except Exception as e:
                health_status["checks"]["model_load_test"] = "failed"
                health_status["status"] = "unhealthy"
                health_status["errors"] = health_status.get("errors", [])
                health_status["errors"].append(f"Model loading failed: {e}")
                
                # Specific diagnosis for "load key 'v'" error
                if "load key 'v'" in str(e):
                    health_status["errors"].append(
                        "This indicates a PyTorch model file corruption or version incompatibility. "
                        "The model file may need to be re-saved or re-deployed."
                    )
        
        # Determine overall health status
        critical_checks = [
            "model_file_exists", "mapping_file_exists", 
            "model_file_readable", "mapping_file_readable", 
            "pytorch_available"
        ]
        
        if not all(health_status["checks"].get(check, False) for check in critical_checks):
            health_status["status"] = "unhealthy"
        elif health_status["checks"]["model_load_test"] == "failed":
            health_status["status"] = "unhealthy"
        
        logger.info(f"Health check completed with status: {health_status['status']}")
        
        # Return appropriate HTTP status
        if health_status["status"] == "healthy":
            return jsonify(health_status), 200
        else:
            return jsonify(health_status), 503  # Service Unavailable
            
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}", exc_info=True)
        return jsonify({
            "status": "error",
            "message": f"Health check failed: {str(e)}"
        }), 500

if __name__ == "__main__":
    # Example usage
    audio_file = "../../database/audio/dev/XC114492.ogg"
    try:
        result = process_audio(audio_file)
        print("Inference Result:", result)
    except Exception as e:
        print(f"Error during inference: {e}")
