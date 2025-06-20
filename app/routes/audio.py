from flask import Blueprint, request, jsonify
import os
import uuid
import logging
from datetime import datetime
from werkzeug.utils import secure_filename
from app.services.audio_processing import process_audio

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
        result = process_audio(save_path)
        
        logger.info("Audio processing completed successfully")
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
        logger.error(f"Error processing upload: {str(e)}", exc_info=True)
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

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
        print(f"Error listing files: {str(e)}")
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
                    print(f"Deleted old file: {filename}")
                except Exception as e:
                    print(f"Failed to delete {filename}: {e}")
        
        return jsonify({
            "success": True,
            "message": f"Cleaned up files older than {days} days",
            "deleted_count": len(deleted_files),
            "deleted_files": deleted_files
        })
        
    except Exception as e:
        print(f"Error during cleanup: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500
