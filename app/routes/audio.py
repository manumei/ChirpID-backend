from flask import Blueprint, request, jsonify
import os
from werkzeug.utils import secure_filename
from app.services.audio_processing import process_audio

audio_bp = Blueprint("audio", __name__)

@audio_bp.route("/upload", methods=["POST"])
def upload_audio():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    save_path = os.path.join("uploads", filename)
    file.save(save_path)

    result = process_audio(save_path)
    return jsonify({"result": result})
