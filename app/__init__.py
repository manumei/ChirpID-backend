from flask import Flask, jsonify
from app.routes.audio import audio_bp
import os

def create_app():
    app = Flask(__name__)
    
    # CORS is handled by nginx - no need for Flask-CORS to prevent duplicate headers
    
    # Add ping route for connection testing
    @app.route("/ping", methods=["GET"])
    def ping():
        return jsonify({"status": "ok", "message": "ChirpID backend is running"})
    
    # Health check route
    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "healthy", "service": "ChirpID Backend"})
    
    app.register_blueprint(audio_bp, url_prefix="/api/audio")
    return app
