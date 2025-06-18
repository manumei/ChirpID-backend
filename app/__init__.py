from flask import Flask, jsonify
from flask_cors import CORS
from app.routes.audio import audio_bp

def create_app():
    app = Flask(__name__)
    
    # Configure CORS for React Native app
    CORS(app, 
        origins=["*"],  # Allow all origins for development
        methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization"])
    
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
