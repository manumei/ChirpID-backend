from flask import Flask, jsonify
from flask_cors import CORS
from app.routes.audio import audio_bp
import os

def create_app():
    app = Flask(__name__)
    
    # Enable CORS for local development
    if os.getenv('FLASK_ENV') == 'development' or os.getenv('FLASK_DEBUG') == '1':
        CORS(app, origins=['http://localhost:4321', 'http://127.0.0.1:4321'])
    else:
        CORS(app, origins=['http://localhost:4321'])
    
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
