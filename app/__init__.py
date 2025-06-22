from flask import Flask, jsonify
from app.routes.audio import audio_bp

def create_app():
    app = Flask(__name__)
    
    # CORS is handled by nginx reverse proxy
    # No need to configure CORS here to avoid duplicate headers
    
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
