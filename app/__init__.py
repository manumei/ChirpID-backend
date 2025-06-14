from flask import Flask
from flask_cors import CORS
from app.routes.audio import audio_bp

def create_app():
    app = Flask(__name__)
    CORS(app)
    app.register_blueprint(audio_bp, url_prefix="/api/audio")
    return app
