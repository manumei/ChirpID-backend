#!/usr/bin/env python3
"""
Test script to verify ChirpID Backend can start properly
"""
import sys
import os
import traceback

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_app_creation():
    """Test if the Flask app can be created without errors"""
    try:
        print("Testing Flask app creation...")
        from app import create_app
        
        app = create_app()
        print("✅ Flask app created successfully")
        
        # Test if health endpoint is registered
        with app.test_client() as client:
            response = client.get('/health')
            print(f"✅ Health endpoint test: {response.status_code} - {response.get_json()}")
            
            response = client.get('/ping')
            print(f"✅ Ping endpoint test: {response.status_code} - {response.get_json()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating Flask app: {str(e)}")
        print("Traceback:")
        traceback.print_exc()
        return False

def test_imports():
    """Test if all required modules can be imported"""
    try:
        print("Testing imports...")
        
        import importlib.metadata
        
        # Test Flask imports
        import flask
        try:
            flask_version = importlib.metadata.version("flask")
        except Exception:
            flask_version = getattr(flask, '__version__', 'unknown')
        print(f"✅ Flask: {flask_version}")
        
        import flask_cors
        try:
            cors_version = importlib.metadata.version("flask-cors")
        except Exception:
            cors_version = getattr(flask_cors, '__version__', 'unknown')
        print(f"✅ Flask-CORS: {cors_version}")
        
        # Test audio processing imports
        import librosa
        try:
            librosa_version = importlib.metadata.version("librosa")
        except Exception:
            librosa_version = getattr(librosa, '__version__', 'unknown')
        print(f"✅ Librosa: {librosa_version}")
        
        import soundfile
        try:
            soundfile_version = importlib.metadata.version("soundfile")
        except Exception:
            soundfile_version = getattr(soundfile, '__version__', 'unknown')
        print(f"✅ SoundFile: {soundfile_version}")
        
        import numpy
        try:
            numpy_version = importlib.metadata.version("numpy")
        except Exception:
            numpy_version = getattr(numpy, '__version__', 'unknown')
        print(f"✅ NumPy: {numpy_version}")
        
        return True
        
    except Exception as e:
        print(f"❌ Import error: {str(e)}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("ChirpID Backend Startup Test")
    print("=" * 50)
    
    success = True
    
    # Test imports first
    if not test_imports():
        success = False
    
    print()
    
    # Test app creation
    if not test_app_creation():
        success = False
    
    print()
    print("=" * 50)
    if success:
        print("✅ All tests passed! The app should start correctly.")
    else:
        print("❌ Some tests failed. Check the errors above.")
    print("=" * 50)
    
    sys.exit(0 if success else 1)
