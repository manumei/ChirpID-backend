"""WSGI entry point for ChirpID Backend"""
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app

# Create the application instance
application = create_app()

if __name__ == "__main__":
    application.run()
