#!/bin/bash
"""
Startup script for ChirpID Backend
Supports both development and production modes
"""

import os
import sys

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app

def main():
    app = create_app()
    
    # Get environment variables
    flask_env = os.getenv('FLASK_ENV', 'development')
    debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    port = int(os.getenv('PORT', 5001))
    host = os.getenv('HOST', '0.0.0.0')
    
    print(f"Starting ChirpID Backend Server...")
    print(f"Environment: {flask_env}")
    print(f"Debug mode: {debug_mode}")
    print(f"Host: {host}")
    print(f"Port: {port}")
    
    if flask_env == 'production':
        print("Production mode: Use Gunicorn for better performance")
        print("Run: gunicorn --config gunicorn.conf.py server.wsgi:application")
    
    # Run Flask development server
    app.run(host=host, port=port, debug=debug_mode)

if __name__ == "__main__":
    main()
