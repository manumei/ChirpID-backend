#!/usr/bin/env python3
"""
Simple test script to verify the Flask backend is working correctly.
Run this after starting the server to test endpoints.
"""
import requests
import os

def test_ping():
    """Test the ping endpoint"""
    try:
        response = requests.get("http://localhost:5001/ping")
        print(f"Ping test: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Ping test failed: {e}")
        return False

def test_health():
    """Test the health endpoint"""
    try:
        response = requests.get("http://localhost:5001/health")
        print(f"Health test: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health test failed: {e}")
        return False

def test_upload():
    """Test the upload endpoint with a dummy file"""
    try:
        # Create a simple test wav file
        test_file_path = "test_audio.wav"
        with open(test_file_path, 'wb') as f:
            # Write a minimal WAV file header (44 bytes)
            f.write(b'RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00')
        
        with open(test_file_path, 'rb') as f:
            files = {'file': ('test_audio.wav', f, 'audio/wav')}
            response = requests.post("http://localhost:5001/api/audio/upload", files=files)
        
        print(f"Upload test: {response.status_code}")
        print(f"Response: {response.json()}")
        
        # Clean up
        if os.path.exists(test_file_path):
            os.remove(test_file_path)
            
        return response.status_code == 200
    except Exception as e:
        print(f"Upload test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing ChirpID Backend Server...")
    print("=" * 40)
    
    ping_ok = test_ping()
    print()
    
    health_ok = test_health()
    print()
    
    upload_ok = test_upload()
    print()
    
    print("=" * 40)
    print("Test Results:")
    print(f"Ping: {'✓' if ping_ok else '✗'}")
    print(f"Health: {'✓' if health_ok else '✗'}")
    print(f"Upload: {'✓' if upload_ok else '✗'}")
    
    if all([ping_ok, health_ok, upload_ok]):
        print("\nAll tests passed! 🎉")
    else:
        print("\nSome tests failed. Check the server logs.")
