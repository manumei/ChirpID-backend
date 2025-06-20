#!/usr/bin/env python3
"""
Health check script for ChirpID Backend
"""
import requests
import sys
import time
import os

def check_health(base_url="http://localhost:5001", max_attempts=30, delay=2):
    """Check if the health endpoint is responding"""
    health_url = f"{base_url}/health"
    
    print(f"Checking health endpoint: {health_url}")
    print(f"Max attempts: {max_attempts}, Delay: {delay}s")
    
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(health_url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Health check passed on attempt {attempt}")
                print(f"Response: {data}")
                return True
            else:
                print(f"❌ Attempt {attempt}/{max_attempts}: HTTP {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"❌ Attempt {attempt}/{max_attempts}: {str(e)}")
        
        if attempt < max_attempts:
            print(f"Waiting {delay}s before next attempt...")
            time.sleep(delay)
    
    print(f"❌ Health check failed after {max_attempts} attempts")
    return False

if __name__ == "__main__":
    # Get configuration from environment or command line
    base_url = os.getenv('HEALTH_CHECK_URL', 'http://localhost:5001')
    max_attempts = int(os.getenv('HEALTH_CHECK_ATTEMPTS', '30'))
    delay = int(os.getenv('HEALTH_CHECK_DELAY', '2'))
    
    if len(sys.argv) > 1:
        base_url = sys.argv[1]
    
    success = check_health(base_url, max_attempts, delay)
    sys.exit(0 if success else 1)
