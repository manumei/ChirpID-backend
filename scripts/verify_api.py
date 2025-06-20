#!/usr/bin/env python3
"""
Quick API verification script for ChirpID Backend
Usage: python scripts/verify_api.py [base_url]
"""
import requests
import json
import sys
import time
from datetime import datetime

def test_endpoint(name, url, method="GET", timeout=10):
    """Test a single endpoint and return success status"""
    try:
        start_time = time.time()
        
        if method == "GET":
            response = requests.get(url, timeout=timeout)
        elif method == "POST":
            response = requests.post(url, timeout=timeout)
        else:
            response = requests.request(method, url, timeout=timeout)
        
        end_time = time.time()
        duration = round((end_time - start_time) * 1000, 2)  # Convert to ms
        
        if response.status_code == 200:
            try:
                data = response.json()
                print(f"✅ {name}: OK ({duration}ms) - {data}")
                return True, data
            except json.JSONDecodeError:
                print(f"✅ {name}: OK ({duration}ms) - Non-JSON response")
                return True, response.text
        else:
            print(f"❌ {name}: FAILED (HTTP {response.status_code}) - {response.text}")
            return False, None
            
    except requests.exceptions.Timeout:
        print(f"❌ {name}: TIMEOUT (>{timeout}s)")
        return False, None
    except requests.exceptions.ConnectionError:
        print(f"❌ {name}: CONNECTION ERROR")
        return False, None
    except Exception as e:
        print(f"❌ {name}: ERROR - {str(e)}")
        return False, None

def verify_api(base_url="http://localhost:5000"):
    """Verify all API endpoints"""
    print(f"🔍 Verifying ChirpID Backend API at: {base_url}")
    print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    tests = [
        ("Health Check", f"{base_url}/health"),
        ("Ping Test", f"{base_url}/ping"),
        ("Audio Files List", f"{base_url}/api/audio/files"),
    ]
    
    results = []
    for name, url in tests:
        success, data = test_endpoint(name, url)
        results.append((name, success, data))
        time.sleep(0.5)  # Small delay between tests
    
    # Test CORS (if possible)
    print("\n🌐 Testing CORS configuration...")
    try:
        headers = {
            'Origin': 'http://localhost:3000',
            'Access-Control-Request-Method': 'POST',
            'Access-Control-Request-Headers': 'Content-Type'
        }
        response = requests.options(f"{base_url}/api/audio/upload", headers=headers, timeout=10)
        cors_headers = response.headers
        
        if 'Access-Control-Allow-Origin' in cors_headers:
            print(f"✅ CORS: Configured (Allow-Origin: {cors_headers.get('Access-Control-Allow-Origin')})")
        else:
            print("❌ CORS: Missing Access-Control-Allow-Origin header")
            
    except Exception as e:
        print(f"❌ CORS: Could not test - {str(e)}")
    
    # Summary
    print("\n" + "=" * 60)
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    if passed == total:
        print(f"🎉 ALL TESTS PASSED! ({passed}/{total})")
        print(f"✅ Your API is working correctly at {base_url}")
        print(f"🌐 Ready for frontend integration!")
    else:
        print(f"⚠️  SOME TESTS FAILED ({passed}/{total})")
        print("❌ Please check the failed endpoints above")
    
    print(f"\n📊 Quick Stats:")
    print(f"   🔗 Base URL: {base_url}")
    print(f"   ✅ Working endpoints: {passed}")
    print(f"   ❌ Failed endpoints: {total - passed}")
    print(f"   🕐 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return passed == total

if __name__ == "__main__":
    # Get base URL from command line or use default
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:5000"
    
    # Remove trailing slash
    base_url = base_url.rstrip('/')
    
    # Run verification
    success = verify_api(base_url)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)
