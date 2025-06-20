# API Testing Guide - ChirpID Backend

## 🎯 How to Confirm API is Working After Deployment

After deploying your ChirpID Backend, you can verify that the API is working correctly using several methods:

## 📍 Available Endpoints

Based on your Flask application, here are the available endpoints:

### Core Health Endpoints

- `GET /health` - Health check endpoint
- `GET /ping` - Ping endpoint for connectivity testing

### Audio Processing API

- `POST /api/audio/upload` - Upload audio file for bird identification
- `GET /api/audio/files` - List uploaded audio files
- `POST /api/audio/cleanup` - Clean up uploaded files

## 🔍 Manual Testing Methods

### 1. **Browser Testing (Simple)**

Open your browser and navigate to:

```
http://your-server-ip:5000/health
http://your-server-ip:5000/ping
```

Expected responses:

```json
// /health
{"status": "healthy", "service": "ChirpID Backend"}

// /ping
{"status": "ok", "message": "ChirpID backend is running"}
```

### 2. **cURL Testing (Command Line)**

#### Basic Health Checks:

```bash
# Health check
curl -X GET http://your-server-ip:5000/health

# Ping test
curl -X GET http://your-server-ip:5000/ping

# Test with verbose output
curl -v http://your-server-ip:5000/health
```

#### Audio API Testing:

```bash
# List files endpoint
curl -X GET http://your-server-ip:5000/api/audio/files

# Upload audio file (replace with actual audio file)
curl -X POST \
  -F "file=@path/to/your/audio.wav" \
  http://your-server-ip:5000/api/audio/upload

# Cleanup files
curl -X POST http://your-server-ip:5000/api/audio/cleanup
```

### 3. **Python Script Testing**

Save this as `test_api.py`:

```python
#!/usr/bin/env python3
"""
Comprehensive API testing script for ChirpID Backend
"""
import requests
import json
import sys
import os
from pathlib import Path

def test_api(base_url="http://localhost:5000"):
    """Test all API endpoints"""
    results = []

    print(f"🧪 Testing ChirpID Backend API at: {base_url}")
    print("=" * 50)

    # Test 1: Health Check
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health Check: {data}")
            results.append(("Health", True, data))
        else:
            print(f"❌ Health Check Failed: {response.status_code}")
            results.append(("Health", False, response.status_code))
    except Exception as e:
        print(f"❌ Health Check Error: {e}")
        results.append(("Health", False, str(e)))

    # Test 2: Ping
    try:
        response = requests.get(f"{base_url}/ping", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Ping: {data}")
            results.append(("Ping", True, data))
        else:
            print(f"❌ Ping Failed: {response.status_code}")
            results.append(("Ping", False, response.status_code))
    except Exception as e:
        print(f"❌ Ping Error: {e}")
        results.append(("Ping", False, str(e)))

    # Test 3: Audio Files List
    try:
        response = requests.get(f"{base_url}/api/audio/files", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Audio Files: {len(data.get('files', []))} files found")
            results.append(("AudioFiles", True, data))
        else:
            print(f"❌ Audio Files Failed: {response.status_code}")
            results.append(("AudioFiles", False, response.status_code))
    except Exception as e:
        print(f"❌ Audio Files Error: {e}")
        results.append(("AudioFiles", False, str(e)))

    # Test 4: Audio Upload (if test file exists)
    test_file_path = "tests/audioPrueba.ogg"
    if os.path.exists(test_file_path):
        try:
            with open(test_file_path, 'rb') as f:
                files = {'file': f}
                response = requests.post(f"{base_url}/api/audio/upload", files=files, timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    print(f"✅ Audio Upload: {data}")
                    results.append(("AudioUpload", True, data))
                else:
                    print(f"❌ Audio Upload Failed: {response.status_code}")
                    results.append(("AudioUpload", False, response.status_code))
        except Exception as e:
            print(f"❌ Audio Upload Error: {e}")
            results.append(("AudioUpload", False, str(e)))
    else:
        print(f"⚠️  Audio Upload: Test file not found at {test_file_path}")
        results.append(("AudioUpload", False, "Test file not found"))

    # Summary
    print("\n" + "=" * 50)
    print("🏁 Test Summary:")
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    print(f"✅ Passed: {passed}/{total}")

    for test_name, success, result in results:
        status = "✅" if success else "❌"
        print(f"{status} {test_name}: {result}")

    return passed == total

if __name__ == "__main__":
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:5000"
    success = test_api(base_url)
    sys.exit(0 if success else 1)
```

Run it with:

```bash
python test_api.py http://your-server-ip:5000
```

### 4. **Postman Collection**

Create a Postman collection with these requests:

1. **Health Check**

   - Method: GET
   - URL: `{{base_url}}/health`

2. **Ping Test**

   - Method: GET
   - URL: `{{base_url}}/ping`

3. **List Audio Files**

   - Method: GET
   - URL: `{{base_url}}/api/audio/files`

4. **Upload Audio**
   - Method: POST
   - URL: `{{base_url}}/api/audio/upload`
   - Body: form-data with `file` field

## 🤖 Automated Post-Deployment Testing

To automatically verify the API after deployment, I'll add a comprehensive verification step to your workflow:

### Add to GitHub Actions Workflow

Here's a post-deployment verification script that should be added to your workflow:

```yaml
- name: Comprehensive API Testing
  uses: appleboy/ssh-action@v1.0.3
  with:
    host: ${{ secrets.SSH_HOST }}
    username: ${{ secrets.SSH_USERNAME }}
    key: ${{ secrets.SSH_PRIVATE_KEY }}
    port: ${{ secrets.SSH_PORT || 22 }}
    script: |
      echo "🧪 Running comprehensive API tests..."

      # Test all endpoints
      BASE_URL="http://localhost:5001"

      # Health check
      echo "Testing /health endpoint..."
      HEALTH_RESPONSE=$(curl -s -w "%{http_code}" -o /tmp/health.json "$BASE_URL/health")
      if [ "$HEALTH_RESPONSE" = "200" ]; then
        echo "✅ Health check passed"
        cat /tmp/health.json
      else
        echo "❌ Health check failed with code: $HEALTH_RESPONSE"
        exit 1
      fi

      # Ping test
      echo "Testing /ping endpoint..."
      PING_RESPONSE=$(curl -s -w "%{http_code}" -o /tmp/ping.json "$BASE_URL/ping")
      if [ "$PING_RESPONSE" = "200" ]; then
        echo "✅ Ping test passed"
        cat /tmp/ping.json
      else
        echo "❌ Ping test failed with code: $PING_RESPONSE"
        exit 1
      fi

      # Audio files endpoint
      echo "Testing /api/audio/files endpoint..."
      FILES_RESPONSE=$(curl -s -w "%{http_code}" -o /tmp/files.json "$BASE_URL/api/audio/files")
      if [ "$FILES_RESPONSE" = "200" ]; then
        echo "✅ Audio files endpoint passed"
        cat /tmp/files.json
      else
        echo "❌ Audio files endpoint failed with code: $FILES_RESPONSE"
        exit 1
      fi

      # CORS test
      echo "Testing CORS headers..."
      CORS_HEADERS=$(curl -s -H "Origin: http://localhost:3000" -H "Access-Control-Request-Method: POST" -H "Access-Control-Request-Headers: Content-Type" -X OPTIONS "$BASE_URL/api/audio/upload" -I)
      if echo "$CORS_HEADERS" | grep -q "Access-Control-Allow-Origin"; then
        echo "✅ CORS headers present"
      else
        echo "❌ CORS headers missing"
        echo "$CORS_HEADERS"
        exit 1
      fi

      echo "🎉 All API tests passed!"
```

## 📊 Monitoring and Alerting

### 1. **Set Up Uptime Monitoring**

Use services like:

- **UptimeRobot** (free tier available)
- **Pingdom**
- **StatusCake**

Configure them to monitor:

- `http://your-server:5000/health`
- `http://your-server:5000/ping`

### 2. **Log Monitoring**

Check application logs:

```bash
# View container logs
docker logs chirpid-backend

# Follow logs in real-time
docker logs -f chirpid-backend

# View last 50 lines
docker logs --tail 50 chirpid-backend
```

### 3. **Performance Testing**

Use tools like:

```bash
# Apache Bench
ab -n 100 -c 10 http://your-server:5000/health

# Hey (Go-based tool)
hey -n 100 -c 10 http://your-server:5000/health
```

## 🚨 Troubleshooting Common Issues

### API Not Responding

1. Check if container is running: `docker ps`
2. Check container logs: `docker logs chirpid-backend`
3. Check port mapping: `docker port chirpid-backend`
4. Test internal connectivity: `docker exec chirpid-backend curl http://localhost:5001/health`

### CORS Issues

1. Verify CORS headers in response
2. Check allowed origins in Flask-CORS configuration
3. Test with browser developer tools

### File Upload Issues

1. Check upload directory permissions
2. Verify file size limits
3. Check supported file formats
4. Monitor disk space

## 🎯 Success Criteria

Your API deployment is successful when:

✅ **Health endpoint** returns 200 with correct JSON  
✅ **Ping endpoint** returns 200 with correct JSON  
✅ **Audio endpoints** return appropriate responses  
✅ **CORS headers** are present for cross-origin requests  
✅ **Container health check** shows as healthy  
✅ **Logs show** no critical errors  
✅ **Response times** are under acceptable limits (< 2s for health checks)

## 📞 Quick Verification Commands

```bash
# One-liner health check
curl -f http://your-server:5000/health && echo "✅ API is healthy"

# Full endpoint test
for endpoint in health ping api/audio/files; do
  echo "Testing /$endpoint..."
  curl -s "http://your-server:5000/$endpoint" | jq . || echo "Failed"
done
```

Save this guide and use it after every deployment to ensure your API is working correctly!
