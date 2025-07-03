#!/bin/bash

# Complete ChirpID Fix Script
# This script fixes both the model file and code issues

echo "🔧 ChirpID Complete Fix Script"
echo "============================="

CONTAINER_NAME="chirpid-backend"
LOCAL_MODEL="models/bird_cnn.pth"

echo "1️⃣ Checking current status..."

# Check if local model exists and its size
if [[ -f "$LOCAL_MODEL" ]]; then
    LOCAL_SIZE=$(stat -c%s "$LOCAL_MODEL" 2>/dev/null || stat -f%z "$LOCAL_MODEL" 2>/dev/null)
    echo "📁 Local model: $LOCAL_SIZE bytes"
    
    if [[ $LOCAL_SIZE -lt 1000000 ]]; then
        echo "⚠️  Local model file appears to be a Git LFS pointer (too small)"
        echo "   Try running: git lfs pull"
        echo "   Or copy the real model file manually"
    fi
else
    echo "❌ Local model file not found: $LOCAL_MODEL"
    exit 1
fi

echo ""
echo "2️⃣ Stopping and rebuilding container..."

# Stop container
docker-compose down

# Rebuild with no cache to ensure latest code
docker-compose build --no-cache

# Start container
docker-compose up -d

# Wait for container to be ready
echo "⏳ Waiting for container to start..."
sleep 10

echo ""
echo "3️⃣ Copying model file..."

# Copy model file
docker cp "$LOCAL_MODEL" "$CONTAINER_NAME:/app/models/bird_cnn.pth"

if [[ $? -eq 0 ]]; then
    echo "✅ Model file copied"
    
    # Verify size in container
    CONTAINER_SIZE=$(docker exec "$CONTAINER_NAME" stat -c%s "/app/models/bird_cnn.pth" 2>/dev/null)
    echo "📁 Container model: $CONTAINER_SIZE bytes"
else
    echo "❌ Failed to copy model file"
    exit 1
fi

echo ""
echo "4️⃣ Testing the fix..."

# Test health endpoint
echo "🧪 Testing health endpoint..."
sleep 5  # Give the server a moment to start

# Use Python health check script
if python3 scripts/health_check.py "http://localhost:5001"; then
    echo "✅ Health check passed!"
else
    echo "❌ Health check failed!"
    
    # Fallback to curl health check
    echo "🔄 Trying fallback health check..."
    for i in {1..5}; do
        if curl -s http://localhost:5001/api/audio/health > /dev/null; then
            echo "✅ Health endpoint is responding (fallback check)"
            break
        else
            echo "⏳ Attempt $i/5: Health endpoint not ready, waiting..."
            sleep 3
        fi
    done
fi

# Show detailed health info
echo ""
echo "📊 Detailed health check:"
curl -s http://localhost:5001/api/audio/health | python3 -m json.tool 2>/dev/null || curl -s http://localhost:5001/api/audio/health

echo ""
echo "🎉 Fix completed!"
echo ""
echo "📋 Next steps:"
echo "  • Monitor logs: docker logs -f $CONTAINER_NAME"
echo "  • Test upload: Use your frontend or curl with a test file"
echo "  • If issues persist, check the logs for specific errors"
