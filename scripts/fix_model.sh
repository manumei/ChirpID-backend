#!/bin/bash

# Fix Model File Script
# This script helps fix the Git LFS pointer issue by copying the real model file

echo "🔧 ChirpID Model File Fixer"
echo "=========================="

# Configuration
LOCAL_MODEL_PATH="models/bird_cnn.pth"
SERVER_MODEL_PATH="/app/models/bird_cnn.pth"
CONTAINER_NAME="chirpid-backend"

# Check if running from correct directory
if [[ ! -f "$LOCAL_MODEL_PATH" ]]; then
    echo "❌ Model file not found at: $LOCAL_MODEL_PATH"
    echo "Please run this script from the chirpid-backend directory"
    exit 1
fi

# Check local model file
LOCAL_SIZE=$(stat -c%s "$LOCAL_MODEL_PATH" 2>/dev/null || stat -f%z "$LOCAL_MODEL_PATH" 2>/dev/null)
echo "📁 Local model file size: $LOCAL_SIZE bytes ($(echo "scale=2; $LOCAL_SIZE/1048576" | bc 2>/dev/null || echo "calc failed") MB)"

if [[ $LOCAL_SIZE -lt 1000000 ]]; then
    echo "⚠️  WARNING: Local model file seems too small. It might be a Git LFS pointer."
    echo "   If you're using Git LFS, run: git lfs pull"
    read -p "   Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if Docker container is running
if ! docker ps | grep -q "$CONTAINER_NAME"; then
    echo "❌ Docker container '$CONTAINER_NAME' is not running"
    echo "   Start it with: docker-compose up -d"
    exit 1
fi

echo "🚀 Copying model file to Docker container..."

# Copy file to container
docker cp "$LOCAL_MODEL_PATH" "$CONTAINER_NAME:$SERVER_MODEL_PATH"

if [[ $? -eq 0 ]]; then
    echo "✅ Model file copied successfully!"
    
    # Verify the copy
    echo "🔍 Verifying copied file..."
    CONTAINER_SIZE=$(docker exec "$CONTAINER_NAME" stat -c%s "$SERVER_MODEL_PATH" 2>/dev/null)
    echo "📁 Container model file size: $CONTAINER_SIZE bytes"
    
    if [[ "$LOCAL_SIZE" == "$CONTAINER_SIZE" ]]; then
        echo "✅ File sizes match - copy verified!"
        
        # Test model loading
        echo "🧪 Testing model loading..."
        docker exec "$CONTAINER_NAME" python -c "
import torch
try:
    checkpoint = torch.load('$SERVER_MODEL_PATH', map_location='cpu', weights_only=False)
    print('✅ Model loads successfully!')
    print(f'📊 Model type: {type(checkpoint)}')
except Exception as e:
    print(f'❌ Model loading failed: {e}')
"
    else
        echo "⚠️  WARNING: File sizes don't match!"
        echo "   Local: $LOCAL_SIZE bytes"
        echo "   Container: $CONTAINER_SIZE bytes"
    fi
else
    echo "❌ Failed to copy model file"
    exit 1
fi

echo ""
echo "🎉 Model file fix completed!"
echo "💡 Now test with: curl http://localhost:5001/api/audio/health"
