#!/bin/bash

# Model Transfer and Verification Script
# This script helps transfer the PyTorch model file to Ubuntu server safely

set -e  # Exit on any error

echo "ChirpID Model Transfer Script"
echo "============================"

# Configuration - modify these paths as needed
LOCAL_MODEL_PATH="models/bird_cnn.pth"
SERVER_USER="your_username"
SERVER_IP="your_server_ip"
SERVER_PATH="/path/to/chirpid-backend/models/bird_cnn.pth"

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  -l LOCAL_PATH    Local path to model file (default: models/bird_cnn.pth)"
    echo "  -u USER          Server username"
    echo "  -s SERVER_IP     Server IP address"
    echo "  -r REMOTE_PATH   Remote path on server"
    echo "  -h               Show this help"
    echo ""
    echo "Example:"
    echo "  $0 -u ubuntu -s 192.168.1.100 -r /home/ubuntu/chirpid-backend/models/bird_cnn.pth"
}

# Parse command line arguments
while getopts "l:u:s:r:h" opt; do
    case $opt in
        l) LOCAL_MODEL_PATH="$OPTARG" ;;
        u) SERVER_USER="$OPTARG" ;;
        s) SERVER_IP="$OPTARG" ;;
        r) SERVER_PATH="$OPTARG" ;;
        h) usage; exit 0 ;;
        \?) echo "Invalid option -$OPTARG" >&2; usage; exit 1 ;;
    esac
done

# Validate required parameters
if [[ -z "$SERVER_USER" || -z "$SERVER_IP" || -z "$SERVER_PATH" ]]; then
    echo "Error: Missing required parameters"
    usage
    exit 1
fi

echo "Configuration:"
echo "  Local model: $LOCAL_MODEL_PATH"
echo "  Server: $SERVER_USER@$SERVER_IP"
echo "  Remote path: $SERVER_PATH"
echo ""

# Check if local model exists
if [[ ! -f "$LOCAL_MODEL_PATH" ]]; then
    echo "❌ Local model file not found: $LOCAL_MODEL_PATH"
    exit 1
fi

# Get local file info
LOCAL_SIZE=$(stat -f%z "$LOCAL_MODEL_PATH" 2>/dev/null || stat -c%s "$LOCAL_MODEL_PATH" 2>/dev/null)
LOCAL_MD5=$(md5sum "$LOCAL_MODEL_PATH" | cut -d' ' -f1 2>/dev/null || md5 -q "$LOCAL_MODEL_PATH" 2>/dev/null)

echo "Local file info:"
echo "  Size: $LOCAL_SIZE bytes ($(echo "scale=2; $LOCAL_SIZE/1048576" | bc) MB)"
echo "  MD5: $LOCAL_MD5"
echo ""

# Create remote directory if needed
echo "Creating remote directory..."
ssh "$SERVER_USER@$SERVER_IP" "mkdir -p $(dirname $SERVER_PATH)"

# Transfer file with verification
echo "Transferring file..."
scp -C "$LOCAL_MODEL_PATH" "$SERVER_USER@$SERVER_IP:$SERVER_PATH"

# Verify transfer
echo "Verifying transfer..."
REMOTE_SIZE=$(ssh "$SERVER_USER@$SERVER_IP" "stat -c%s $SERVER_PATH 2>/dev/null || echo 0")
REMOTE_MD5=$(ssh "$SERVER_USER@$SERVER_IP" "md5sum $SERVER_PATH | cut -d' ' -f1 2>/dev/null || echo 'unknown'")

echo "Remote file info:"
echo "  Size: $REMOTE_SIZE bytes"
echo "  MD5: $REMOTE_MD5"
echo ""

# Compare checksums
if [[ "$LOCAL_MD5" == "$REMOTE_MD5" ]]; then
    echo "✅ File transfer verified successfully!"
else
    echo "❌ File transfer verification failed!"
    echo "   Local MD5:  $LOCAL_MD5"
    echo "   Remote MD5: $REMOTE_MD5"
    exit 1
fi

# Test model loading on server
echo "Testing model loading on server..."
ssh "$SERVER_USER@$SERVER_IP" "cd $(dirname $SERVER_PATH)/.. && python scripts/validate_model.py"

if [[ $? -eq 0 ]]; then
    echo "✅ Model validation successful on server!"
else
    echo "❌ Model validation failed on server!"
    echo "Check the server logs and PyTorch version compatibility."
    exit 1
fi

echo ""
echo "🎉 Model transfer and validation completed successfully!"
echo "You can now restart your ChirpID backend service."
