#!/bin/bash

# Backend Log Checker Script
# This script helps you quickly check backend logs and health

echo "ChirpID Backend Log Checker"
echo "==========================="

# Check if server is running
echo "1. Checking if backend server is running..."
SERVER_URL="http://localhost:5001"
if curl -s "$SERVER_URL/ping" > /dev/null 2>&1; then
    echo "✅ Backend server is running"
else
    echo "❌ Backend server is not responding"
    echo "   Make sure your backend is running on port 5001"
    exit 1
fi

echo ""
echo "2. Checking backend health..."
echo "----------------------------"
curl -s "$SERVER_URL/api/audio/health" | jq . || curl -s "$SERVER_URL/api/audio/health"

echo ""
echo "3. To see live backend logs, run one of these commands:"
echo "   For direct Python: Check the terminal where you ran 'python app.py'"
echo "   For systemd service: sudo journalctl -u your-service-name -f"
echo "   For gunicorn: Check the terminal where you ran gunicorn"
echo "   For Docker: docker logs container-name -f"

echo ""
echo "4. To test with a sample file:"
echo "   curl -X POST -F \"file=@sample.wav\" $SERVER_URL/api/audio/upload"
