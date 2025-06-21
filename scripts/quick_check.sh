#!/bin/bash
"""
Quick one-liner API health check for ChirpID Backend
Usage: ./scripts/quick_check.sh [server_url]
"""

# Default to localhost if no URL provided
SERVER_URL=${1:-"http://localhost:5000"}

echo "🔍 Quick health check for: $SERVER_URL"
echo "========================================="

# Remove trailing slash
SERVER_URL=${SERVER_URL%/}

# Test health endpoint
echo -n "Health Check: "
if curl -s -f --connect-timeout 5 --max-time 10 "$SERVER_URL/health" > /dev/null; then
    HEALTH_DATA=$(curl -s "$SERVER_URL/health")
    echo "✅ OK - $HEALTH_DATA"
else
    echo "❌ FAILED"
    exit 1
fi

# Test ping endpoint  
echo -n "Ping Test: "
if curl -s -f --connect-timeout 5 --max-time 10 "$SERVER_URL/ping" > /dev/null; then
    PING_DATA=$(curl -s "$SERVER_URL/ping")
    echo "✅ OK - $PING_DATA"
else
    echo "❌ FAILED"
    exit 1
fi

echo ""
echo "🎉 API is working! Ready to use at: $SERVER_URL"
