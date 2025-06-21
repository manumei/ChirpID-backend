#!/bin/bash

# ChirpID Health Check Script
# Tests the deployed API endpoints

echo "🏥 ChirpID Health Check - chirpid.utictactoe.online"
echo "================================================"

# Test HTTP redirect
echo "🔄 Testing HTTP to HTTPS redirect..."
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -L http://chirpid.utictactoe.online/health)
if [ "$HTTP_CODE" = "200" ]; then
    echo "✅ HTTP redirect working"
else
    echo "❌ HTTP redirect failed (Code: $HTTP_CODE)"
fi

# Test HTTPS health endpoint
echo "🔒 Testing HTTPS health endpoint..."
HTTPS_CODE=$(curl -s -o /dev/null -w "%{http_code}" https://chirpid.utictactoe.online/health)
if [ "$HTTPS_CODE" = "200" ]; then
    echo "✅ HTTPS health endpoint working"
else
    echo "❌ HTTPS health endpoint failed (Code: $HTTPS_CODE)"
fi

# Test ping endpoint
echo "🏓 Testing ping endpoint..."
PING_CODE=$(curl -s -o /dev/null -w "%{http_code}" https://chirpid.utictactoe.online/ping)
if [ "$PING_CODE" = "200" ]; then
    echo "✅ Ping endpoint working"
else
    echo "❌ Ping endpoint failed (Code: $PING_CODE)"
fi

# Test API endpoint structure
echo "🔌 Testing API endpoint structure..."
API_CODE=$(curl -s -o /dev/null -w "%{http_code}" https://chirpid.utictactoe.online/api/)
if [ "$API_CODE" = "404" ] || [ "$API_CODE" = "405" ] || [ "$API_CODE" = "200" ]; then
    echo "✅ API endpoint accessible (responds with $API_CODE)"
else
    echo "❌ API endpoint failed (Code: $API_CODE)"
fi

# Check SSL certificate
echo "🔐 Checking SSL certificate..."
SSL_INFO=$(echo | openssl s_client -servername chirpid.utictactoe.online -connect chirpid.utictactoe.online:443 2>/dev/null | openssl x509 -noout -dates 2>/dev/null)
if [ $? -eq 0 ]; then
    echo "✅ SSL certificate valid"
    echo "   $SSL_INFO"
else
    echo "❌ SSL certificate check failed"
fi

# Check Docker services
echo "🐳 Checking Docker services..."
if docker-compose ps | grep -q "Up"; then
    echo "✅ Docker services running:"
    docker-compose ps --format "table {{.Service}}\t{{.State}}\t{{.Ports}}"
else
    echo "❌ Docker services not running properly"
    docker-compose ps
fi

echo ""
echo "🎯 Health check completed!"
