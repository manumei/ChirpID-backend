#!/bin/bash

# ChirpID Production Deployment Script
# Domain: chirpid.utictactoe.online

set -e

echo "🚀 Starting ChirpID production deployment..."

# Check if running as root for certificate operations
if [[ $EUID -eq 0 ]]; then
   echo "⚠️  This script should not be run as root. Use sudo for individual commands when needed."
   exit 1
fi

# 1. Create Docker volume for Let's Encrypt certificates
echo "📦 Creating Docker volume for SSL certificates..."
docker volume create letsencrypt 2>/dev/null || echo "Volume already exists"

# 2. Install Certbot if not present
echo "🔐 Checking Certbot installation..."
if ! command -v certbot &> /dev/null; then
    echo "Installing Certbot..."
    sudo apt update
    sudo apt install -y certbot
else
    echo "Certbot already installed"
fi

# 3. Stop any running services on ports 80/443
echo "🛑 Stopping existing services on ports 80/443..."
sudo fuser -k 80/tcp 2>/dev/null || true
sudo fuser -k 443/tcp 2>/dev/null || true
docker-compose down 2>/dev/null || true

# 4. Generate SSL certificates
echo "🔒 Generating SSL certificates for chirpid.utictactoe.online..."
sudo certbot certonly --standalone \
    -d chirpid.utictactoe.online \
    --email admin@utictactoe.online \
    --agree-tos \
    --non-interactive \
    --force-renewal

# 5. Copy certificates to Docker volume
echo "📋 Copying certificates to Docker volume..."
sudo docker run --rm \
    -v letsencrypt:/etc/letsencrypt \
    -v /etc/letsencrypt:/host-certs \
    alpine cp -r /host-certs/. /etc/letsencrypt/

# 6. Build and start services
echo "🏗️  Building and starting services..."
docker-compose build --no-cache
docker-compose up -d

# 7. Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 10

# 8. Check service health
echo "🏥 Checking service health..."
if docker-compose ps | grep -q "Up"; then
    echo "✅ Services are running"
    
    # Test HTTP redirect
    echo "🧪 Testing HTTP to HTTPS redirect..."
    curl -s -o /dev/null -w "%{http_code}" http://chirpid.utictactoe.online/health || echo "HTTP test failed"
    
    # Test HTTPS endpoint
    echo "🧪 Testing HTTPS endpoint..."
    curl -s -o /dev/null -w "%{http_code}" https://chirpid.utictactoe.online/health || echo "HTTPS test failed"
    
    echo "🎉 Deployment completed successfully!"
    echo "🌐 Your API is available at: https://chirpid.utictactoe.online"
    echo "🔍 Health check: https://chirpid.utictactoe.online/health"
    echo "📡 API endpoints: https://chirpid.utictactoe.online/api/"
else
    echo "❌ Services failed to start. Check logs:"
    docker-compose logs
    exit 1
fi

# 9. Setup automatic certificate renewal
echo "⚙️  Setting up automatic certificate renewal..."
CRON_JOB="0 12 * * * /usr/bin/certbot renew --quiet && docker run --rm -v letsencrypt:/etc/letsencrypt -v /etc/letsencrypt:/host-certs alpine cp -r /host-certs/. /etc/letsencrypt/ && cd $(pwd) && docker-compose exec nginx nginx -s reload"

# Check if cron job already exists
if ! crontab -l 2>/dev/null | grep -q "certbot renew"; then
    (crontab -l 2>/dev/null; echo "$CRON_JOB") | crontab -
    echo "✅ Automatic certificate renewal configured"
else
    echo "ℹ️  Certificate renewal cron job already exists"
fi

echo ""
echo "🎯 Deployment Summary:"
echo "   Domain: chirpid.utictactoe.online"
echo "   HTTP: Redirects to HTTPS"
echo "   HTTPS: Port 443 with SSL"
echo "   Backend: Accessible via /api/, /health, /ping"
echo "   Auto-renewal: Configured for certificates"
echo ""
echo "📝 Useful commands:"
echo "   View logs: docker-compose logs -f"
echo "   Restart: docker-compose restart"
echo "   Stop: docker-compose down"
echo "   Manual cert renewal: sudo certbot renew"
