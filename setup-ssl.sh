#!/bin/bash

# SSL Certificate Setup for chirpid.utictactoe.online
set -e

echo "🔒 Setting up SSL certificates for chirpid.utictactoe.online"

# Create external volume for Let's Encrypt certificates
echo "📦 Creating Docker volume for SSL certificates..."
docker volume create letsencrypt 2>/dev/null || echo "Volume already exists"

# Stop any running services on ports 80/443
echo "🛑 Stopping existing services on ports 80/443..."
sudo fuser -k 80/tcp 2>/dev/null || true
sudo fuser -k 443/tcp 2>/dev/null || true
docker-compose down 2>/dev/null || true

# Generate SSL certificates
echo "🔐 Generating SSL certificates..."
sudo certbot certonly --standalone \
    -d chirpid.utictactoe.online \
    --email admin@utictactoe.online \
    --agree-tos \
    --non-interactive \
    --force-renewal

# Copy certificates to Docker volume
echo "📋 Copying certificates to Docker volume..."
sudo docker run --rm \
    -v letsencrypt:/etc/letsencrypt \
    -v /etc/letsencrypt:/host-certs \
    alpine cp -r /host-certs/. /etc/letsencrypt/

echo "✅ SSL certificates setup complete"
