#!/bin/bash

# Service management commands for chirpid.utictactoe.online

# Start services
docker-compose up -d

# Restart services
docker-compose restart

# Reload Nginx configuration
docker-compose exec nginx nginx -s reload

# Check service status
docker-compose ps

# View logs
docker-compose logs -f

# Renew SSL certificates and reload Nginx
sudo certbot renew --quiet
sudo docker run --rm -v letsencrypt:/etc/letsencrypt -v /etc/letsencrypt:/host-certs alpine cp -r /host-certs/. /etc/letsencrypt/
docker-compose exec nginx nginx -s reload
