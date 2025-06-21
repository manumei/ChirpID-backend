# Create external volume for Let's Encrypt certificates
docker volume create letsencrypt

# Install Certbot on host (Ubuntu/Debian)
sudo apt update
sudo apt install certbot

# Create initial certificates for chirpid.utictactoe.online
sudo certbot certonly --standalone -d chirpid.utictactoe.online --email admin@utictactoe.online --agree-tos --non-interactive

# Copy certificates to Docker volume
sudo docker run --rm -v letsencrypt:/etc/letsencrypt -v /etc/letsencrypt:/host-certs alpine cp -r /host-certs/. /etc/letsencrypt/

# Start services
docker-compose up -d

# Renewal command (add to crontab)
# 0 12 * * * /usr/bin/certbot renew --quiet && docker run --rm -v letsencrypt:/etc/letsencrypt -v /etc/letsencrypt:/host-certs alpine cp -r /host-certs/. /etc/letsencrypt/ && docker-compose exec nginx nginx -s reload

# Manual renewal and reload
sudo certbot renew
sudo docker run --rm -v letsencrypt:/etc/letsencrypt -v /etc/letsencrypt:/host-certs alpine cp -r /host-certs/. /etc/letsencrypt/
docker-compose exec nginx nginx -s reload
