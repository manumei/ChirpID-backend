# SSH Deployment Setup Guide

This guide will help you configure GitHub Actions to deploy your ChirpID backend to your Ubuntu server via SSH.

## Prerequisites

1. **Ubuntu Server** running on your laptop
2. **SSH access** configured on the Ubuntu server
3. **GitHub repository** with the ChirpID backend code

## Step 1: Prepare Your Ubuntu Server

### 1.1 Enable SSH (if not already enabled)

```bash
# On your Ubuntu server
sudo apt update
sudo apt install openssh-server
sudo systemctl start ssh
sudo systemctl enable ssh
```

### 1.2 Find your server's IP address

```bash
# Get local IP address
ip addr show | grep "inet " | grep -v 127.0.0.1

# Or use hostname -I
hostname -I
```

### 1.3 Create a deployment user (recommended)

```bash
# Create a dedicated user for deployments
sudo adduser chirpid-deploy
sudo usermod -aG docker chirpid-deploy
sudo usermod -aG sudo chirpid-deploy

# Switch to the new user
sudo su - chirpid-deploy
```

## Step 2: Generate SSH Key Pair

### 2.1 Generate SSH keys on your development machine

```bash
# Generate a new SSH key pair for GitHub Actions
ssh-keygen -t ed25519 -C "github-actions-chirpid" -f ~/.ssh/chirpid-deploy

# This creates:
# - ~/.ssh/chirpid-deploy (private key)
# - ~/.ssh/chirpid-deploy.pub (public key)
```

### 2.2 Copy public key to Ubuntu server

```bash
# Copy the public key to your Ubuntu server
ssh-copy-id -i ~/.ssh/chirpid-deploy.pub chirpid-deploy@YOUR_UBUNTU_IP

# Or manually:
cat ~/.ssh/chirpid-deploy.pub | ssh chirpid-deploy@YOUR_UBUNTU_IP "mkdir -p ~/.ssh && **cat** >> ~/.ssh/authorized_keys"
```

### 2.3 Test SSH connection

```bash
# Test the connection
ssh -i ~/.ssh/chirpid-deploy chirpid-deploy@YOUR_UBUNTU_IP
```

## Step 3: Configure GitHub Secrets

Go to your GitHub repository → Settings → Secrets and variables → Actions

Add the following secrets:

### Required Secrets:

1. **`SSH_HOST`**

   - Value: Your Ubuntu server's IP address (e.g., `192.168.1.100`)

2. **`SSH_USERNAME`**

   - Value: `chirpid-deploy` (or your chosen username)

3. **`SSH_PRIVATE_KEY`**
   - Value: Contents of the private key file
   ```bash
   cat ~/.ssh/chirpid-deploy
   ```
   Copy the entire output including `-----BEGIN OPENSSH PRIVATE KEY-----` and `-----END OPENSSH PRIVATE KEY-----`

### Optional Secrets:

4. **`SSH_PORT`** (optional, defaults to 22)
   - Value: `22` (or your custom SSH port)

## Step 4: Configure Firewall (if enabled)

```bash
# On your Ubuntu server
sudo ufw allow ssh
sudo ufw allow 5000/tcp  # For the ChirpID backend
sudo ufw enable
```

## Step 5: Test the Deployment

### 5.1 Push code to trigger deployment

```bash
git add .
git commit -m "Configure SSH deployment"
git push origin main  # or master
```

### 5.2 Monitor the deployment

1. Go to your GitHub repository
2. Click on "Actions" tab
3. Watch the deployment workflow run

### 5.3 Verify deployment

Once the workflow completes:

```bash
# Check if the service is running
curl http://YOUR_UBUNTU_IP:5001/health

# Or from your Ubuntu server
curl http://localhost:5001/health
```

## Step 6: Access Your Application

Your ChirpID backend will be available at:

- **Local access**: `http://localhost:5001`
- **Network access**: `http://YOUR_UBUNTU_IP:5001`

### Important endpoints:

- Health check: `http://YOUR_UBUNTU_IP:5001/health`
- Ping: `http://YOUR_UBUNTU_IP:5001/ping`
- API: `http://YOUR_UBUNTU_IP:5001/api/audio/*`

## Troubleshooting

### SSH Connection Issues

```bash
# Test SSH connection manually
ssh -i ~/.ssh/chirpid-deploy chirpid-deploy@YOUR_UBUNTU_IP

# Check SSH service status on Ubuntu
sudo systemctl status ssh

# Check SSH logs
sudo tail -f /var/log/auth.log
```

### Docker Issues

```bash
# On Ubuntu server, check Docker status
sudo systemctl status docker

# Check container logs
docker logs chirpid-backend

# Check running containers
docker ps
```

### Firewall Issues

```bash
# Check firewall status
sudo ufw status

# Allow the application port
sudo ufw allow 5000/tcp
```

### GitHub Actions Issues

1. Check the Actions logs in GitHub
2. Verify all secrets are correctly set
3. Ensure the private key format is correct (include the full key with headers)

## Security Best Practices

1. **Use a dedicated deployment user** instead of your personal account
2. **Limit SSH key permissions** to only what's needed
3. **Consider using a non-standard SSH port**
4. **Enable fail2ban** for additional SSH protection:
   ```bash
   sudo apt install fail2ban
   sudo systemctl enable fail2ban
   ```
5. **Use firewall rules** to restrict access to necessary ports only

## Manual Deployment (Backup Method)

If GitHub Actions fails, you can deploy manually:

```bash
# SSH into your server
ssh chirpid-deploy@YOUR_UBUNTU_IP

# Pull and run the latest image
docker pull ghcr.io/YOUR_USERNAME/chirpid-backend/chirpid-backend:latest
docker stop chirpid-backend || true
docker rm chirpid-backend || true
docker run -d --name chirpid-backend --restart unless-stopped -p 5000:5001 \
  -v ~/chirpid/uploads:/app/app/uploads \
  -v ~/chirpid/database:/app/database \
  ghcr.io/YOUR_USERNAME/chirpid-backend/chirpid-backend:latest
```

## Next Steps

1. **Set up SSL/HTTPS** using Let's Encrypt
2. **Configure a reverse proxy** (Nginx) for better performance
3. **Set up monitoring** and logging
4. **Configure automatic backups** for your data

That's it! Your ChirpID backend should now automatically deploy to your Ubuntu server whenever you push code to the main branch.
