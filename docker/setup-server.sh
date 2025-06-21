#!/bin/bash

# ChirpID Backend Server Setup Script
# Run this on your Ubuntu server to prepare it for automated deployments

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}[SETUP]${NC} $1"
}

# Function to check if running as root
check_root() {
    if [[ $EUID -eq 0 ]]; then
        print_error "This script should not be run as root!"
        print_status "Please run as a regular user with sudo privileges."
        exit 1
    fi
}

# Function to install Docker
install_docker() {
    print_header "Installing Docker..."
    
    if command -v docker &> /dev/null; then
        print_status "Docker is already installed"
        docker --version
        return 0
    fi
    
    # Install Docker
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    
    # Add current user to docker group
    sudo usermod -aG docker $USER
    
    # Start and enable Docker
    sudo systemctl start docker
    sudo systemctl enable docker
    
    print_status "Docker installed successfully!"
    print_warning "Please log out and log back in for docker group changes to take effect"
}

# Function to install required packages
install_packages() {
    print_header "Installing required packages..."
    
    sudo apt update
    sudo apt install -y \
        curl \
        wget \
        git \
        ufw \
        fail2ban \
        htop \
        unzip
    
    print_status "Required packages installed!"
}

# Function to configure firewall
configure_firewall() {
    print_header "Configuring firewall..."
    
    # Enable UFW if not already enabled
    if ! sudo ufw status | grep -q "Status: active"; then
        print_status "Enabling UFW firewall..."
        sudo ufw --force enable
    fi
    
    # Allow SSH
    sudo ufw allow ssh
    
    # Allow ChirpID backend port
    sudo ufw allow 5000/tcp
    
    # Show status
    sudo ufw status
    
    print_status "Firewall configured!"
}

# Function to create deployment directories
create_directories() {
    print_header "Creating deployment directories..."
    
    mkdir -p ~/chirpid/uploads
    mkdir -p ~/chirpid/database
    mkdir -p ~/chirpid/logs
    mkdir -p ~/chirpid/backup
    
    print_status "Deployment directories created:"
    ls -la ~/chirpid/
}

# Function to configure SSH (optional)
configure_ssh() {
    print_header "SSH Configuration"
    
    if systemctl is-active --quiet ssh; then
        print_status "SSH service is already running"
    else
        print_status "Starting SSH service..."
        sudo systemctl start ssh
        sudo systemctl enable ssh
    fi
    
    print_status "SSH service status:"
    sudo systemctl status ssh --no-pager -l
}

# Function to show network information
show_network_info() {
    print_header "Network Information"
    
    print_status "Server IP addresses:"
    ip addr show | grep "inet " | grep -v 127.0.0.1 | awk '{print $2}' | cut -d/ -f1
    
    echo ""
    print_status "Hostname:"
    hostname
    
    echo ""
    print_status "To access your ChirpID backend after deployment:"
    LOCAL_IP=$(hostname -I | awk '{print $1}')
    echo "  - Local: http://localhost:5001"
    echo "  - Network: http://$LOCAL_IP:5001"
}

# Function to create a sample systemd service (optional)
create_systemd_service() {
    print_header "Creating systemd service template..."
    
    cat > ~/chirpid/chirpid-backend.service << 'EOF'
[Unit]
Description=ChirpID Backend Container
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/usr/bin/docker run -d --name chirpid-backend --restart unless-stopped -p 5000:5001 -v %h/chirpid/uploads:/app/app/uploads -v %h/chirpid/database:/app/database -v %h/chirpid/logs:/app/logs ghcr.io/your-username/chirpid-backend/chirpid-backend:latest
ExecStop=/usr/bin/docker stop chirpid-backend
ExecStopPost=/usr/bin/docker rm chirpid-backend

[Install]
WantedBy=multi-user.target
EOF

    print_status "Systemd service template created at ~/chirpid/chirpid-backend.service"
    print_status "To install: sudo cp ~/chirpid/chirpid-backend.service /etc/systemd/system/"
}

# Function to show final instructions
show_final_instructions() {
    print_header "Setup Complete!"
    
    echo ""
    print_status "Next steps:"
    echo "1. Configure GitHub Secrets in your repository:"
    echo "   - SSH_HOST: $(hostname -I | awk '{print $1}')"
    echo "   - SSH_USERNAME: $USER"
    echo "   - SSH_PRIVATE_KEY: (contents of your private key)"
    echo ""
    echo "2. Generate SSH key pair on your development machine:"
    echo "   ssh-keygen -t ed25519 -C 'github-actions-chirpid' -f ~/.ssh/chirpid-deploy"
    echo ""
    echo "3. Copy public key to this server:"
    echo "   ssh-copy-id -i ~/.ssh/chirpid-deploy.pub $USER@$(hostname -I | awk '{print $1}')"
    echo ""
    echo "4. Test deployment by pushing code to your main branch"
    echo ""
    print_status "For detailed instructions, see DEPLOYMENT.md"
}

# Main execution
main() {
    print_header "ChirpID Backend Server Setup"
    echo "This script will prepare your Ubuntu server for automated deployments"
    echo ""
    
    check_root
    
    read -p "Continue with setup? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_status "Setup cancelled"
        exit 0
    fi
    
    install_packages
    install_docker
    configure_firewall
    configure_ssh
    create_directories
    create_systemd_service
    show_network_info
    show_final_instructions
    
    print_status "Server setup completed successfully!"
    print_warning "Please reboot or log out/in to ensure all changes take effect"
}

# Run main function
main "$@"
