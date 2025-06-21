#!/bin/bash

# ChirpID Backend Deployment Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="ghcr.io/your-username/chirpid-backend/chirpid-backend"
CONTAINER_NAME="chirpid-backend"

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker and try again."
        exit 1
    fi
}

# Function to pull latest image
pull_image() {
    print_status "Pulling latest image: $IMAGE_NAME:latest"
    docker pull $IMAGE_NAME:latest
}

# Function to stop and remove existing container
stop_container() {
    if docker ps -q -f name=$CONTAINER_NAME | grep -q .; then
        print_status "Stopping existing container: $CONTAINER_NAME"
        docker stop $CONTAINER_NAME
    fi
    
    if docker ps -aq -f name=$CONTAINER_NAME | grep -q .; then
        print_status "Removing existing container: $CONTAINER_NAME"
        docker rm $CONTAINER_NAME
    fi
}

# Function to start new container
start_container() {
    print_status "Starting new container: $CONTAINER_NAME"
    docker run -d \
        --name $CONTAINER_NAME \
        -p 5000:5001 \
        -v $(pwd)/app/uploads:/app/app/uploads \
        -v $(pwd)/database:/app/database \
        --restart unless-stopped \
        $IMAGE_NAME:latest
}

# Function to check container health
check_health() {
    print_status "Checking container health..."
    sleep 10
    
    if docker ps -q -f name=$CONTAINER_NAME | grep -q .; then
        # Wait for the service to be ready
        for i in {1..30}; do
            if curl -f http://localhost:5001/health > /dev/null 2>&1; then
                print_status "Container is healthy and responding!"
                return 0
            fi
            echo "Waiting for service to be ready... ($i/30)"
            sleep 2
        done
        print_warning "Service might not be fully ready yet. Check logs with: docker logs $CONTAINER_NAME"
    else
        print_error "Container failed to start!"
        exit 1
    fi
}

# Function to show logs
show_logs() {
    print_status "Recent logs:"
    docker logs --tail 20 $CONTAINER_NAME
}

# Main deployment function
deploy() {
    print_status "Starting ChirpID Backend deployment..."
    
    check_docker
    pull_image
    stop_container
    start_container
    check_health
    show_logs
    
    print_status "Deployment completed successfully!"
    print_status "Service is available at: http://localhost:5001"
    print_status "Health check: http://localhost:5001/health"
}

# Function to deploy with docker-compose
deploy_compose() {
    print_status "Deploying with docker-compose (production mode)..."
    
    check_docker
    
    if [ ! -f "docker-compose.prod.yml" ]; then
        print_error "docker-compose.prod.yml not found!"
        exit 1
    fi
    
    # Update the image name in docker-compose.prod.yml if needed
    print_status "Pulling latest images..."
    docker-compose -f docker-compose.prod.yml pull
    
    print_status "Starting services..."
    docker-compose -f docker-compose.prod.yml up -d
    
    print_status "Checking service health..."
    sleep 15
    
    if curl -f http://localhost/health > /dev/null 2>&1; then
        print_status "Services are healthy and responding!"
    else
        print_warning "Services might not be fully ready yet."
        print_status "Check logs with: docker-compose -f docker-compose.prod.yml logs"
    fi
    
    print_status "Deployment completed successfully!"
    print_status "Service is available at: http://localhost"
}

# Function to build and deploy locally
build_and_deploy() {
    print_status "Building and deploying locally..."
    
    check_docker
    
    print_status "Building Docker image..."
    docker build -t chirpid-backend:local .
    
    print_status "Stopping existing container..."
    docker stop $CONTAINER_NAME 2>/dev/null || true
    docker rm $CONTAINER_NAME 2>/dev/null || true
    
    print_status "Starting new container with local image..."
    docker run -d \
        --name $CONTAINER_NAME \
        -p 5000:5001 \
        -v $(pwd)/app/uploads:/app/app/uploads \
        -v $(pwd)/database:/app/database \
        --restart unless-stopped \
        chirpid-backend:local
    
    check_health
    show_logs
    
    print_status "Local deployment completed successfully!"
}

# Function to show usage
usage() {
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  deploy          Deploy using pre-built image from registry"
    echo "  deploy-compose  Deploy using docker-compose (with nginx)"
    echo "  build           Build and deploy locally"
    echo "  logs            Show container logs"
    echo "  stop            Stop the container"
    echo "  restart         Restart the container"
    echo "  status          Show container status"
    echo "  help            Show this help message"
    echo ""
}

# Main script logic
case "${1:-deploy}" in
    deploy)
        deploy
        ;;
    deploy-compose)
        deploy_compose
        ;;
    build)
        build_and_deploy
        ;;
    logs)
        docker logs -f $CONTAINER_NAME
        ;;
    stop)
        print_status "Stopping container..."
        docker stop $CONTAINER_NAME 2>/dev/null || print_warning "Container was not running"
        ;;
    restart)
        print_status "Restarting container..."
        docker restart $CONTAINER_NAME
        ;;
    status)
        print_status "Container status:"
        docker ps -f name=$CONTAINER_NAME
        echo ""
        print_status "Service health:"
        curl -s http://localhost:5001/health | python -m json.tool 2>/dev/null || echo "Service not responding"
        ;;
    help)
        usage
        ;;
    *)
        print_error "Unknown command: $1"
        usage
        exit 1
        ;;
esac
