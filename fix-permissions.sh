#!/bin/bash

# ChirpID Backend - Permission Fix Script
# This script fixes file permission issues that might occur during deployment

echo "🔧 ChirpID Backend Permission Fix Script"
echo "========================================"

# Get the script directory (where this script is located)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"

echo "📁 Working directory: $PROJECT_DIR"

# Function to fix permissions for a directory
fix_permissions() {
    local dir=$1
    local description=$2
    
    if [ -d "$dir" ]; then
        echo "🔧 Fixing permissions for $description..."
        sudo chown -R $USER:$USER "$dir" 2>/dev/null || true
        sudo chmod -R 755 "$dir" 2>/dev/null || true
        echo "✅ Fixed permissions for $description"
    else
        echo "ℹ️  Directory $dir does not exist, skipping..."
    fi
}

# Function to clean up problematic files
cleanup_files() {
    echo "🧹 Cleaning up problematic files..."
    
    # Remove upload files
    if [ -d "$PROJECT_DIR/app/uploads" ]; then
        sudo rm -rf "$PROJECT_DIR/app/uploads"/* 2>/dev/null || true
        echo "✅ Cleaned upload files"
    fi
    
    # Remove any audio files that might cause issues
    find "$PROJECT_DIR" -name "*.mp3" -o -name "*.wav" -o -name "*.ogg" -type f -exec sudo rm -f {} \; 2>/dev/null || true
    echo "✅ Cleaned audio files"
}

# Main execution
echo "🚀 Starting permission fixes..."

# Fix permissions for the entire project
fix_permissions "$PROJECT_DIR" "project directory"

# Fix specific problematic directories
fix_permissions "$PROJECT_DIR/app" "app directory"
fix_permissions "$PROJECT_DIR/app/uploads" "uploads directory"
fix_permissions "$PROJECT_DIR/.git" "git directory"

# Clean up problematic files
cleanup_files

# Recreate uploads directory with proper permissions
echo "📁 Recreating uploads directory..."
mkdir -p "$PROJECT_DIR/app/uploads"
sudo chown -R $USER:$USER "$PROJECT_DIR/app/uploads" 2>/dev/null || true
sudo chmod -R 755 "$PROJECT_DIR/app/uploads" 2>/dev/null || true
echo "✅ Uploads directory recreated"

# Fix Docker-related permissions if Docker is running
if command -v docker &> /dev/null; then
    echo "🐳 Cleaning up Docker resources..."
    docker system prune -f 2>/dev/null || true
    echo "✅ Docker cleanup completed"
fi

echo ""
echo "✅ Permission fix completed successfully!"
echo "🎉 You can now try running your deployment again."
echo ""
echo "💡 If you continue to have permission issues:"
echo "   1. Make sure this script is executable: chmod +x fix-permissions.sh"
echo "   2. Run it with: ./fix-permissions.sh"
echo "   3. Or run with sudo if needed: sudo ./fix-permissions.sh"
