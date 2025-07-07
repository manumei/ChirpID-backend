# Fix Model File Script (PowerShell)
# This script helps fix the Git LFS pointer issue by copying the real model file

Write-Host "🔧 ChirpID Model File Fixer" -ForegroundColor Cyan
Write-Host "==========================" -ForegroundColor Cyan

# Configuration
$LocalModelPath = "models\bird_cnn.pth"
$ServerModelPath = "/app/models/bird_cnn.pth"
$ContainerName = "chirpid-backend"

# Check if running from correct directory
if (-not (Test-Path $LocalModelPath)) {
    Write-Host "❌ Model file not found at: $LocalModelPath" -ForegroundColor Red
    Write-Host "Please run this script from the chirpid-backend directory" -ForegroundColor Red
    exit 1
}

# Check local model file
$LocalFile = Get-Item $LocalModelPath
$LocalSize = $LocalFile.Length
$LocalSizeMB = [math]::Round($LocalSize / 1MB, 2)

Write-Host "📁 Local model file size: $LocalSize bytes ($LocalSizeMB MB)"

if ($LocalSize -lt 1000000) {
    Write-Host "⚠️  WARNING: Local model file seems too small. It might be a Git LFS pointer." -ForegroundColor Yellow
    Write-Host "   If you're using Git LFS, run: git lfs pull" -ForegroundColor Yellow
    $continue = Read-Host "   Continue anyway? (y/N)"
    if ($continue -notmatch "^[Yy]$") {
        exit 1
    }
}

# Check if Docker container is running
$containerRunning = docker ps --filter "name=$ContainerName" --format "{{.Names}}"
if (-not $containerRunning) {
    Write-Host "❌ Docker container '$ContainerName' is not running" -ForegroundColor Red
    Write-Host "   Start it with: docker-compose up -d" -ForegroundColor Red
    exit 1
}

Write-Host "🚀 Copying model file to Docker container..." -ForegroundColor Green

# Copy file to container
docker cp $LocalModelPath "${ContainerName}:${ServerModelPath}"

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Model file copied successfully!" -ForegroundColor Green
    
    # Verify the copy
    Write-Host "🔍 Verifying copied file..." -ForegroundColor Yellow
    $ContainerSize = docker exec $ContainerName stat -c%s $ServerModelPath
    Write-Host "📁 Container model file size: $ContainerSize bytes"
    
    if ($LocalSize -eq [int]$ContainerSize) {
        Write-Host "✅ File sizes match - copy verified!" -ForegroundColor Green
        
        # Test model loading
        Write-Host "🧪 Testing model loading..." -ForegroundColor Yellow
        $testScript = @"
import torch
try:
    checkpoint = torch.load('$ServerModelPath', map_location='cpu', weights_only=False)
    print('✅ Model loads successfully!')
    print(f'📊 Model type: {type(checkpoint)}')
except Exception as e:
    print(f'❌ Model loading failed: {e}')
"@
        docker exec $ContainerName python -c $testScript
    } else {
        Write-Host "⚠️  WARNING: File sizes don't match!" -ForegroundColor Yellow
        Write-Host "   Local: $LocalSize bytes" -ForegroundColor Yellow
        Write-Host "   Container: $ContainerSize bytes" -ForegroundColor Yellow
    }
} else {
    Write-Host "❌ Failed to copy model file" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "🎉 Model file fix completed!" -ForegroundColor Green
Write-Host "💡 Now test with: curl http://localhost:5001/api/audio/health" -ForegroundColor Cyan
