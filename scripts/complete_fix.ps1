# Complete ChirpID Fix Script (PowerShell)
# This script fixes both the model file and code issues

Write-Host "🔧 ChirpID Complete Fix Script" -ForegroundColor Cyan
Write-Host "=============================" -ForegroundColor Cyan

$ContainerName = "chirpid-backend"
$LocalModel = "models\bird_cnn.pth"

Write-Host "1️⃣ Checking current status..." -ForegroundColor Yellow

# Check if local model exists and its size
if (Test-Path $LocalModel) {
    $LocalFile = Get-Item $LocalModel
    $LocalSize = $LocalFile.Length
    $LocalSizeMB = [math]::Round($LocalSize / 1MB, 2)
    Write-Host "📁 Local model: $LocalSize bytes ($LocalSizeMB MB)"
    
    if ($LocalSize -lt 1000000) {
        Write-Host "⚠️  Local model file appears to be a Git LFS pointer (too small)" -ForegroundColor Yellow
        Write-Host "   Try running: git lfs pull" -ForegroundColor Yellow
        Write-Host "   Or copy the real model file manually" -ForegroundColor Yellow
    }
} else {
    Write-Host "❌ Local model file not found: $LocalModel" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "2️⃣ Stopping and rebuilding container..." -ForegroundColor Yellow

# Stop container
Write-Host "🛑 Stopping container..."
docker-compose down

# Rebuild with no cache to ensure latest code
Write-Host "🔨 Rebuilding container with latest code..."
docker-compose build --no-cache

# Start container
Write-Host "🚀 Starting container..."
docker-compose up -d

# Wait for container to be ready
Write-Host "⏳ Waiting for container to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

Write-Host ""
Write-Host "3️⃣ Copying model file..." -ForegroundColor Yellow

# Copy model file
docker cp $LocalModel "${ContainerName}:/app/models/bird_cnn.pth"

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Model file copied" -ForegroundColor Green
    
    # Verify size in container
    $ContainerSize = docker exec $ContainerName stat -c%s "/app/models/bird_cnn.pth"
    $ContainerSizeMB = [math]::Round([int]$ContainerSize / 1MB, 2)
    Write-Host "📁 Container model: $ContainerSize bytes ($ContainerSizeMB MB)"
} else {
    Write-Host "❌ Failed to copy model file" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "4️⃣ Testing the fix..." -ForegroundColor Yellow

# Test health endpoint
Write-Host "🧪 Testing health endpoint..." -ForegroundColor Yellow
Start-Sleep -Seconds 5  # Give the server a moment to start

# Use Python health check script
try {
    python scripts\health_check.py "http://localhost:5001"
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Health check passed!" -ForegroundColor Green
    } else {
        Write-Host "❌ Health check failed!" -ForegroundColor Red
    }
} catch {
    Write-Host "⚠️  Could not run Python health check: $($_.Exception.Message)" -ForegroundColor Yellow
    
    # Fallback to PowerShell health check
    for ($i = 1; $i -le 5; $i++) {
        try {
            $response = Invoke-WebRequest -Uri "http://localhost:5001/api/audio/health" -TimeoutSec 5 -ErrorAction Stop
            if ($response.StatusCode -eq 200) {
                Write-Host "✅ Health endpoint is responding (fallback check)" -ForegroundColor Green
                break
            }
        } catch {
            Write-Host "⏳ Attempt $i/5: Health endpoint not ready, waiting..." -ForegroundColor Yellow
            Start-Sleep -Seconds 3
        }
    }
}

# Show detailed health info
Write-Host ""
Write-Host "📊 Detailed health check:" -ForegroundColor Cyan
try {
    $healthData = Invoke-RestMethod -Uri "http://localhost:5001/api/audio/health"
    $healthData | ConvertTo-Json -Depth 10
} catch {
    Write-Host "Could not retrieve detailed health info: $($_.Exception.Message)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "🎉 Fix completed!" -ForegroundColor Green
Write-Host ""
Write-Host "📋 Next steps:" -ForegroundColor Cyan
Write-Host "  • Monitor logs: docker logs -f $ContainerName" -ForegroundColor White
Write-Host "  • Test upload: Use your frontend or curl with a test file" -ForegroundColor White
Write-Host "  • If issues persist, check the logs for specific errors" -ForegroundColor White
