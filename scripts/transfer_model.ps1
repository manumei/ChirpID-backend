# Model Transfer and Verification Script (PowerShell)
# This script helps transfer the PyTorch model file to Ubuntu server safely

param(
    [string]$LocalModelPath = "models\bird_cnn.pth",
    [string]$ServerUser,
    [string]$ServerIP,
    [string]$ServerPath,
    [switch]$Help
)

function Show-Usage {
    Write-Host "ChirpID Model Transfer Script (PowerShell)" -ForegroundColor Cyan
    Write-Host "=========================================="
    Write-Host ""
    Write-Host "Usage: .\transfer_model.ps1 [OPTIONS]"
    Write-Host "Options:"
    Write-Host "  -LocalModelPath PATH   Local path to model file (default: models\bird_cnn.pth)"
    Write-Host "  -ServerUser USER       Server username"
    Write-Host "  -ServerIP IP           Server IP address"
    Write-Host "  -ServerPath PATH       Remote path on server"
    Write-Host "  -Help                  Show this help"
    Write-Host ""
    Write-Host "Example:"
    Write-Host "  .\transfer_model.ps1 -ServerUser ubuntu -ServerIP 192.168.1.100 -ServerPath /home/ubuntu/chirpid-backend/models/bird_cnn.pth"
    Write-Host ""
    Write-Host "Requirements:"
    Write-Host "  - scp command available (install OpenSSH client or use WSL)"
    Write-Host "  - SSH key authentication configured"
}

if ($Help) {
    Show-Usage
    exit 0
}

Write-Host "ChirpID Model Transfer Script" -ForegroundColor Cyan
Write-Host "============================="

# Validate required parameters
if (-not $ServerUser -or -not $ServerIP -or -not $ServerPath) {
    Write-Host "Error: Missing required parameters" -ForegroundColor Red
    Write-Host ""
    Show-Usage
    exit 1
}

Write-Host "Configuration:"
Write-Host "  Local model: $LocalModelPath"
Write-Host "  Server: $ServerUser@$ServerIP"
Write-Host "  Remote path: $ServerPath"
Write-Host ""

# Check if local model exists
if (-not (Test-Path $LocalModelPath)) {
    Write-Host "❌ Local model file not found: $LocalModelPath" -ForegroundColor Red
    exit 1
}

# Get local file info
$LocalFile = Get-Item $LocalModelPath
$LocalSize = $LocalFile.Length
$LocalSizeMB = [math]::Round($LocalSize / 1MB, 2)

Write-Host "Local file info:"
Write-Host "  Size: $LocalSize bytes ($LocalSizeMB MB)"

# Calculate MD5 hash
$LocalMD5 = (Get-FileHash $LocalModelPath -Algorithm MD5).Hash.ToLower()
Write-Host "  MD5: $LocalMD5"
Write-Host ""

# Check if scp is available
try {
    $null = Get-Command scp -ErrorAction Stop
} catch {
    Write-Host "❌ scp command not found. Please install OpenSSH client or use WSL." -ForegroundColor Red
    Write-Host "   Windows 10/11: Settings > Apps > Optional Features > Add OpenSSH Client"
    exit 1
}

# Create remote directory if needed
Write-Host "Creating remote directory..."
$RemoteDir = Split-Path $ServerPath -Parent
ssh "$ServerUser@$ServerIP" "mkdir -p $RemoteDir"

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to create remote directory" -ForegroundColor Red
    exit 1
}

# Transfer file with verification
Write-Host "Transferring file..."
scp -C $LocalModelPath "$ServerUser@$ServerIP`:$ServerPath"

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ File transfer failed" -ForegroundColor Red
    exit 1
}

# Verify transfer
Write-Host "Verifying transfer..."
$RemoteSize = ssh "$ServerUser@$ServerIP" "stat -c%s $ServerPath 2>/dev/null || echo 0"
$RemoteMD5 = ssh "$ServerUser@$ServerIP" "md5sum $ServerPath | cut -d' ' -f1 2>/dev/null || echo 'unknown'"

Write-Host "Remote file info:"
Write-Host "  Size: $RemoteSize bytes"
Write-Host "  MD5: $RemoteMD5"
Write-Host ""

# Compare checksums
if ($LocalMD5 -eq $RemoteMD5.Trim()) {
    Write-Host "✅ File transfer verified successfully!" -ForegroundColor Green
} else {
    Write-Host "❌ File transfer verification failed!" -ForegroundColor Red
    Write-Host "   Local MD5:  $LocalMD5"
    Write-Host "   Remote MD5: $($RemoteMD5.Trim())"
    exit 1
}

# Test model loading on server
Write-Host "Testing model loading on server..."
$RemoteRepoDir = Split-Path $RemoteDir -Parent
ssh "$ServerUser@$ServerIP" "cd $RemoteRepoDir && python scripts/validate_model.py"

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Model validation successful on server!" -ForegroundColor Green
} else {
    Write-Host "❌ Model validation failed on server!" -ForegroundColor Red
    Write-Host "Check the server logs and PyTorch version compatibility."
    exit 1
}

Write-Host ""
Write-Host "🎉 Model transfer and validation completed successfully!" -ForegroundColor Green
Write-Host "You can now restart your ChirpID backend service."
