# Backend Log Checker Script (PowerShell)
# This script helps you quickly check backend logs and health

Write-Host "ChirpID Backend Log Checker" -ForegroundColor Cyan
Write-Host "===========================" -ForegroundColor Cyan

# Check if server is running
Write-Host "1. Checking if backend server is running..." -ForegroundColor Yellow
$ServerURL = "http://localhost:5001"

try {
    $response = Invoke-WebRequest -Uri "$ServerURL/ping" -Method GET -TimeoutSec 5
    Write-Host "✅ Backend server is running" -ForegroundColor Green
} catch {
    Write-Host "❌ Backend server is not responding" -ForegroundColor Red
    Write-Host "   Make sure your backend is running on port 5001" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "2. Checking backend health..." -ForegroundColor Yellow
Write-Host "----------------------------" -ForegroundColor Yellow

try {
    $healthResponse = Invoke-RestMethod -Uri "$ServerURL/api/audio/health" -Method GET
    $healthResponse | ConvertTo-Json -Depth 10
} catch {
    Write-Host "Health check failed: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host ""
Write-Host "3. To see live backend logs, check:" -ForegroundColor Yellow
Write-Host "   - The terminal/command prompt where you started the backend" -ForegroundColor Cyan
Write-Host "   - If using systemd: sudo journalctl -u your-service-name -f" -ForegroundColor Cyan
Write-Host "   - If using Docker: docker logs container-name -f" -ForegroundColor Cyan

Write-Host ""
Write-Host "4. To test with a sample file:" -ForegroundColor Yellow
Write-Host "   curl -X POST -F `"file=@sample.wav`" $ServerURL/api/audio/upload" -ForegroundColor Cyan
