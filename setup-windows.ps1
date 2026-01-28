# 🌞 SolarEye Integration Setup Script (Windows PowerShell)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  SolarEye API Server Setup" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Python is installed
Write-Host "[1/4] Checking Python installation..." -ForegroundColor Green
python --version | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Python is not installed or not in PATH" -ForegroundColor Red
    exit 1
}
python --version
Write-Host ""

# Install/Update dependencies
Write-Host "[2/4] Installing/Updating dependencies..." -ForegroundColor Green
pip install --upgrade pip
pip install -r requirements.txt --quiet

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to install dependencies" -ForegroundColor Red
    exit 1
}
Write-Host "✅ Dependencies installed" -ForegroundColor Green
Write-Host ""

# Check if model file exists
Write-Host "[3/4] Checking model file..." -ForegroundColor Green
$modelPath = "models/saved_models/best_solar_panel_classifier.pt"
if (Test-Path $modelPath) {
    Write-Host "✅ Model found: $modelPath" -ForegroundColor Green
} else {
    Write-Host "⚠️ WARNING: Model file not found at $modelPath" -ForegroundColor Yellow
    Write-Host "Please ensure the model file is in the correct location" -ForegroundColor Yellow
}
Write-Host ""

# Ready to start
Write-Host "[4/4] Setup completed!" -ForegroundColor Green
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Next Steps" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Run the API server:" -ForegroundColor White
Write-Host "   python api_server.py" -ForegroundColor Yellow
Write-Host ""
Write-Host "2. Open in browser:" -ForegroundColor White
Write-Host "   Open: Web_Implementation/index-integrated.html" -ForegroundColor Yellow
Write-Host "   Or edit: Web_Implementation/index.html and add:" -ForegroundColor White
Write-Host "   - <link rel=""stylesheet"" href=""./panel-styles.css"">" -ForegroundColor Cyan
Write-Host "   - <script src=""./panel-predictor.js""></script>" -ForegroundColor Cyan
Write-Host ""
Write-Host "3. Test API:" -ForegroundColor White
Write-Host "   curl http://127.0.0.1:5000/" -ForegroundColor Yellow
Write-Host ""
Write-Host "Documentation: INTEGRATION_GUIDE.md" -ForegroundColor Magenta
Write-Host ""
