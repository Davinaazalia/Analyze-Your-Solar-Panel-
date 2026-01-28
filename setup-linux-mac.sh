#!/bin/bash
# 🌞 SolarEye Quick Setup Script

echo "╔════════════════════════════════════════════╗"
echo "║  🌞 SolarEye Integration Setup             ║"
echo "╚════════════════════════════════════════════╝"
echo ""

# Check Python
echo "[1/4] Checking Python..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found"
    exit 1
fi
python3 --version
echo ""

# Install dependencies
echo "[2/4] Installing dependencies..."
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed"
else
    echo "❌ Failed to install dependencies"
    exit 1
fi
echo ""

# Check model
echo "[3/4] Checking model file..."
if [ -f "models/saved_models/best_solar_panel_classifier.pt" ]; then
    echo "✅ Model found"
else
    echo "⚠️  WARNING: Model file not found"
fi
echo ""

# Done
echo "[4/4] Setup completed!"
echo ""
echo "╔════════════════════════════════════════════╗"
echo "║  Next Steps:                               ║"
echo "╚════════════════════════════════════════════╝"
echo ""
echo "1. Start API server:"
echo "   python3 api_server.py"
echo ""
echo "2. Open in browser:"
echo "   Web_Implementation/index-integrated.html"
echo ""
echo "3. Or test with:"
echo "   python3 test_integration.py"
echo ""
echo "📚 Documentation: INTEGRATION_GUIDE.md"
echo ""
