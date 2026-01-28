#!/bin/bash

# 🚀 INSTALLATION & DEPLOYMENT SCRIPT
# Untuk quick setup production environment

echo "======================================"
echo "☀️  Solar Panel Monitor - Setup"
echo "======================================"

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="Linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macOS"
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    OS="Windows"
else
    OS="Unknown"
fi

echo "Detected OS: $OS"

# Step 1: Check Python
echo ""
echo "✓ Checking Python..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.10+"
    exit 1
fi
python3 --version

# Step 2: Create virtual environment
echo ""
echo "✓ Creating virtual environment..."
python3 -m venv venv

# Activate venv
if [[ "$OS" == "Windows" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# Step 3: Install dependencies
echo ""
echo "✓ Installing dependencies..."
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# Step 4: Download model (if not exists)
echo ""
echo "✓ Checking model..."
if [ ! -f "models/saved_models/best_solar_panel_classifier.pt" ]; then
    echo "⚠️  Model not found. Please ensure model file exists:"
    echo "   models/saved_models/best_solar_panel_classifier.pt"
else
    echo "✅ Model found!"
fi

# Step 5: Test imports
echo ""
echo "✓ Testing imports..."
python3 -c "from inference_helper import SolarPanelInference; print('✅ Imports OK')" || {
    echo "❌ Import failed"
    exit 1
}

# Step 6: Offer to run app
echo ""
echo "======================================"
echo "✅ Setup Complete!"
echo "======================================"
echo ""
echo "📍 To run the web app:"
echo "   streamlit run app.py"
echo ""
echo "📍 To run with Docker:"
echo "   docker-compose up --build"
echo ""
echo "📍 To deploy to cloud:"
echo "   See DEPLOYMENT_GUIDE.md"
echo ""
