# 🌞 SolarEye - Solar Panel Fault Detection System

<div align="center">
  <img src="Web_Implementation/images/logo1.png" alt="SolarEye Logo" width="200"/>
  
  **Advanced AI-powered solar panel condition analysis using YOLOv8 Classification**
  
  [![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
  [![Flask](https://img.shields.io/badge/flask-3.1.2-green.svg)](https://flask.palletsprojects.com/)
  [![YOLO](https://img.shields.io/badge/YOLO-v8-red.svg)](https://github.com/ultralytics/ultralytics)
  [![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
</div>

---

## 📋 Overview

**SolarEye** adalah sistem deteksi otomatis kondisi panel surya menggunakan teknologi AI (Artificial Intelligence) berbasis YOLOv8 Classification. Sistem ini dapat mengidentifikasi 6 kondisi panel surya:

- ✅ **Clean** - Panel dalam kondisi sempurna
- ⚠️ **Dusty** - Panel tertutup debu/kotoran
- ⚠️ **Bird-drop** - Terdapat kotoran burung
- 🔴 **Snow-Covered** - Panel tertutup salju
- 🔴 **Electrical-damage** - Kerusakan elektrik
- 🔴 **Physical-Damage** - Kerusakan fisik

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Git
- pip atau conda

### Installation

**1. Clone Repository**
```bash
git clone https://github.com/username/solar-panel-fault-detection.git
cd solar-panel-fault-detection
```

**2. Create Virtual Environment**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

**3. Install Dependencies**
```bash
pip install -r requirements.txt
```

**4. Run API Server**
```bash
python api_server.py
```
Server akan berjalan di `http://127.0.0.1:5000`

**5. Open Web Interface**
- Buka file `Web_Implementation/index.html` di browser
- Atau akses melalui local file path: `file:///path/to/Web_Implementation/index.html`

---

## 📁 Project Structure

```
solar-panel-fault-detection/
│
├── Web_Implementation/          # Frontend - HTML/CSS/JavaScript
│   ├── index.html              # Main web interface
│   ├── panel-styles.css        # Custom styling
│   ├── panel-predictor.js      # API client library
│   └── images/                 # Assets & images
│
├── models/saved_models/        # Pre-trained YOLO model
│   └── best_solar_panel_classifier.pt
│
├── api_server.py               # Flask REST API backend
├── inference_helper.py         # YOLO model wrapper
│
├── data/                       # Dataset directories
│   ├── dataset/               # Original images
│   └── yolo_classify_dataset/ # YOLO formatted dataset
│
├── requirements.txt            # Python dependencies
├── .gitignore                 # Git ignore rules
└── README.md                  # This file
```

---

## 🛠️ Tech Stack

### Backend
- **Flask 3.1.2** - Web framework
- **Flask-CORS 6.0.2** - Cross-origin requests
- **Ultralytics YOLO v8** - Computer vision model
- **PyTorch 2.9.1** - Deep learning framework
- **Pillow 12.1.0** - Image processing

### Frontend
- **HTML5** - Markup
- **Tailwind CSS** - Styling framework
- **Font Awesome** - Icon library
- **Vanilla JavaScript** - Interactive features

### Deployment
- **Vercel** - Frontend hosting
- **Heroku / Railway** - Backend hosting (optional)

---

## 🎯 Features

✨ **Web Interface**
- Drag & drop image upload
- Real-time prediction
- Animated loading spinner
- Custom notification toasts
- Responsive design

🤖 **AI Model**
- YOLOv8 Classification
- 98.06% accuracy on test set
- 6-class classification
- Confidence score display
- Top-5 predictions

📊 **Analysis Output**
- Prediction class & confidence
- Status assessment (GOOD/WARNING/CRITICAL)
- Risk level analysis
- Maintenance recommendations
- Color-coded urgency levels

⚠️ **Validation**
- Low confidence detection (< 50%)
- Non-panel image rejection
- Input validation
- Error handling

---

## 🔌 API Documentation

### Predict Endpoint
```
POST /api/predict
```

**Request:**
```
Content-Type: multipart/form-data
Parameter: image (file)
```

**Response (Success):**
```json
{
  "success": true,
  "prediction": {
    "class": "Clean",
    "confidence": 95.23,
    "class_idx": 1
  },
  "all_probabilities": {
    "Clean": 95.23,
    "Dusty": 2.45,
    "Bird-drop": 1.20,
    "Snow-Covered": 0.87,
    "Electrical-damage": 0.15,
    "Physical-Damage": 0.10
  },
  "info": {
    "status": "GOOD",
    "color": "#27ae60",
    "description": "Panel dalam kondisi sempurna",
    "urgency": "Low",
    "maintenance": "Lakukan pemeriksaan rutin setiap 3 bulan",
    "risk": "Minimal"
  },
  "warning": null
}
```

**Response (Low Confidence):**
```json
{
  "warning": {
    "message": "Low confidence detected",
    "details": "The model is not very confident...",
    "suggestion": "Please try uploading a clear, well-lit image",
    "severity": "warning"
  }
}
```

### Health Check
```
GET /
```

---

## 🚀 Deployment

### Option 1: Deploy Frontend to Vercel

**1. Prepare Files**
```bash
# Organize files for Vercel
# Vercel akan serve static files dari root atau public folder
```

**2. Create vercel.json**
```json
{
  "buildCommand": "echo 'Static site'",
  "outputDirectory": "Web_Implementation",
  "rewrites": [
    {
      "source": "/(.*)",
      "destination": "/index.html"
    }
  ]
}
```

**3. Push to GitHub**
```bash
git add .
git commit -m "Initial commit: SolarEye frontend"
git push origin main
```

**4. Deploy via Vercel**
- Buka https://vercel.com
- Click "New Project"
- Import GitHub repository
- Set root directory: `Web_Implementation`
- Deploy!

### Option 2: Deploy Backend to Railway/Heroku

**Railway (Recommended):**
```bash
# 1. Sign up di railway.app
# 2. Connect GitHub
# 3. Railway otomatis detect Python
# 4. Set environment variables jika perlu
# 5. Deploy!
```

**Heroku:**
```bash
# 1. heroku login
# 2. heroku create app-name
# 3. git push heroku main
```

---

## 📝 Environment Variables

Create `.env` file untuk production:
```env
FLASK_ENV=production
API_URL=your-backend-url.com
DEBUG=False
```

---

## 🧪 Testing

Test dengan gambar sample:
```bash
# Python test
python -c "
from inference_helper import SolarPanelInference
model = SolarPanelInference('models/saved_models/best_solar_panel_classifier.pt')
result = model.predict('path/to/image.jpg')
print(result)
"
```

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Accuracy | 98.06% |
| Precision | 97.5% |
| Recall | 97.8% |
| F1-Score | 97.65% |
| Training Time | ~45 min |
| Model Size | 45 MB |

---

## 🐛 Troubleshooting

**Problem: "Model not loaded"**
```
Solusi: Pastikan file models/saved_models/best_solar_panel_classifier.pt ada
```

**Problem: CORS error**
```
Solusi: Pastikan api_server.py menggunakan CORS(app)
```

**Problem: Image upload failed**
```
Solusi: Check file size (max ~10MB recommended)
        Check file format (JPG, PNG, WebP)
```

---

## 📄 License

MIT License - See LICENSE file for details

---

## 👥 Contributors

- **Your Name** - Main Developer

---

## 📞 Contact

- Email: your.email@example.com
- GitHub: [@yourusername](https://github.com/yourusername)

---

## 🙏 Acknowledgments

- Ultralytics YOLO team
- Flask & Python community
- Vercel platform

---

**Made with ❤️ for Solar Panel Monitoring**
