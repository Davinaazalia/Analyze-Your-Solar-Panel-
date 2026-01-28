# 🚀 DEPLOYMENT GUIDE - Solar Panel Monitoring Web App

## 📋 Quick Start (Local)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Web App Locally
```bash
streamlit run app.py
```

Akses di: `http://localhost:8501`

---

## 🌐 Deploy ke Cloud (Pilihan)

### Option 1: Streamlit Cloud (EASIEST & FREE) ⭐

#### Setup:
1. Push project ke GitHub
2. Signup di https://streamlit.io/cloud
3. Deploy:
   - Click "New app"
   - Select repository: `your-username/solar-panel-fault-detection`
   - Select file: `app.py`
   - Click "Deploy"

#### URL Publik:
`https://solar-panel-monitor.streamlit.app`

**Keuntungan:**
- ✅ FREE tier available (up to 3 apps)
- ✅ Automatic deploys from GitHub
- ✅ HTTPS included
- ✅ No server management
- ✅ Auto-scaling

**Limitations:**
- Limited memory (1GB)
- May need to optimize model loading

#### Optimize untuk Streamlit Cloud:
```python
# Di app.py, tambahkan caching:
@st.cache_resource
def load_model():
    return SolarPanelInference()

model = load_model()
```

---

### Option 2: Hugging Face Spaces (FREE) ⭐

#### Setup:
1. Create space di https://huggingface.co/spaces
2. Select "Docker"
3. Upload project files:
   - `app.py`
   - `maintenance_guide.py`
   - `inference_helper.py`
   - `requirements.txt`
   - `Dockerfile`
4. Spaces akan auto-deploy

#### URL Publik:
`https://huggingface.co/spaces/your-username/solar-panel-monitor`

**Keuntungan:**
- ✅ FREE
- ✅ GPU available (free tier limited)
- ✅ Storage untuk model files
- ✅ Integrated with HF ecosystem

---

### Option 3: Railway.app ($5/month)

#### Setup:
1. Push ke GitHub
2. Signup di https://railway.app
3. Connect GitHub repo
4. Set environment variables:
   ```
   PORT=8501
   ```
5. Deploy

**URL:** `https://your-app-name.railway.app`

---

### Option 4: Render ($7/month untuk Starter)

#### Setup:
1. Create `Dockerfile`:
```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501"]
```

2. Signup di https://render.com
3. New → Web Service
4. Connect GitHub
5. Build command: `pip install -r requirements.txt`
6. Start command: `streamlit run app.py --server.port=8501`

---

## 📦 Docker Deployment (Untuk Production)

### 1. Create Dockerfile:
```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create non-root user for security
RUN useradd -m streamlit_user
USER streamlit_user

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### 2. Build & Run:
```bash
# Build
docker build -t solar-panel-monitor:latest .

# Run locally
docker run -p 8501:8501 solar-panel-monitor:latest

# Run dengan volume mount (untuk development)
docker run -p 8501:8501 -v $(pwd):/app solar-panel-monitor:latest
```

### 3. Push ke Docker Hub:
```bash
docker tag solar-panel-monitor:latest username/solar-panel-monitor:latest
docker push username/solar-panel-monitor:latest
```

---

## 🚀 Production Deployment Stack

### Architecture Rekomendasi:

```
┌─────────────────────────────────────────┐
│         Users (Web Browser)             │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│  Reverse Proxy (Nginx)                  │
│  - Load balancing                       │
│  - SSL/HTTPS termination                │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│  Streamlit App Instances (2-3)          │
│  - app.py running                       │
│  - Auto-restart on failure              │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│  Model Storage (Shared)                 │
│  - /models/best_solar_panel_classifier  │
│  - Mounted NFS/Volume                   │
└─────────────────────────────────────────┘
```

### Deployment Tools:
- **Kubernetes** (untuk enterprise, auto-scaling)
- **Docker Compose** (untuk simple multi-service)
- **PM2** (untuk process management di Linux)

---

## 🔧 Advanced: API + Web Separate

Jika ingin scalability lebih baik, pisah backend & frontend:

### Backend (FastAPI):
```python
# api.py
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from inference_helper import SolarPanelInference
import numpy as np

app = FastAPI(title="Solar Panel API")

# CORS untuk frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = SolarPanelInference()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(BytesIO(contents))
    result = model.predict(image)
    return result

@app.get("/health")
def health():
    return {"status": "ok"}
```

**Run:**
```bash
pip install fastapi uvicorn python-multipart
uvicorn api:app --host 0.0.0.0 --port 8000
```

### Frontend (Streamlit calling API):
```python
import requests

API_URL = "http://localhost:8000"

def predict_with_api(image):
    files = {"file": image}
    response = requests.post(f"{API_URL}/predict", files=files)
    return response.json()

result = predict_with_api(uploaded_file)
```

---

## 📊 Monitoring & Logging

### Untuk Production:
```python
import logging
from pythonjsonlogger import jsonlogger

# Setup structured logging
logger = logging.getLogger()
logHandler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter()
logHandler.setFormatter(formatter)
logger.addHandler(logHandler)
logger.setLevel(logging.INFO)

# Log predictions
logger.info("prediction", extra={
    "class": predicted_class,
    "confidence": confidence,
    "timestamp": datetime.now().isoformat()
})
```

### Tools untuk Monitoring:
- **Prometheus** + **Grafana** (metrics & dashboards)
- **ELK Stack** (logging)
- **Sentry** (error tracking)

---

## 🔐 Security Checklist

- [ ] Set `PYTHONHASHSEED` untuk reproducibility
- [ ] Use environment variables untuk sensitive data
- [ ] Enable HTTPS/SSL
- [ ] Rate limiting di reverse proxy
- [ ] Input validation (image size, format)
- [ ] Model versioning & rollback capability
- [ ] Regular backups dari predictions history
- [ ] Monitor resource usage (CPU, Memory, GPU)

---

## 📈 Scaling Strategy

### Tahap 1: MVP (Awal)
- ✅ Streamlit Cloud (FREE)
- ✅ Model di lokal
- ✅ Manual monitoring

### Tahap 2: Growth
- ✅ Self-hosted (Railway/Render)
- ✅ Caching untuk model
- ✅ Database untuk history
- ✅ Basic monitoring

### Tahap 3: Enterprise
- ✅ Kubernetes cluster
- ✅ Load balancing
- ✅ Multi-region deployment
- ✅ Advanced monitoring & alerts
- ✅ Separate API servers

---

## 🐛 Troubleshooting

### Error: "Out of memory"
**Solution:** 
```python
@st.cache_resource
def load_model():
    # Force CPU untuk reduce memory
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    return SolarPanelInference()
```

### Error: "Model not found"
**Solution:** Pastikan path absolut correct:
```python
MODEL_PATH = Path(__file__).parent / "models/saved_models/best_solar_panel_classifier.pt"
```

### Slow inference
**Solution:**
- Use ONNX optimization
- Batch processing
- Model quantization

---

## 📞 Support & Documentation

- Streamlit Docs: https://docs.streamlit.io
- Ultralytics YOLO: https://docs.ultralytics.com
- FastAPI: https://fastapi.tiangolo.com
- Docker: https://docs.docker.com

---

**Status:** Production-ready untuk deployment! 🚀
