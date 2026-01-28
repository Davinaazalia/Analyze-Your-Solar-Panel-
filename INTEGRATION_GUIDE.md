# 🌞 Solar Panel YOLO Integration Guide

## 📋 Ringkasan Integrasi

Sistem ini mengintegrasikan model YOLO classification untuk prediksi kondisi solar panel dengan web interface. Terdiri dari:

1. **Backend API Server** (`api_server.py`) - REST API yang menjalankan model YOLO
2. **Frontend JavaScript** (`panel-predictor.js`) - Client untuk berkomunikasi dengan API
3. **Frontend CSS** (`panel-styles.css`) - Styling untuk hasil prediksi
4. **HTML Interface** (`index.html`) - Web interface untuk user

---

## 🚀 Quick Start

### Step 1: Install Dependencies

```bash
pip install flask flask-cors pillow ultralytics torch numpy
```

### Step 2: Pastikan Model Tersedia

Model harus ada di: `models/saved_models/best_solar_panel_classifier.pt`

```
d:\solar-panel-fault-detection\
├── models/
│   └── saved_models/
│       └── best_solar_panel_classifier.pt
└── api_server.py
```

### Step 3: Start API Server

**Windows:**
```powershell
python api_server.py
```

**Linux/Mac:**
```bash
python3 api_server.py
```

Server akan berjalan di: `http://127.0.0.1:5000`

### Step 4: Integrasikan dengan HTML

Tambahkan ke file `index.html` (di dalam `<head>`):

```html
<!-- CSS untuk styling hasil prediksi -->
<link rel="stylesheet" href="./panel-styles.css">
```

Tambahkan di akhir `<body>`:

```html
<!-- JavaScript untuk integrasi API -->
<script src="./panel-predictor.js"></script>
```

---

## 📚 API Endpoints

### 1. Health Check
```
GET /
```
Response:
```json
{
    "status": "healthy",
    "model_loaded": true,
    "service": "SolarEye Panel Detection API"
}
```

### 2. Predict Image
```
POST /api/predict
Content-Type: multipart/form-data
```

**Request:**
```
Form Data:
  image: <file>
```

**Response (Success):**
```json
{
    "success": true,
    "prediction": {
        "class": "Dusty",
        "confidence": 95.67,
        "class_idx": 2
    },
    "all_probabilities": {
        "Bird-drop": 2.3,
        "Clean": 0.5,
        "Dusty": 95.67,
        "Electrical-damage": 0.8,
        "Physical-Damage": 0.5,
        "Snow-Covered": 0.2
    },
    "top5": [
        {"class": "Dusty", "confidence": 95.67},
        {"class": "Bird-drop", "confidence": 2.3},
        {"class": "Electrical-damage", "confidence": 0.8},
        {"class": "Physical-Damage", "confidence": 0.5},
        {"class": "Clean", "confidence": 0.5}
    ],
    "info": {
        "status": "WARNING",
        "color": "#f39c12",
        "description": "Panel tertutup debu/kotoran",
        "urgency": "Medium",
        "maintenance": "Bersihkan panel segera untuk efisiensi maksimal",
        "risk": "Moderate - Mengurangi output hingga 25%"
    },
    "image": "data:image/jpeg;base64,..."
}
```

### 3. Get Model Info
```
GET /api/model-info
```
Response:
```json
{
    "model_type": "YOLOv8s-Classification",
    "classes": ["Bird-drop", "Clean", "Dusty", "Electrical-damage", "Physical-Damage", "Snow-Covered"],
    "num_classes": 6,
    "device": "cuda",
    "input_size": 224,
    "accuracy": "98.06%",
    "model_size": "10.2 MB"
}
```

### 4. Get Classes Info
```
GET /api/classes
```

### 5. Batch Predict
```
POST /api/batch-predict
Content-Type: application/json
```

**Request:**
```json
{
    "images": [
        "data:image/jpeg;base64,...",
        "data:image/jpeg;base64,..."
    ]
}
```

---

## 🎨 HTML Integration Example

### Minimal Setup

```html
<!DOCTYPE html>
<html>
<head>
    <link rel="stylesheet" href="./panel-styles.css">
</head>
<body>
    <!-- Upload Section -->
    <section>
        <h1>Solar Panel Analyzer</h1>
        
        <!-- Drop Zone -->
        <div id="dropZone" data-drop-zone>
            <p>Drag & drop image atau klik untuk upload</p>
            <input type="file" id="imageInput" data-file-input accept="image/*" style="display:none">
        </div>
        
        <!-- Image Preview -->
        <img id="imagePreview" data-image-preview>
        
        <!-- Loading Spinner -->
        <div id="loadingSpinner" data-loading>
            <div class="loading-spinner"></div>
            <p class="loading-text">Analyzing panel...</p>
        </div>
        
        <!-- Result Container -->
        <div id="resultContainer" data-result-container></div>
    </section>
    
    <!-- Include Script -->
    <script src="./panel-predictor.js"></script>
</body>
</html>
```

### Advanced Setup dengan Custom Events

```html
<!DOCTYPE html>
<html>
<head>
    <link rel="stylesheet" href="./panel-styles.css">
</head>
<body>
    <div id="app">
        <input type="file" id="imageFile" accept="image/*">
        <button id="analyzeBtn">Analyze</button>
        <div id="results"></div>
    </div>

    <script src="./panel-predictor.js"></script>
    <script>
        // Customize API URL jika berbeda
        panelClient = new SolarPanelClient('http://your-api-url:5000');
        
        // Custom event handler
        document.getElementById('analyzeBtn').addEventListener('click', async () => {
            const file = document.getElementById('imageFile').files[0];
            if (!file) return;
            
            const result = await panelClient.predictImage(file);
            
            if (result.success) {
                const pred = result.prediction;
                document.getElementById('results').innerHTML = `
                    <h2>${pred.class}</h2>
                    <p>Confidence: ${pred.confidence}%</p>
                    <p>${result.info.description}</p>
                `;
            }
        });
    </script>
</body>
</html>
```

---

## 🔧 JavaScript API Reference

### Constructor
```javascript
const client = new SolarPanelClient(apiUrl);
// Default: http://127.0.0.1:5000
```

### Methods

#### 1. `predictImage(file)`
Predict dari File object

```javascript
const file = document.getElementById('imageInput').files[0];
const result = await client.predictImage(file);
```

#### 2. `predictBase64(base64String)`
Predict dari base64 string

```javascript
const result = await client.predictBase64(base64Image);
```

#### 3. `getModelInfo()`
Get informasi model

```javascript
const info = await client.getModelInfo();
console.log(info.classes);
```

#### 4. `getClasses()`
Get list classes dengan info

```javascript
const classes = await client.getClasses();
```

#### 5. `checkHealth()`
Check API connection

```javascript
const health = await client.checkHealth();
if (health.status === 'healthy') {
    console.log('API is running');
}
```

---

## 📊 Class Predictions Explanation

### Classes & Status

| Class | Status | Urgency | Meaning |
|-------|--------|---------|---------|
| **Clean** | ✅ GOOD | Low | Panel bersih, performa optimal |
| **Dusty** | ⚠️ WARNING | Medium | Tertutup debu, bersihkan segera |
| **Bird-drop** | ⚠️ WARNING | High | Kotoran burung, risiko hot spot |
| **Snow-Covered** | ❌ CRITICAL | Critical | Tertutup salju, output rendah |
| **Electrical-damage** | ❌ CRITICAL | Critical | Kerusakan elektrik, risiko keselamatan |
| **Physical-Damage** | ❌ CRITICAL | Critical | Kerusakan fisik, butuh penggantian |

---

## 🛠️ Troubleshooting

### API tidak respond

1. **Pastikan API berjalan:**
   ```bash
   curl http://127.0.0.1:5000/
   ```

2. **Check port availability:**
   ```bash
   netstat -an | findstr 5000  # Windows
   lsof -i :5000               # Mac/Linux
   ```

3. **Restart server:**
   ```bash
   python api_server.py
   ```

### CORS Error

Jika error CORS (Cross-Origin):
- Server sudah menggunakan `CORS` enabled
- Pastikan request dari URL yang benar
- Untuk production, configure CORS properly

### Model tidak load

1. **Check path model:**
   ```bash
   ls models/saved_models/
   ```

2. **Verify model file:**
   ```
   Best_solar_panel_classifier.pt harus exist
   ```

3. **Reinstall dependencies:**
   ```bash
   pip install --upgrade ultralytics torch
   ```

### Memory Issues

Jika memory penuh:
1. Reduce batch size
2. Clear temp files (`temp_*.jpg`)
3. Restart server

---

## 🚀 Production Deployment

### Option 1: Gunicorn + Nginx

```bash
pip install gunicorn

# Run with Gunicorn
gunicorn --workers 4 --bind 0.0.0.0:5000 api_server:app
```

### Option 2: Docker

Create `Dockerfile`:
```dockerfile
FROM python:3.10
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "api_server.py"]
```

Build & run:
```bash
docker build -t solareye-api .
docker run -p 5000:5000 solareye-api
```

### Option 3: modify `api_server.py`

```python
if __name__ == '__main__':
    # Change to production settings
    app.run(debug=False, host='0.0.0.0', port=5000)
```

---

## 📝 Example Use Cases

### 1. Single Image Analysis
```javascript
// Upload form
const file = document.getElementById('imageInput').files[0];
const result = await panelClient.predictImage(file);

if (result.success) {
    // Display hasil
    console.log(`Class: ${result.prediction.class}`);
    console.log(`Confidence: ${result.prediction.confidence}%`);
    console.log(`Action: ${result.info.maintenance}`);
}
```

### 2. Multiple Panels Monitoring
```javascript
// Batch analysis
const images = [
    base64Image1,
    base64Image2,
    base64Image3
];

// Note: Current implementation requires custom batch endpoint
// Atau gunakan loop
for (const img of images) {
    const result = await panelClient.predictBase64(img);
    console.log(result);
}
```

### 3. Real-time Dashboard
```javascript
// Check API & get latest model info
const health = await panelClient.checkHealth();
const modelInfo = await panelClient.getModelInfo();

// Display in dashboard
document.getElementById('apiStatus').textContent = health.status;
document.getElementById('modelVersion').textContent = modelInfo.model_type;
```

---

## ✅ Checklist Integrasi

- [ ] API server running pada port 5000
- [ ] Model file ada di lokasi yang benar
- [ ] Dependencies terinstall (Flask, CORS, Pillow, ultralytics)
- [ ] HTML include `panel-predictor.js`
- [ ] HTML include `panel-styles.css`
- [ ] Drop zone atau file input tersedia
- [ ] Result container tersedia
- [ ] Test upload image → check browser console
- [ ] Hasil prediksi ditampilkan

---

## 📞 Support

Untuk bantuan lebih lanjut:
1. Check console browser untuk error messages
2. Check server logs untuk API errors
3. Verify file paths dan model availability
4. Test API directly dengan curl atau Postman

---

**Happy analyzing! 🌞**
