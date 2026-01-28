# 🌞 SolarEye Integration Cheat Sheet

## Quick Reference Guide

### 🚀 Start Everything in 3 Commands

```bash
# Terminal 1: Install dependencies (run once)
pip install -r requirements.txt

# Terminal 2: Run API Server (always keep running)
python api_server.py

# Terminal 3: Test Integration
python test_integration.py

# Then open browser:
# Web_Implementation/index-integrated.html
```

---

## 📝 HTML Integration Template

### Minimal Working Example

```html
<!DOCTYPE html>
<html>
<head>
    <link rel="stylesheet" href="./panel-styles.css">
</head>
<body>
    <!-- Upload Area -->
    <div id="uploadZone" data-drop-zone>
        <p>Drop image here or click</p>
        <input type="file" id="imageInput" data-file-input accept="image/*" style="display:none">
    </div>
    
    <!-- Preview -->
    <img id="imagePreview" data-image-preview>
    
    <!-- Loading -->
    <div id="loadingSpinner" data-loading>
        <div class="spinner"></div>
    </div>
    
    <!-- Results -->
    <div id="resultContainer" data-result-container></div>
    
    <!-- Script -->
    <script src="./panel-predictor.js"></script>
</body>
</html>
```

---

## 🔌 API Endpoints Reference

```bash
# Health Check
curl http://127.0.0.1:5000/

# Get Model Info
curl http://127.0.0.1:5000/api/model-info

# Get Classes
curl http://127.0.0.1:5000/api/classes

# Predict Image
curl -X POST -F "image=@photo.jpg" http://127.0.0.1:5000/api/predict
```

---

## 💻 JavaScript API Reference

```javascript
// Initialize client
const client = new SolarPanelClient();

// Predict from file
const file = document.getElementById('imageInput').files[0];
const result = await client.predictImage(file);

// Predict from base64
const result = await client.predictBase64(base64String);

// Get model info
const info = await client.getModelInfo();

// Check health
const health = await client.checkHealth();

// Result structure:
{
    success: true,
    prediction: {
        class: "Dusty",
        confidence: 95.67,
        class_idx: 2
    },
    info: {
        status: "WARNING",
        color: "#f39c12",
        description: "...",
        urgency: "Medium",
        maintenance: "...",
        risk: "..."
    },
    all_probabilities: { /* ... */ },
    image: "data:image/jpeg;base64,..."
}
```

---

## 🎨 CSS Classes Reference

```css
/* Result card styling */
.result-card { }
.result-header { }
.status-badge { }
.urgency-badge { }

/* Content sections */
.predicted-class { }
.confidence-text { }
.description { }
.risk-section { }
.maintenance-section { }
.probabilities-section { }

/* Lists & bars */
.probabilities-list li { }
.probability-bar { }
.probability-fill { }

/* States */
[data-loading] { }
[data-result-container] { }
[data-image-preview] { }
[data-drop-zone] { }
```

---

## 🔧 Common Customizations

### Change API URL
```javascript
// In panel-predictor.js or custom script:
const client = new SolarPanelClient('http://your-server:5000');
```

### Change Confidence Threshold
```python
# In api_server.py:
result = model.predict(temp_path, conf=0.5)  # Default: 0.25
```

### Add Custom Status Colors
```python
# In api_server.py, CLASS_INFO dict:
'MyClass': {
    'status': 'CUSTOM',
    'color': '#FF5733',  # Your color
    'description': '...',
    # ...
}
```

### Custom Result Display
```javascript
// Override displayPredictionResult() in panel-predictor.js:
function displayPredictionResult(result) {
    // Your custom HTML generation
    const html = `<div>Custom result: ${result.prediction.class}</div>`;
    document.getElementById('resultContainer').innerHTML = html;
}
```

---

## 📊 Class Status Colors

| Class | Status | Color | Use Case |
|-------|--------|-------|----------|
| Clean | ✅ GOOD | #27ae60 (Green) | Optimal condition |
| Dusty | ⚠️ WARNING | #f39c12 (Orange) | Needs cleaning |
| Bird-drop | ⚠️ WARNING | #f39c12 (Orange) | Urgent cleaning |
| Snow-Covered | ❌ CRITICAL | #e74c3c (Red) | Remove immediately |
| Electrical-damage | ❌ CRITICAL | #e74c3c (Red) | Safety risk |
| Physical-Damage | ❌ CRITICAL | #e74c3c (Red) | Needs repair |

---

## 🐛 Debugging

### Browser Console (DevTools - F12)
```javascript
// Check if client initialized
console.log(panelClient);

// Test prediction
const testFile = document.getElementById('imageInput').files[0];
panelClient.predictImage(testFile).then(r => console.log(r));

// Check API
panelClient.checkHealth().then(h => console.log(h));
```

### Server Logs
```bash
# Watch API server output
# Should show:
# * Running on http://127.0.0.1:5000
# * Model loaded successfully
```

### Test API with cURL
```bash
# Windows PowerShell
$file = Get-Item "photo.jpg"
$form = @{'image' = $file}
Invoke-RestMethod -Uri "http://127.0.0.1:5000/api/predict" -Method POST -Form $form

# Linux/Mac
curl -X POST -F "image=@photo.jpg" http://127.0.0.1:5000/api/predict | jq .
```

---

## ⚡ Performance Tips

1. **Compress images** before upload (< 2MB recommended)
2. **Cache model** in memory (already done in api_server.py)
3. **Use CORS proxy** for production
4. **Enable caching** headers in Flask for static files
5. **Use async/await** in JavaScript for better UX

---

## 📦 File Structure

```
d:\solar-panel-fault-detection\
├── api_server.py                    ← Start this first
├── inference_helper.py              ← Model loading logic
├── test_integration.py              ← Test script
├── setup-windows.ps1                ← Setup automation
├── requirements.txt                 ← Dependencies (WITH Flask!)
├── QUICK_START.md                   ← Quick guide
├── INTEGRATION_GUIDE.md             ← Complete docs
│
├── models/
│   └── saved_models/
│       └── best_solar_panel_classifier.pt  ← Model file (required)
│
└── Web_Implementation/
    ├── index.html                   ← Existing (add imports)
    ├── index-integrated.html        ← Template (ready to use)
    ├── panel-predictor.js           ← Client library (NEW)
    ├── panel-styles.css             ← Styling (NEW)
    └── images/
        └── [your images]
```

---

## ✅ Validation Checklist

- [ ] `python api_server.py` runs without errors
- [ ] `curl http://127.0.0.1:5000/` returns JSON
- [ ] Model file exists: `models/saved_models/best_solar_panel_classifier.pt`
- [ ] HTML has: `<link rel="stylesheet" href="./panel-styles.css">`
- [ ] HTML has: `<script src="./panel-predictor.js"></script>`
- [ ] HTML has: `id="uploadZone"` and `id="imageInput"`
- [ ] HTML has: `id="resultContainer"` and `id="loadingSpinner"`
- [ ] Upload image works in HTML
- [ ] Results display correctly
- [ ] Browser console has no errors

---

## 🚨 Common Errors & Solutions

| Error | Cause | Fix |
|-------|-------|-----|
| `Cannot connect to API` | API not running | `python api_server.py` |
| `Model not loaded` | File path wrong | Check model path in api_server.py |
| `CORS error` | Wrong origin | Flask-CORS already configured |
| `Image not sent` | Form issue | Check form element id & file input |
| `No results shown` | JS error | Check browser console (F12) |
| `Prediction timeout` | Slow model | Model should process in 1-2 seconds |

---

## 🎯 Next Steps After Integration

1. **Customize styling** to match your brand
2. **Add history tracking** to store predictions
3. **Implement database** to save results
4. **Add export feature** (PDF, CSV)
5. **Deploy to server** (Heroku, AWS, etc)
6. **Add user authentication** if needed
7. **Create admin dashboard** for monitoring

---

## 📞 Quick Help

- **API not responding?** → Check if running + firewall
- **Model loading slow?** → First run is slower, subsequent are cached
- **JavaScript errors?** → Open DevTools (F12), check Console tab
- **Integration not working?** → Run `python test_integration.py`
- **Want custom colors?** → Edit `CLASS_INFO` in api_server.py
- **Need batch processing?** → See `/api/batch-predict` endpoint

---

**Last Updated:** January 28, 2026
**Version:** 1.0
**Status:** ✅ Production Ready
