# 🌞 SolarEye - Integration Quick Start

## ⚡ 3 Langkah Integrasi

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 2️⃣ Start API Server
```bash
python api_server.py
```
Server akan berjalan di: **http://127.0.0.1:5000**

### 3️⃣ Buka HTML di Browser
Gunakan file: `Web_Implementation/index-integrated.html`

**Atau** edit `Web_Implementation/index.html` existing Anda:

Tambahkan di `<head>`:
```html
<link rel="stylesheet" href="./panel-styles.css">
```

Tambahkan di akhir `<body>`:
```html
<script src="./panel-predictor.js"></script>
```

---

## 📁 File yang Dibuat

```
d:\solar-panel-fault-detection\
├── api_server.py                  # ← Backend API Server (JALANKAN INI)
├── INTEGRATION_GUIDE.md           # ← Dokumentasi lengkap
├── requirements.txt               # ← Updated dengan Flask & CORS
└── Web_Implementation\
    ├── panel-predictor.js         # ← JavaScript client (NEW)
    ├── panel-styles.css           # ← CSS styling (NEW)
    ├── index.html                 # ← Original (update sesuai petunjuk)
    └── index-integrated.html      # ← Template siap pakai (NEW)
```

---

## 🚀 Cara Kerja

```
User Upload Image
    ↓
HTML Form (panel-predictor.js)
    ↓
API Request → api_server.py
    ↓
YOLO Model Prediction
    ↓
JSON Response
    ↓
Display Result in HTML
```

---

## 📊 Hasil Prediksi

Setiap prediksi menampilkan:

- ✅ **Class**: Kondisi panel (Clean, Dusty, Bird-drop, etc)
- 📊 **Confidence**: Tingkat kepercayaan (%)
- 🎯 **Status**: GOOD / WARNING / CRITICAL
- 💡 **Recommendation**: Aksi yang direkomendasikan
- ⚠️ **Risk Level**: Tingkat risiko
- 📈 **All Probabilities**: Grafik semua kelas

---

## 🔍 Testing

### Test API dengan curl:
```bash
curl -X POST -F "image=@test.jpg" http://127.0.0.1:5000/api/predict
```

### Check API status:
```bash
curl http://127.0.0.1:5000/
```

---

## ⚙️ Konfigurasi

### Ubah API URL (jika berbeda)

Di file `panel-predictor.js`, edit:
```javascript
let panelClient = new SolarPanelClient('http://your-api-url:5000');
```

Atau di `index.html`:
```javascript
<script>
    window.API_URL = 'http://your-api-url:5000';
</script>
```

### Ubah Confidence Threshold

Di `api_server.py`:
```python
result = model.predict(temp_path, conf=0.5)  # Default 0.25
```

---

## 🛠️ Troubleshooting

| Problem | Solution |
|---------|----------|
| "API tidak terhubung" | Pastikan `python api_server.py` berjalan |
| "Model not loaded" | Check file ada di `models/saved_models/best_solar_panel_classifier.pt` |
| CORS Error | Pastikan Flask sudah run, bukan issue |
| Memory Error | Restart server, clear temp files |

---

## 📝 Struktur Response API

```json
{
    "success": true,
    "prediction": {
        "class": "Dusty",
        "confidence": 95.67,
        "class_idx": 2
    },
    "info": {
        "status": "WARNING",
        "color": "#f39c12",
        "description": "Panel tertutup debu/kotoran",
        "urgency": "Medium",
        "maintenance": "Bersihkan panel segera...",
        "risk": "Moderate - Mengurangi output hingga 25%"
    },
    "all_probabilities": {
        "Clean": 0.5,
        "Dusty": 95.67,
        ...
    }
}
```

---

## 📚 Dokumentasi Lengkap

Lihat: **INTEGRATION_GUIDE.md** untuk:
- API endpoints lengkap
- JavaScript API reference
- Production deployment
- Advanced features
- Complete troubleshooting

---

## 🎯 Next Steps

1. ✅ Run API server
2. ✅ Test dengan HTML
3. ✅ Customize styling sesuai kebutuhan
4. ✅ Deploy ke production (lihat INTEGRATION_GUIDE.md)

---

**Questions? Check INTEGRATION_GUIDE.md atau console browser untuk error details!** 🌞
