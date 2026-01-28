# 🚀 Complete Deployment Guide - Frontend & Backend

## 📋 Overview
- **Frontend (HTML/CSS/JS)**: Deploy ke Vercel
- **Backend (Flask API)**: Deploy ke Railway
- **Total waktu**: ~10 menit

---

## 🎯 STEP 1: Deploy Backend ke Railway

### 1.1 Sign Up Railway
1. Buka https://railway.app
2. Sign up dengan GitHub account
3. Authorize Railway to access your repos

### 1.2 Create New Project
1. Klik **"New Project"**
2. Pilih **"Deploy from GitHub repo"**
3. Pilih repository: `Analyze-Your-Solar-Panel-`
4. Klik repository tersebut

### 1.3 Configure (Otomatis!)
Railway akan otomatis detect:
- ✅ Python project
- ✅ requirements.txt
- ✅ Procfile (untuk run command)
- ✅ Runtime (Python 3.11)

### 1.4 Wait for Build (~5 menit)
Railway akan:
1. Install dependencies
2. Download model weights
3. Start Flask server

### 1.5 Get API URL
1. Setelah deploy sukses, klik tab **"Settings"**
2. Scroll ke **"Networking"**
3. Klik **"Generate Domain"**
4. Copy URL (contoh: `https://analyze-your-solar-panel-production.up.railway.app`)

**⚠️ SAVE URL INI! Kita perlu untuk frontend**

---

## 🌐 STEP 2: Update Frontend dengan Backend URL

### 2.1 Buka index.html
File: `Web_Implementation/index.html`

### 2.2 Cari baris ~471 (fetch API)
```javascript
const response = await fetch('http://127.0.0.1:5000/api/predict', {
```

### 2.3 Ganti dengan Railway URL
```javascript
const response = await fetch('https://YOUR-RAILWAY-URL.railway.app/api/predict', {
```

**Ganti `YOUR-RAILWAY-URL` dengan URL dari Railway!**

### 2.4 Commit & Push
```bash
git add Web_Implementation/index.html
git commit -m "Update: Connect to Railway backend API"
git push origin main
```

---

## 🎨 STEP 3: Deploy Frontend ke Vercel

### 3.1 Buka Vercel
1. Go to https://vercel.com
2. Login dengan GitHub

### 3.2 Import Project
1. Klik **"Add New..."** → **"Project"**
2. Cari repository: `Analyze-Your-Solar-Panel-`
3. Klik **"Import"**

### 3.3 Configure Project
```
Framework Preset: Other
Root Directory: Web_Implementation
Build Command: (kosongkan)
Output Directory: (kosongkan)
Install Command: (kosongkan)
```

### 3.4 Deploy!
1. Klik **"Deploy"**
2. Tunggu ~2 menit
3. Done! ✅

### 3.5 Get Frontend URL
Vercel akan generate URL (contoh: `https://analyze-your-solar-panel.vercel.app`)

---

## ✅ STEP 4: Test Complete System

### 4.1 Buka Frontend URL
`https://your-project.vercel.app`

### 4.2 Test Upload Image
1. Klik upload zone
2. Pilih gambar solar panel
3. Wait for prediction
4. Check results!

### 4.3 Check Network
1. Buka DevTools (F12)
2. Tab Network
3. Upload image
4. Lihat request ke Railway API
5. Status 200 = SUCCESS ✅

---

## 🔧 Troubleshooting

### Problem 1: "Cannot connect to API"
**Solusi:**
- Check Railway deployment status (harus "Active")
- Verify URL di index.html benar
- Check Railway logs: Settings → Deployments → View Logs

### Problem 2: "Model not loaded"
**Solusi:**
- Railway perlu download model (45MB)
- Wait 5-10 menit untuk first deployment
- Check logs untuk error messages

### Problem 3: CORS Error
**Solusi:**
- Sudah di-handle di `api_server.py` (CORS enabled)
- Tapi kalau masih error, check Railway environment variables
- Add: `FLASK_ENV=production`

### Problem 4: Railway deploy failed
**Solusi:**
- Check requirements.txt complete
- Verify Procfile exists: `web: python api_server.py`
- Check runtime.txt: `python-3.11.0`

---

## 📊 Architecture Diagram

```
User Browser
    ↓
Vercel (Frontend HTML/CSS/JS)
    ↓ API Request
Railway (Backend Flask + YOLO)
    ↓ Response
Vercel (Display Results)
    ↓
User sees prediction!
```

---

## 🎯 Quick Commands Cheat Sheet

**Push to GitHub:**
```bash
git add .
git commit -m "Deploy: Ready for production"
git push origin main
```

**Test Local Backend:**
```bash
python api_server.py
# Visit: http://localhost:5000
```

**Test Local Frontend:**
```bash
# Just open index.html in browser
start Web_Implementation/index.html
```

---

## 🌟 Final URLs

After deployment, you'll have:

**Frontend (Vercel):**
`https://analyze-your-solar-panel.vercel.app`

**Backend (Railway):**
`https://your-project.railway.app`

**API Endpoint:**
`https://your-project.railway.app/api/predict`

---

## 📝 Notes

1. **Railway Free Tier:**
   - 500 jam/bulan (cukup!)
   - 1GB RAM
   - 1GB disk
   - Perfect untuk project ini

2. **Vercel Free Tier:**
   - Unlimited deployments
   - 100GB bandwidth/bulan
   - Auto SSL/HTTPS
   - CDN worldwide

3. **Auto Deploy:**
   - Setiap `git push` → otomatis deploy
   - Railway: backend update
   - Vercel: frontend update

---

## 🎉 You're Done!

Your full-stack AI solar panel detection system is now LIVE on the internet! 🚀

Share your project:
- Frontend URL → For users
- GitHub Repo → For portfolio
- Railway Dashboard → For monitoring

---

**Created by: Davina Azalia Tara**
**Date: January 28, 2026**
