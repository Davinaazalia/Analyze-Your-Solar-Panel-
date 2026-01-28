# 📚 Step-by-Step Guide: GitHub & Vercel Deployment

## 📋 Prerequisites
- GitHub account (free di https://github.com)
- Vercel account (free di https://vercel.com)
- Git installed (https://git-scm.com)

---

## 🎯 STEP 1: Setup Git Local

### 1.1 Configure Git (if first time)
```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@gmail.com"
```

### 1.2 Initialize Repository
```bash
cd d:\solar-panel-fault-detection
git init
```

### 1.3 Add All Files
```bash
git add .
```

### 1.4 Create First Commit
```bash
git commit -m "Initial commit: SolarEye - Solar Panel Fault Detection System"
```

Lihat status:
```bash
git status
```

---

## 🚀 STEP 2: Create GitHub Repository

### 2.1 Go to GitHub
1. Buka https://github.com
2. Login ke akun kamu
3. Klik **"+"** di top-right → **"New repository"**

### 2.2 Create Repo
```
Repository name: solar-panel-fault-detection
Description: AI-powered solar panel condition analysis using YOLOv8
Visibility: Public (agar orang bisa lihat)
Initialize: NO (jangan check ini)
```

### 2.3 Klik "Create repository"

---

## 🔗 STEP 3: Connect Local ke GitHub

### 3.1 Setelah buat repo, GitHub akan tunjukin command:
```bash
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/solar-panel-fault-detection.git
git push -u origin main
```

**Ganti `YOUR_USERNAME` dengan username GitHub kamu!**

### 3.2 Copy & paste 3 command tadi di terminal

**Contoh:**
```bash
# Terminal di folder solar-panel-fault-detection

git branch -M main
git remote add origin https://github.com/johndoe/solar-panel-fault-detection.git
git push -u origin main

# Akan diminta login GitHub - gunakan token atau password
```

### 3.3 Tunggu push selesai ✅

---

## 📡 STEP 4: Verify di GitHub

1. Buka https://github.com/YOUR_USERNAME/solar-panel-fault-detection
2. Harus bisa lihat semua file sudah tercopy
3. Klik **"Code"** untuk verify

---

## 🌐 STEP 5: Setup Backend API

### ⚠️ **PENTING**: Backend (api_server.py) perlu di-deploy terpisah!

**Opsi A: Railway (RECOMMENDED - Gratis)**

1. Buka https://railway.app
2. Login / Sign up dengan GitHub
3. Klik "New Project" → "Deploy from GitHub repo"
4. Pilih `solar-panel-fault-detection`
5. Railway otomatis detect Python
6. Set environment: `FLASK_ENV=production`
7. Railway akan generate URL (contoh: `https://api-xyz.railway.app`)
8. **SAVE URL INI** - kita perlukan untuk frontend

**Opsi B: Heroku (Alternative)**
```bash
heroku login
heroku create your-app-name
heroku config:set FLASK_ENV=production
git push heroku main
```

### Setelah Backend Deployed:
- Simpen API URL (misal: `https://api-xyz.railway.app`)
- Update di `Web_Implementation/index.html` - ubah:
  ```javascript
  const response = await fetch('http://127.0.0.1:5000/api/predict', {
  ```
  Menjadi:
  ```javascript
  const response = await fetch('https://api-xyz.railway.app/api/predict', {
  ```

---

## 🚀 STEP 6: Deploy Frontend ke Vercel

### 6.1 Buka Vercel
1. Buka https://vercel.com
2. Login dengan GitHub

### 6.2 Import Project
1. Klik "New Project"
2. Klik "Import Git Repository"
3. Paste: `https://github.com/YOUR_USERNAME/solar-panel-fault-detection`
4. Klik "Import"

### 6.3 Configure Project
```
Framework: Other (untuk static site)
Root Directory: Web_Implementation
Environment Variables: (kosongkan dulu)
```

### 6.4 Deploy!
Klik "Deploy" dan tunggu selesai (~2 menit)

Vercel akan generate URL (contoh: `https://your-project.vercel.app`)

---

## ✅ STEP 7: Test di Production

1. Buka URL dari Vercel di browser
2. Test upload image
3. Jika error "Cannot connect to API" → update URL di index.html
4. Push update:
   ```bash
   git add Web_Implementation/index.html
   git commit -m "Update API URL for production"
   git push origin main
   ```
5. Vercel otomatis re-deploy

---

## 📝 STEP 8: Update Kode (Ongoing)

### Setiap kali ada perubahan:
```bash
# 1. Ubah file sesuai kebutuhan
# 2. Check status
git status

# 3. Add changes
git add .

# 4. Commit dengan pesan deskriptif
git commit -m "Fix: improve loading spinner animation"

# 5. Push ke GitHub
git push origin main

# GitHub → Vercel otomatis re-deploy frontend
# GitHub → Railway otomatis re-deploy backend (jika setup)
```

---

## 🎨 Summary: Files yang Sudah Dibuat

✅ **.gitignore** - Ignore file besar & cache
✅ **README.md** - Project documentation  
✅ **vercel.json** - Vercel configuration
✅ **Web_Implementation/** - Frontend (siap deploy)
✅ **api_server.py** - Backend API

---

## 🚨 Common Issues & Solutions

### Error: "fatal: not a git repository"
```bash
# Solusi: cd ke folder yang benar
cd d:\solar-panel-fault-detection
```

### Error: "Authentication failed"
```bash
# Solusi 1: Use GitHub Token
# Di terminal: git config --global credential.helper store
# Paste token saat diminta

# Solusi 2: Setup SSH Key
# https://docs.github.com/en/authentication/connecting-to-github-with-ssh
```

### Error: "Cannot connect to API in production"
```bash
# Solusi: Update fetch URL di index.html
// FROM:
const response = await fetch('http://127.0.0.1:5000/api/predict', {

// TO:
const response = await fetch('https://your-backend-url.com/api/predict', {
```

### Vercel shows blank page
```bash
# Solusi: Root directory harus "Web_Implementation"
# Cek di Vercel Project Settings → General → Root Directory
```

---

## 🔗 Useful Links

- GitHub Docs: https://docs.github.com
- Vercel Docs: https://vercel.com/docs
- Railway Docs: https://docs.railway.app
- Git Cheatsheet: https://github.github.com/training-kit/downloads/github-git-cheat-sheet.pdf

---

## ✨ Final Checklist

- [ ] Git initialized & initial commit done
- [ ] GitHub repository created
- [ ] Code pushed to GitHub
- [ ] Backend API deployed (Railway/Heroku)
- [ ] Frontend deployed to Vercel
- [ ] API URL updated in frontend
- [ ] Test all features in production
- [ ] README updated with project info
- [ ] Share URL ke friends! 🎉

---

**Selesai! 🚀 Project kamu sudah live di internet!**
