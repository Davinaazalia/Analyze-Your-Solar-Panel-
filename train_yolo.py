#!/usr/bin/env python3
"""
Solar Panel Fault Detection - YOLO Training Script
Program untuk melatih AI model yang bisa detect kerusakan solar panel
"""

# ============================================
# BAGIAN 1: IMPORT LIBRARY (Tools yang kita pakai)
# ============================================
import os                      # Untuk operasi file/folder
import shutil                  # Untuk copy-paste file
import random                  # Untuk random shuffle data
from pathlib import Path       # Untuk kelola path file dengan mudah
import yaml                    # Format file untuk config (data.yaml)
from ultralytics import YOLO   # Library YOLO8 dari Ultralytics
import tensorflow as tf        # Deep Learning library (optional di sini)
import numpy as np             # Untuk math operations
import matplotlib.pyplot as plt # Untuk plot/visualisasi

print("=" * 60)
print("🚀 YOLO Training Pipeline - Solar Panel Fault Detection")
print("=" * 60)
# Penjelasan: 
# - YOLO = You Only Look Once (AI yang bisa detect object cepat)
# - Training = Mengajari AI pake data gambar solar panel
# - Detection = AI bisa lihat mana panel yang rusak

# ============ STEP 1: Prepare Dataset ============
# Penjelasan: YOLO butuh data dengan struktur tertentu
# Struktur folder yang diminta YOLO:
# yolo_dataset/
#   ├─ images/
#   │  ├─ train/    (70% gambar untuk training)
#   │  ├─ val/      (20% gambar untuk validation/testing)
#   │  └─ test/     (10% gambar untuk final test)
#   └─ labels/      (text file berisi anotasi - class dan bounding box)
#      ├─ train/
#      ├─ val/
#      └─ test/

print("\n[STEP 1] Preparing dataset in YOLO format...")

# Definisikan path folder
BASE_DIR = Path("./data")                    # Folder utama data
SOURCE_DATASET = BASE_DIR / "dataset"        # Folder asli dengan subfolder per kelas
YOLO_DATASET = BASE_DIR / "yolo_dataset"     # Folder output YOLO format

# Daftar kelas (jenis fault) yang kita detect
# Ini harus sama dengan folder yang ada di ./data/dataset/
CLASSES = ['Bird-drop', 'Clean', 'Dusty', 'Electrical-damage', 'Physical-Damage', 'Snow-Covered']

# Mapping kelas ke ID angka (YOLO butuh angka, bukan text)
# Misal: 'Bird-drop' = 0, 'Clean' = 1, dst
class_to_id = {cls: idx for idx, cls in enumerate(CLASSES)}
print("Classes mapping:")
for cls, idx in class_to_id.items():
    print(f"  {idx}: {cls}")

# Ratio pembagian data
TRAIN_RATIO = 0.7   # 70% data untuk training (belajar)
VAL_RATIO = 0.2     # 20% data untuk validation (test saat training)
TEST_RATIO = 0.1    # 10% data untuk final test (test akhir)

print(f"\nData distribution: train={TRAIN_RATIO*100}% val={VAL_RATIO*100}% test={TEST_RATIO*100}%")
print(f"Source: {SOURCE_DATASET}")
print(f"Destination: {YOLO_DATASET}")

# Step 1A: Buat folder struktur YOLO
# Kita perlu 3 folder: train, val, test
# Setiap folder punya subfolder images dan labels
for split in ['train', 'val', 'test']:
    # Buat folder images/train, images/val, images/test
    (YOLO_DATASET / 'images' / split).mkdir(parents=True, exist_ok=True)
    # Buat folder labels/train, labels/val, labels/test
    (YOLO_DATASET / 'labels' / split).mkdir(parents=True, exist_ok=True)
    print(f"  ✓ Created {split} folders")

# Step 1B: Copy dan organize gambar ke folder YOLO
# Loop setiap kelas (Bird-drop, Clean, Dusty, dll)
for class_name in CLASSES:
    # Path folder untuk kelas ini
    class_dir = SOURCE_DATASET / class_name
    
    # Cek apakah folder kelas ada
    if not class_dir.exists():
        print(f"⚠️  Folder {class_name} tidak ditemukan")
        continue
    
    # Ambil semua file gambar (.jpg dan .png)
    images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
    
    if not images:
        print(f"⚠️  No images in {class_name}")
        continue
    
    # Acak urutan gambar agar random
    random.shuffle(images)
    
    # Hitung berapa banyak gambar di split train, val, test
    total = len(images)
    train_count = int(total * TRAIN_RATIO)  # Gambar untuk train
    val_count = int(total * VAL_RATIO)      # Gambar untuk val
    # test otomatis sisa (total - train - val)
    
    # Pisah gambar sesuai ratio
    train_imgs = images[:train_count]
    val_imgs = images[train_count:train_count + val_count]
    test_imgs = images[train_count + val_count:]
    
    print(f"📦 {class_name}: {total} total → train:{len(train_imgs)} val:{len(val_imgs)} test:{len(test_imgs)}", end="")
    
    # Ambil ID class (angka) untuk file label
    class_id = class_to_id[class_name]
    
    # Step 1C: Copy setiap gambar ke folder yang sesuai
    # dan buat file label untuk YOLO
    for split_name, img_list in [('train', train_imgs), ('val', val_imgs), ('test', test_imgs)]:
        for img_path in img_list:
            # Copy gambar ke folder destination
            dst_img = YOLO_DATASET / 'images' / split_name / img_path.name
            shutil.copy2(img_path, dst_img)
            
            # Buat file label (.txt)
            # Format YOLO: class_id center_x center_y width height (normalized 0-1)
            # Kita pakai dummy label: full frame dengan class_id
            label_name = img_path.stem + '.txt'  # stem = nama tanpa extension
            label_path = YOLO_DATASET / 'labels' / split_name / label_name
            
            # Tulis ke file label
            # Format: {class_id} 0.5 0.5 1.0 1.0
            # - class_id = jenis fault (0-5)
            # - 0.5 0.5 = center point di tengah (x, y) normalized
            # - 1.0 1.0 = width, height (normalized) = full image
            with open(label_path, 'w') as f:
                f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")
    
    print(" ✓")

print(f"\n✅ Dataset YOLO siap di: {YOLO_DATASET}\n")

# ============ STEP 2: Create YOLO Config ============
# YAML = format file konfigurasi yang simple (key: value)
# File ini memberitahu YOLO dimana lokasi data dan berapa class
print("[STEP 2] Creating YOLO dataset config...")

# String yang berisi config data dalam format YAML
yaml_content = """path: ./data/yolo_dataset  # Path root dataset
train: images/train        # Folder training images (relative to path)
val: images/val            # Folder validation images
test: images/test          # Folder test images

nc: 6                      # Jumlah class (nc = number of classes)
names: ['Bird-drop', 'Clean', 'Dusty', 'Electrical-damage', 'Physical-Damage', 'Snow-Covered']  # Nama setiap class
"""

# Path file config (bakal jadi data/yolo_dataset/data.yaml)
yaml_path = BASE_DIR / 'yolo_dataset' / 'data.yaml'

# Buat file dan tulis config
with open(yaml_path, 'w') as f:
    f.write(yaml_content)

print(f"✅ Config created: {yaml_path}\n")

# ============ STEP 3: Load Model ============
# Penjelasan:
# - YOLO punya model pre-trained dari Ultralytics (sudah dilatih data besar)
# - Kita pakai itu sebagai starting point, terus fine-tune dengan data kita
# - Ada berbagai ukuran: n(nano), s(small), m(medium), b(big), l(large), x(xlarge)
# - Kita pakai 's' (small) = cepat dan hemat GPU memory
print("[STEP 3] Loading YOLOv10s model...")

# Load model pre-trained YOLOv10 small
# Ini akan download file .pt (weights) dari internet kalau belum ada
model = YOLO('yolov10s.pt')
print("✅ Model loaded successfully\n")

# ============ STEP 4: Training ============
# Penjelasan Training:
# - AI belajar dari gambar + label untuk detect object
# - Setiap epoch = 1 kali pass through semua data
# - Gradient descent = algorithm untuk adjust weights agar error berkurang
# - Batch = jumlah gambar diproses sebelum update weights
# - Epoch vs Batch:
#   * 1 Epoch = semua data sudah diproses 1x
#   * 1 Batch = subset dari data (misal 16 gambar)
#   * Dalam 1 epoch ada banyak batch

print("[STEP 4] Starting training...")
print("⏱️  Ini akan butuh 15-30 menit tergantung GPU kamu...\n")

# Hyperparameter (setting training)
EPOCHS = 50           # Berapa kali AI belajar dari semua data
                      # Lebih besar = akurat tapi risiko overfit
IMG_SIZE = 640        # Standard YOLO size (pixel), harus 32 kelipatan
BATCH_SIZE = 16       # Berapa gambar diproses per batch
                      # Lebih besar = lebih cepat tapi butuh RAM/VRAM besar
                      # Lebih kecil = hemat memory tapi lebih lambat
DEVICE = 0            # GPU device: 0 = GPU pertama, 'cpu' = CPU only
                      # GPU jauh lebih cepat dari CPU

# Penjelasan model.train():
results = model.train(
    data=str(yaml_path),        # Path ke file data.yaml config
    epochs=EPOCHS,              # Total epoch
    imgsz=IMG_SIZE,             # Input image size
    batch=BATCH_SIZE,           # Batch size
    device=DEVICE,              # Pakai GPU atau CPU
    patience=10,                # Early stopping: stop kalau ga improve 10 epoch
    save=True,                  # Simpan checkpoint setiap epoch
    project='./models',         # Folder output
    name='yolo_solar_detector', # Nama subfolder output
    pretrained=True,            # Pakai weight pre-trained (transfer learning)
    augment=True,               # Data augmentation (flip, rotate image saat training)
    verbose=True,               # Print detailed logs
)
# Hasil training disimpan di ./models/yolo_solar_detector/

print("\n✅ Training completed!\n")

# ============ STEP 5: Evaluate ============
# Penjelasan Evaluation:
# - Testing performance model pada data validation set
# - Metrics yang penting:
#   * mAP50 = Mean Average Precision at IOU 50% (IoU = overlap antara detected box dan ground truth)
#   * mAP50-95 = mAP di IoU 50% sampai 95% (standard metric)
#   * Semakin tinggi = semakin bagus

print("[STEP 5] Evaluating model...")

# Path folder hasil training
results_path = Path('./models/yolo_solar_detector')

# Load model terbaik dari training (weight terbaik)
# best.pt = checkpoint dengan validation score tertinggi
best_model_path = results_path / 'weights' / 'best.pt'
best_model = YOLO(str(best_model_path))

# Evaluate pada validation set
# Ini hitung precision, recall, mAP, dll
metrics = best_model.val(data=str(yaml_path), device=DEVICE)

print(f"✅ Evaluation complete")
print(f"   mAP50: {metrics.box.map50:.4f}")      # Metric IoU threshold 50%
print(f"   mAP50-95: {metrics.box.map:.4f}\n")   # Standard metric

# ============ STEP 6: Test Inference ============
# Penjelasan Inference:
# - Menggunakan model untuk predict object pada gambar baru
# - Ini adalah fase "production" - gunakan model untuk real world task
# - Confidence threshold = filter detections di bawah skor tertentu

print("[STEP 6] Running inference test...")

# Ambil folder test images
test_img_dir = BASE_DIR / 'yolo_dataset' / 'images' / 'test'
# Cari semua file gambar di folder test
test_images = list(test_img_dir.glob('*.jpg')) + list(test_img_dir.glob('*.png'))

if test_images:
    # Ambil gambar pertama untuk test
    test_image = str(test_images[0])
    print(f"Testing on: {Path(test_image).name}")
    
    # Run inference (prediction)
    # conf=0.5 = hanya terima detection dengan confidence >= 50%
    results_inf = best_model.predict(source=test_image, conf=0.5, device=DEVICE)
    result = results_inf[0]  # Hasil dari 1 gambar
    
    # Check apakah ada detection
    if result.boxes:
        print(f"✅ Detected {len(result.boxes)} object(s)")
        # Tampilkan detail setiap detection
        for i, box in enumerate(result.boxes):
            cls_id = int(box.cls[0])
            cls_name = CLASSES[cls_id]
            conf = float(box.conf[0])
            print(f"   [{i+1}] {cls_name}: confidence={conf:.2%}")
    else:
        print("⚠️  No objects detected")
else:
    print("⚠️  No test images found")

print()

# ============ STEP 7: Save Model ============
# Penjelasan:
# - best.pt = hasil training dari Step 4
# - Kita copy ke lokasi yang mudah diakses (./models/best_model.pt)
# - File .pt = PyTorch format (weights + architecture)
# - Ukuran file ~ 20-50 MB tergantung model size

print("[STEP 7] Saving final model...")

# Path dimana kita akan save final model
model_save_path = Path('./models/best_model.pt')
# Buat folder kalau belum ada
model_save_path.parent.mkdir(parents=True, exist_ok=True)

# Path best.pt dari training
best_pt = results_path / 'weights' / 'best.pt'

# Copy best.pt ke lokasi yang mudah diakses
if best_pt.exists():
    shutil.copy(best_pt, model_save_path)
    print(f"✅ Model saved: {model_save_path}")
else:
    print("⚠️  best.pt tidak ditemukan")

# ============ SUMMARY ============
print("\n" + "=" * 60)
print("🎉 YOLO Training Pipeline Complete!")
print("=" * 60)
print("\n📊 Summary:")
print("   ✓ Dataset prepared & split (train/val/test)")
print("   ✓ YOLO model trained")
print("   ✓ Model evaluated")
print("   ✓ Inference tested")
print("   ✓ Best model saved")
print("\n📁 Output locations:")
print(f"   - All models: ./models/")
print(f"   - Best weights: {model_save_path}")
print(f"   - Training logs: {results_path}")
print("\n💡 Next steps:")
print("   1. Check metrics di console - apakah akurat?")
print("   2. Gunakan best_model.pt untuk predict solar panel gambar baru")
print("   3. Buat script inference untuk production")
print("\n📚 Untuk belajar lebih lanjut:")
print("   - YOLO docs: https://docs.ultralytics.com/")
print("   - Deep Learning: https://www.deeplearning.ai/")
print("   - CV tutorials: https://pyimagesearch.com/")
