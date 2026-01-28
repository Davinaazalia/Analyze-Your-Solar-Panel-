#!/usr/bin/env python3
"""
Solar Panel Fault Detection - YOLO Classification Training Script
Program untuk melatih YOLO model classification (bukan detection)
Model ini classify panel: Bird-drop, Clean, Dusty, Electrical-damage, Physical-Damage, atau Snow-Covered
"""

# ============================================
# BAGIAN 1: IMPORT LIBRARY
# ============================================
import os
import shutil
import random
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
import pandas as pd

print("=" * 70)
print("🚀 YOLO CLASSIFICATION Training - Solar Panel Fault Detection")
print("=" * 70)
print("\n📌 Penjelasan Mode:")
print("   Classification: AI classify panel → kelas apa?")
print("   (Beda dari Detection yang detect DIMANA object di image)")
print()

# ============ SETUP PATH & CONFIG ============
print("[CONFIG] Setting up paths...")

# Path folder asli (sudah dikelompok per class)
BASE_DIR = Path("./data")
SOURCE_DATASET = BASE_DIR / "dataset"      # Folder asli: Bird-drop/, Clean/, dll
YOLO_CLASSIFY_DATASET = BASE_DIR / "yolo_classify_dataset"  # Folder output

# Kelas (harus sesuai nama folder di SOURCE_DATASET)
CLASSES = ['Bird-drop', 'Clean', 'Dusty', 'Electrical-damage', 'Physical-Damage', 'Snow-Covered']

# Ratio split data
# Penjelasan:
# - Train (70%): Data untuk belajar AI
# - Val (20%): Data untuk test saat training (validate progress)
# - Test (10%): Data untuk final evaluation (unseen data)
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

print(f"   Source dataset: {SOURCE_DATASET}")
print(f"   Output dataset: {YOLO_CLASSIFY_DATASET}")
print(f"   Classes: {', '.join(CLASSES)}")
print(f"   Split ratio: train {TRAIN_RATIO*100}% | val {VAL_RATIO*100}% | test {TEST_RATIO*100}%")

# ============ STEP 1: Prepare Classification Dataset ============
print("\n[STEP 1] Preparing YOLO classification dataset...")
# Penjelasan:
# YOLO classification butuh struktur folder:
# yolo_classify_dataset/
#   ├─ train/
#   │  ├─ Bird-drop/      ← subfolder per class
#   │  │  ├─ image1.jpg
#   │  │  ├─ image2.jpg
#   │  │  └─ ...
#   │  ├─ Clean/
#   │  │  └─ ...
#   │  └─ ... (class lain)
#   ├─ val/
#   │  ├─ Bird-drop/
#   │  └─ ... (sama struktur)
#   └─ test/
#      └─ ... (sama struktur)

# Step 1A: Buat struktur folder YOLO
for split in ['train', 'val', 'test']:
    for class_name in CLASSES:
        # Path folder untuk split dan class
        folder_path = YOLO_CLASSIFY_DATASET / split / class_name
        folder_path.mkdir(parents=True, exist_ok=True)

print("   ✓ Folder structure created")

# Step 1B: Copy gambar dari SOURCE ke YOLO format
# Loop setiap class
for class_name in CLASSES:
    class_dir = SOURCE_DATASET / class_name
    
    # Cek folder class ada
    if not class_dir.exists():
        print(f"   ⚠️  Folder {class_name} tidak ditemukan, skip")
        continue
    
    # Ambil semua gambar dari folder class ini
    images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.jpeg')) + list(class_dir.glob('*.JPG')) + list(class_dir.glob('*.JPEG')) + list(class_dir.glob('*.png'))
    
    if not images:
        print(f"   ⚠️  No images in {class_name}, skip")
        continue
    
    # Acak urutan gambar
    random.shuffle(images)
    
    # Hitung split count
    total = len(images)
    train_count = int(total * TRAIN_RATIO)
    val_count = int(total * VAL_RATIO)
    
    # Pisah gambar sesuai ratio
    train_imgs = images[:train_count]
    val_imgs = images[train_count:train_count + val_count]
    test_imgs = images[train_count + val_count:]
    
    print(f"   📦 {class_name:20} | Total: {total:3} → train: {len(train_imgs):3} | val: {len(val_imgs):2} | test: {len(test_imgs):2}")
    
    # Copy gambar ke setiap split folder
    # Penjelasan: Kita copy file, bukan move, agar file asli tetap aman
    for split_name, img_list in [('train', train_imgs), ('val', val_imgs), ('test', test_imgs)]:
        dst_folder = YOLO_CLASSIFY_DATASET / split_name / class_name
        for img_path in img_list:
            dst_path = dst_folder / img_path.name
            # Copy file (copy2 = copy + preserve metadata)
            shutil.copy2(img_path, dst_path)

print(f"\n✅ Dataset preparation complete!")
print(f"   Output: {YOLO_CLASSIFY_DATASET}\n")

# ============ STEP 2: Load YOLO Classification Model ============
print("[STEP 2] Loading YOLO classification model...")
# Penjelasan:
# YOLO punya 2 tipe model:
# - Detection: detect object + location (butuh bbox annotation)
# - Classification: classify image (butuh folder per class) ← kami pakai ini
# 
# Model size:
# - nano (n): tercepat, accuracy terendah
# - small (s): balance cepat & akurat ← kita pakai ini
# - medium (m): lebih akurat, lebih lambat
# - large (l), extra-large (x): paling akurat, paling lambat

# Coba load pre-trained YOLO classification model dengan priority:
# 1. YOLOv11 (latest, terbaik)
# 2. YOLOv10 (older, stable)
# 3. YOLOv8 (fallback, proven)
MODEL_NAME = None
try:
    model = YOLO('yolov11s-cls.pt')  # Latest YOLOv11
    MODEL_NAME = 'YOLOv11s-cls'
    print("   ✓ Loaded YOLOv11s classification model (LATEST)")
except:
    print("   ⚠️  YOLOv11 error, trying YOLOv10...")
    try:
        model = YOLO('yolov10s-cls.pt')  # YOLOv10
        MODEL_NAME = 'YOLOv10s-cls'
        print("   ✓ Loaded YOLOv10s classification model")
    except:
        print("   ⚠️  YOLOv10 error, trying YOLOv8...")
        try:
            model = YOLO('yolov8s-cls.pt')  # v8 classification
            MODEL_NAME = 'YOLOv8s-cls'
            print("   ✓ Loaded YOLOv8s classification model")
        except:
            print("   ⚠️  YOLO models error, using ResNet50 with YOLO-style training...")
            # Fallback ke ResNet50 (tensorflow based)
            import tensorflow as tf
            
            # Load ResNet50 pre-trained
            base_model = tf.keras.applications.ResNet50(
                input_shape=(IMG_SIZE, IMG_SIZE, 3),
                include_top=False,
                weights='imagenet'
            )
            
            # Freeze base layers (transfer learning)
            base_model.trainable = False
            
            # Build model dengan classification head
            model = tf.keras.Sequential([
                base_model,
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(len(CLASSES), activation='softmax')
            ])
            
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            print("   ✓ Loaded ResNet50 model (TensorFlow backend)")
            MODEL_NAME = 'ResNet50'

print("✅ Model loaded successfully\n")
print(f"   Using model: {MODEL_NAME}\n")

# Folder hasil training (digunakan berulang di bawah)
results_path = Path('./models/yolo_classify_solar')

# ============ STEP 3: Training Configuration ============
print("[STEP 3] Setting up training configuration...")

# Hyperparameter training (parameter yang mengontrol proses belajar)
EPOCHS = 30           # Total epoch yang akan dijalankan
                       # Lebih besar = model belajar lebih lama
                       # Tapi risk overfit (hafal data tapi ga generalize)
                       # 100 epoch reasonable untuk 907 images

IMG_SIZE = 224         # Input image size (pixel)
                       # YOLO classifier standard: 224x224
                       # Smaller = lebih cepat, less memory
                       # Bigger = more detail, needs more memory

BATCH_SIZE = 16        # Batch size = berapa gambar diproses sebelum update weights
                       # Lebih besar = lebih cepat tapi butuh RAM/VRAM besar
                       # Kalau GPU out of memory, turunkan ke 8 atau 4

import torch
# Auto-detect device: use GPU if available, else fallback to CPU
DEVICE = 0 if torch.cuda.is_available() else "cpu"  # 'cpu' when no GPU detected
device_label = "CPU" if DEVICE == "cpu" else f"GPU {DEVICE}"
                       # GPU ~10x lebih cepat dari CPU

PATIENCE = 20          # Early stopping patience
                       # Jika val loss ga improve 20 epoch, stop training
                       # Ini cegah overfitting & waste time

print(f"   Epochs: {EPOCHS}")
print(f"   Image size: {IMG_SIZE}x{IMG_SIZE}")
print(f"   Batch size: {BATCH_SIZE}")
print(f"   Device: {device_label}")
print(f"   Early stopping patience: {PATIENCE} epochs")
print()

# ============ STEP 4: Train Model ============
print("[STEP 4] Starting training...")
print("⏱️  Ini akan butuh ~15-20 menit tergantung GPU kamu...\n")

# Penjelasan training:
# - AI belajar pattern dari train set
# - Setiap epoch:
#   1. Forward pass: prediksi output
#   2. Hitung loss: error antara prediksi vs actual
#   3. Backward pass: hitung gradient
#   4. Update weights: improve model
# - Validate pada val set setiap epoch untuk monitor progress
# - Kalau val loss ga improve, stop (early stopping)

# Path folder output hasil training
results = model.train(
    data=str(YOLO_CLASSIFY_DATASET),  # Path ke folder dataset
                                       # YOLO auto-detect structure: train/, val/, test/
    
    epochs=EPOCHS,                     # Total epoch
    imgsz=IMG_SIZE,                    # Input image size
    batch=BATCH_SIZE,                  # Batch size
    device=DEVICE,                     # GPU/CPU
    patience=PATIENCE,                 # Early stopping
    
    # Additional configs
    save=True,                         # Simpan checkpoint setiap epoch
    project='./models',                # Folder output
    name='yolo_classify_solar',        # Nama subfolder (hasil di ./models/yolo_classify_solar/)
    
    # Training mode
    pretrained=True,                   # Use pre-trained weights (transfer learning)
    augment=True,                      # Data augmentation (flip, rotate, dll)
                                       # Ini buat AI lebih generalize
    
    # Logging
    verbose=True,                      # Print detailed logs
)

print("\n✅ Training selesai!\n")

# ============ STEP 4b: Simpan dan plot metrik training ============
print("[STEP 4b] Saving training curves...")

results_csv = results_path / 'results.csv'
plot_path = results_path / 'training_curves.png'

if not results_csv.exists():
    print(f"⚠️  results.csv tidak ditemukan di {results_csv}")
else:
    df = pd.read_csv(results_csv)
    epochs = df['epoch'] if 'epoch' in df.columns else range(len(df))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    loss_cols = [c for c in ['train/box_loss', 'train/cls_loss', 'train/dfl_loss', 'val/box_loss', 'val/cls_loss', 'val/dfl_loss'] if c in df.columns]
    if loss_cols:
        for c in loss_cols:
            axes[0].plot(epochs, df[c], label=c)
        axes[0].set_title('Loss per epoch')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
    else:
        axes[0].text(0.5, 0.5, 'Loss columns not found', ha='center')

    metric_cols = [c for c in ['metrics/mAP50', 'metrics/mAP50-95', 'metrics/precision', 'metrics/recall', 'metrics/acc_top1', 'metrics/acc_top5'] if c in df.columns]
    if metric_cols:
        for c in metric_cols:
            axes[1].plot(epochs, df[c], label=c)
        axes[1].set_title('Metrics per epoch')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Score')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, 'Metric columns not found', ha='center')

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"✅ Curves saved: {plot_path}")

# ============ STEP 5: Model Evaluation ============
print("[STEP 5] Evaluating model on test set...")
# Penjelasan Evaluation:
# - Test pada unseen data (test set yang belum pernah dilihat AI)
# - Hitung metrics:
#   * Top-1 Accuracy: % prediksi correct class
#   * Top-5 Accuracy: % correct class ada di top 5 predictions
#   * Loss: error value

best_model_path = results_path / 'weights' / 'best.pt'

# Load best model (checkpoint terbaik)
# best.pt = epoch dengan validation accuracy tertinggi
best_model = YOLO(str(best_model_path))

# Validate pada test set
# Ini akan print accuracy metrics
metrics = best_model.val(data=str(YOLO_CLASSIFY_DATASET), split='test', device=DEVICE)

print(f"\n✅ Evaluation complete\n")

# ============ STEP 6: Test Inference ============
print("[STEP 6] Running inference test...")
# Penjelasan Inference:
# - Gunakan model untuk predict pada gambar baru (production use)
# - Confidence score: berapa yakin AI dengan prediksinya (0-1)
# - Semakin tinggi = semakin confident

# Ambil beberapa gambar test untuk demo
test_folder = YOLO_CLASSIFY_DATASET / 'test'
test_images = list(test_folder.rglob('*.jpg')) + list(test_folder.rglob('*.png'))

if test_images:
    # Ambil 5 gambar random untuk test
    sample_images = random.sample(test_images, min(5, len(test_images)))
    
    print(f"Testing on {len(sample_images)} sample images:\n")
    
    for img_path in sample_images:
        # Run inference (predict)
        # conf=0.5 = hanya terima prediksi dengan confidence >= 50%
        results_inf = best_model.predict(source=str(img_path), conf=0.5, device=DEVICE, verbose=False)
        
        result = results_inf[0]
        
        # Ambil top prediction
        # result.probs = probability untuk setiap class
        # top1_class = index class dengan probability tertinggi
        top1_class = int(result.probs.top1)
        top1_conf = float(result.probs.top1conf)
        
        # Mapping class index ke class name
        class_name = CLASSES[top1_class]
        
        # Actual class dari folder structure
        actual_class = img_path.parent.name
        is_correct = "✓" if class_name == actual_class else "✗"
        
        print(f"   {is_correct} {img_path.name:30} | Predicted: {class_name:20} ({top1_conf:.2%}) | Actual: {actual_class}")

print()

# ============ STEP 7: Save Final Model ============
print("[STEP 7] Saving final model...")
# Penjelasan:
# - best.pt = checkpoint terbaik dari training
# - Ini sudah ada di results_path/weights/best.pt
# - Kita copy ke lokasi mudah diakses

model_save_path = Path('./models/best_classifier_model.pt')
model_save_path.parent.mkdir(parents=True, exist_ok=True)

best_pt = results_path / 'weights' / 'best.pt'
if best_pt.exists():
    shutil.copy(best_pt, model_save_path)
    print(f"✅ Model saved: {model_save_path}")
    print(f"   File size: {model_save_path.stat().st_size / 1e6:.1f} MB")
else:
    print("⚠️  best.pt tidak ditemukan")

print()

# ============ STEP 8: Create Inference Script Template ============
print("[STEP 8] Creating inference script template...\n")

# Template script untuk inference di production
inference_script = '''#!/usr/bin/env python3
"""
Solar Panel Classification - Inference Script
Gunakan model terlatih untuk predict class panel surya
"""

from ultralytics import YOLO
from pathlib import Path

# ============ CONFIG ============
# Load model terlatih
model_path = "./models/best_classifier_model.pt"
model = YOLO(model_path)

# Class names (harus sesuai order training)
CLASSES = ['Bird-drop', 'Clean', 'Dusty', 'Electrical-damage', 'Physical-Damage', 'Snow-Covered']

# ============ INFERENCE ============
def predict_panel(image_path, conf_threshold=0.5):
    """
    Predict class panel surya dari gambar
    
    Args:
        image_path: Path ke gambar panel surya
        conf_threshold: Confidence threshold (0-1)
    
    Returns:
        class_name: Nama class
        confidence: Confidence score
    """
    # Run inference
    results = model.predict(source=image_path, verbose=False)
    result = results[0]
    
    # Get top prediction
    top1_class = int(result.probs.top1)
    top1_conf = float(result.probs.top1conf)
    class_name = CLASSES[top1_class]
    
    # Check confidence threshold
    if top1_conf >= conf_threshold:
        return class_name, top1_conf
    else:
        return "Low Confidence", top1_conf

# ============ EXAMPLE USAGE ============
if __name__ == "__main__":
    # Test pada single image
    test_image = "./data/dataset/Bird-drop/Bird(1).jpg"
    
    if Path(test_image).exists():
        class_name, confidence = predict_panel(test_image)
        print(f"Image: {test_image}")
        print(f"Predicted Class: {class_name}")
        print(f"Confidence: {confidence:.2%}")
    else:
        print(f"Image not found: {test_image}")
    
    # Test pada folder
    print("\\n" + "="*50)
    print("Batch inference on folder:")
    test_folder = Path("./data/dataset/Bird-drop")
    for img_file in list(test_folder.glob("*.jpg"))[:3]:  # First 3 images
        class_name, confidence = predict_panel(str(img_file))
        print(f"  {img_file.name:30} → {class_name:20} ({confidence:.2%})")
'''

script_path = Path('./inference_classifier.py')
with open(script_path, 'w') as f:
    f.write(inference_script)

print(f"✅ Inference script created: {script_path}\n")

# ============ SUMMARY ============
print("=" * 70)
print("🎉 YOLO Classification Training Pipeline Complete!")
print("=" * 70)
print("\n📊 Summary:")
print("   ✓ Dataset prepared (train/val/test split)")
print("   ✓ YOLO classification model trained")
print("   ✓ Model evaluated on test set")
print("   ✓ Inference tested")
print("   ✓ Best model saved")
print("   ✓ Inference script template created")

print("\n📁 Output locations:")
print(f"   - Training logs: {results_path}")
print(f"   - Best model: {model_save_path}")
print(f"   - Inference script: {script_path}")

print("\n📈 Model Architecture:")
print(f"   - Model: {MODEL_NAME}")
print(f"   - Classes: {len(CLASSES)}")
print(f"   - Training images: ~{int(907 * TRAIN_RATIO)}")
print(f"   - Validation images: ~{int(907 * VAL_RATIO)}")
print(f"   - Test images: ~{int(907 * TEST_RATIO)}")

print("\n💡 Next Steps:")
print("   1. Check metrics di console - accuracy berapa?")
print("   2. Jalankan inference_classifier.py untuk test pada gambar baru")
print("   3. Deploy best_classifier_model.pt untuk production")
print("   4. Monitor performance di real world data")

print("\n📚 Resources untuk belajar:")
print("   - YOLO Docs: https://docs.ultralytics.com/")
print("   - Transfer Learning: https://cs231n.github.io/transfer-learning/")
print("   - Deep Learning: https://www.deeplearning.ai/")

print("\n✨ Training complete! Model siap digunakan untuk classify solar panels.\n")
