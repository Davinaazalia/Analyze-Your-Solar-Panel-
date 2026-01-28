# METODOLOGI PENELITIAN
## Deteksi Kegagalan Panel Surya Menggunakan Deep Learning YOLO Classification

---

## 1. PENDAHULUAN

### 1.1 Latar Belakang
Penelitian ini bertujuan untuk mengembangkan sistem klasifikasi otomatis untuk mendeteksi berbagai jenis kerusakan pada panel surya menggunakan metode deep learning.

### 1.2 Rumusan Masalah
- Bagaimana mengklasifikasikan kondisi panel surya secara otomatis?
- Model deep learning apa yang paling efektif untuk klasifikasi kondisi panel surya?
- Berapa tingkat akurasi yang dapat dicapai?

### 1.3 Tujuan Penelitian
- Mengembangkan model deep learning untuk klasifikasi kondisi panel surya
- Mengevaluasi performa model pada berbagai kondisi kerusakan
- Menghasilkan sistem yang dapat diimplementasikan untuk monitoring panel surya

---

## 2. LANDASAN TEORI

### 2.1 Panel Surya
Panel surya merupakan perangkat yang mengubah energi matahari menjadi energi listrik. Berbagai kondisi dapat mempengaruhi efisiensi panel surya.

### 2.2 Deep Learning
Deep learning adalah cabang dari machine learning yang menggunakan jaringan saraf tiruan berlapis untuk pembelajaran pola kompleks dari data.

### 2.3 YOLO (You Only Look Once)
YOLO adalah arsitektur deep learning yang efisien untuk deteksi dan klasifikasi objek. Penelitian ini menggunakan YOLOv8 Classification variant.

### 2.4 Transfer Learning
Teknik pembelajaran yang memanfaatkan model pre-trained untuk mempercepat pembelajaran dan meningkatkan akurasi pada dataset yang lebih kecil.

---

## 3. METODOLOGI PENELITIAN

### 3.1 Kerangka Penelitian

```
┌─────────────────────────────────────────────────────────────┐
│                    PENGUMPULAN DATA                         │
│              (Dataset Gambar Panel Surya)                   │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│              PREPROCESSING & EKSPLORASI DATA                │
│  - Analisis distribusi kelas                                │
│  - Deteksi ketidakseimbangan kelas                          │
│  - Visualisasi dataset                                      │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│                  PERSIAPAN DATASET                          │
│  - Split data (Train 70%, Val 20%, Test 10%)               │
│  - Struktur folder YOLO classification                      │
│  - Data augmentation                                        │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│              KONFIGURASI HYPERPARAMETER                     │
│  - Epoch, batch size, learning rate                         │
│  - Image size, patience                                     │
│  - Augmentation parameters                                  │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│                  TRAINING MODEL                             │
│  - Load pre-trained YOLOv8s-cls                            │
│  - Transfer learning                                        │
│  - Training dengan early stopping                           │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│                 EVALUASI MODEL                              │
│  - Test set evaluation                                      │
│  - Confusion matrix                                         │
│  - Per-class accuracy                                       │
│  - Classification report                                    │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│           VISUALISASI & ANALISIS HASIL                      │
│  - Training curves                                          │
│  - Sample predictions                                       │
│  - Performance metrics                                      │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│              DEPLOYMENT & INFERENCE                         │
│  - Save best model                                          │
│  - Production inference script                              │
│  - Testing pada data baru                                   │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Dataset

#### 3.2.1 Sumber Data
Dataset gambar panel surya dengan 6 kategori kondisi:
1. **Bird-drop**: Panel dengan kotoran burung
2. **Clean**: Panel dalam kondisi bersih
3. **Dusty**: Panel berdebu
4. **Electrical-damage**: Kerusakan elektrikal
5. **Physical-Damage**: Kerusakan fisik
6. **Snow-Covered**: Panel tertutup salju

#### 3.2.2 Distribusi Dataset
- **Total dataset**: ~907 gambar
- **Training set**: 70% (~635 gambar)
- **Validation set**: 20% (~181 gambar)
- **Test set**: 10% (~91 gambar)

#### 3.2.3 Format Data
- Format gambar: JPG, JPEG, PNG
- Struktur folder: Organized by class
- Resolusi: Variable (akan diresize ke 224x224)

### 3.3 Preprocessing Data

#### 3.3.1 Eksplorasi Data
- Analisis jumlah gambar per kelas
- Deteksi class imbalance
- Visualisasi distribusi dataset

#### 3.3.2 Data Preparation
- Pembagian dataset (train/val/test split)
- Reorganisasi struktur folder untuk YOLO
- Random shuffling untuk memastikan distribusi acak

### 3.4 Data Augmentation

Teknik augmentasi yang digunakan:
- **HSV Augmentation**: 
  - Hue (H): 0.015
  - Saturation (S): 0.7
  - Value (V): 0.4
- **Rotation**: ±10 derajat
- **Translation**: 0.1 (10% dari ukuran gambar)
- **Scale**: 0.5
- **Horizontal Flip**: 50% probability
- **Vertical Flip**: 0% (disabled)
- **Mosaic**: 0% (disabled untuk classification)

**Tujuan**: Meningkatkan generalisasi model dan mengurangi overfitting

### 3.5 Arsitektur Model

#### 3.5.1 Model Base
- **Model**: YOLOv8s-cls (Small variant)
- **Pre-trained**: ImageNet weights
- **Input size**: 224 x 224 pixels
- **Output**: 6 classes (softmax)

#### 3.5.2 Transfer Learning
- Menggunakan pre-trained weights dari ImageNet
- Fine-tuning seluruh layer untuk dataset spesifik
- Memanfaatkan feature extraction yang sudah dipelajari

### 3.6 Hyperparameter

#### 3.6.1 Training Parameters
| Parameter | Nilai | Keterangan |
|-----------|-------|------------|
| Epochs | 30 | Jumlah iterasi training |
| Batch Size | 16 | Jumlah gambar per batch |
| Learning Rate | 0.001 | Laju pembelajaran |
| Image Size | 224x224 | Ukuran input |
| Patience | 15 | Early stopping patience |
| Optimizer | SGD/Adam | Default YOLO optimizer |

#### 3.6.2 Device Configuration
- GPU: NVIDIA CUDA (jika tersedia)
- CPU: Fallback jika GPU tidak tersedia
- Mixed Precision: Auto-detect

### 3.7 Training Process

#### 3.7.1 Training Procedure
1. **Initialization**: Load pre-trained YOLOv8s-cls
2. **Forward Pass**: Prediksi pada batch training
3. **Loss Calculation**: Compute classification loss
4. **Backward Pass**: Gradient computation
5. **Weight Update**: Optimizer step
6. **Validation**: Evaluate pada validation set setiap epoch
7. **Early Stopping**: Stop jika tidak ada improvement selama PATIENCE epochs

#### 3.7.2 Loss Function
- **Classification Loss**: Cross-entropy loss
- **Optimization**: Minimize classification error

### 3.8 Evaluasi Model

#### 3.8.1 Metrics
1. **Top-1 Accuracy**: Persentase prediksi benar pada pilihan pertama
2. **Top-5 Accuracy**: Persentase class benar ada di 5 prediksi teratas
3. **Confusion Matrix**: Analisis kesalahan klasifikasi antar kelas
4. **Per-Class Accuracy**: Akurasi untuk setiap kategori
5. **Precision, Recall, F1-Score**: Metrik klasifikasi detail

#### 3.8.2 Validation Strategy
- **Training Validation**: Monitoring performa pada validation set setiap epoch
- **Test Evaluation**: Evaluasi final pada test set yang belum pernah dilihat
- **Cross-validation**: Tidak digunakan (single split)

#### 3.8.3 Confusion Matrix Analysis
```
                 Predicted Class
              BD   CL   DU   ED   PD   SC
True    BD   [ ]  [ ]  [ ]  [ ]  [ ]  [ ]
Class   CL   [ ]  [ ]  [ ]  [ ]  [ ]  [ ]
        DU   [ ]  [ ]  [ ]  [ ]  [ ]  [ ]
        ED   [ ]  [ ]  [ ]  [ ]  [ ]  [ ]
        PD   [ ]  [ ]  [ ]  [ ]  [ ]  [ ]
        SC   [ ]  [ ]  [ ]  [ ]  [ ]  [ ]

BD=Bird-drop, CL=Clean, DU=Dusty, 
ED=Electrical-damage, PD=Physical-Damage, SC=Snow-Covered
```

### 3.9 Visualisasi Hasil

#### 3.9.1 Training Curves
- **Loss Curve**: Training vs Validation loss per epoch
- **Accuracy Curve**: Top-1 dan Top-5 accuracy progression
- **Learning Rate Schedule**: Visualisasi learning rate decay

#### 3.9.2 Prediction Visualization
- Sample predictions dengan ground truth
- Confidence scores
- Correct vs Incorrect predictions marking

### 3.10 Model Deployment

#### 3.10.1 Model Saving
- Best model: Berdasarkan validation accuracy
- Last model: Checkpoint terakhir
- Hyperparameters: Saved to CSV untuk reproducibility

#### 3.10.2 Inference Pipeline
```python
Input Image → Preprocessing → Model Prediction → 
Class + Confidence → Output
```

---

## 4. IMPLEMENTASI

### 4.1 Tools & Libraries
- **Python**: 3.8+
- **PyTorch**: Deep learning framework
- **Ultralytics**: YOLO implementation
- **NumPy**: Numerical operations
- **Pandas**: Data manipulation
- **Matplotlib/Seaborn**: Visualization
- **Scikit-learn**: Metrics calculation

### 4.2 Hardware Requirements
- **Minimum**: CPU, 8GB RAM
- **Recommended**: NVIDIA GPU (CUDA), 16GB RAM
- **Storage**: ~5GB untuk dataset dan model

### 4.3 Jupyter Notebook Structure
1. Import Libraries
2. Hyperparameter Configuration
3. Data Exploration & Preprocessing
4. Dataset Preparation
5. Sample Images Visualization
6. Load YOLO Model
7. Model Training
8. Save Model
9. Training Results Visualization
10. Model Evaluation
11. Confusion Matrix
12. Sample Predictions Visualization
13. Inference on New Images
14. Final Summary & Export

---

## 5. HASIL YANG DIHARAPKAN

### 5.1 Output Model
- ✅ Trained YOLOv8s-cls model (.pt file)
- ✅ Training metrics (CSV)
- ✅ Confusion matrix (PNG, CSV)
- ✅ Sample predictions (PNG)
- ✅ Inference script (Python)

### 5.2 Performance Metrics
- **Target Accuracy**: >85% Top-1 Accuracy
- **Generalization**: Good performance on unseen test data
- **Inference Speed**: Real-time capable

### 5.3 Deliverables
1. Trained model file
2. Training documentation (notebook)
3. Evaluation metrics report
4. Inference script untuk production
5. Visualisasi hasil training

---

## 6. VALIDASI & TESTING

### 6.1 Internal Validation
- Validation set evaluation setiap epoch
- Early stopping berdasarkan validation loss

### 6.2 Test Set Evaluation
- One-time evaluation pada unseen test set
- Final performance reporting

### 6.3 Production Testing
- Inference speed testing
- Prediction accuracy pada data baru
- Edge case handling

---

## 7. LIMITASI PENELITIAN

### 7.1 Dataset Limitations
- Ukuran dataset terbatas (~907 images)
- Distribusi class mungkin tidak seimbang
- Kondisi pengambilan gambar bervariasi

### 7.2 Model Limitations
- Bergantung pada kualitas gambar input
- Performance dapat menurun pada kondisi lighting ekstrem
- Tidak dapat mendeteksi multiple defects dalam satu gambar

### 7.3 Computational Limitations
- Membutuhkan GPU untuk training optimal
- Inference time bergantung pada hardware

---

## 8. FUTURE WORK

### 8.1 Dataset Expansion
- Menambah jumlah gambar per kelas
- Meningkatkan variasi kondisi
- Data collection dari berbagai sumber

### 8.2 Model Improvement
- Experiment dengan YOLOv8m, YOLOv8l untuk akurasi lebih tinggi
- Ensemble methods
- Advanced augmentation techniques

### 8.3 Feature Addition
- Multi-label classification (multiple defects)
- Severity level detection
- Real-time monitoring system integration

### 8.4 Deployment
- Web application deployment
- Mobile app integration
- IoT device implementation

---

## 9. KESIMPULAN

Metodologi penelitian ini menyediakan framework lengkap untuk:
1. Pengumpulan dan preprocessing data panel surya
2. Training model deep learning dengan YOLO
3. Evaluasi komprehensif performa model
4. Deployment untuk aplikasi real-world

Model yang dihasilkan diharapkan dapat mengklasifikasikan kondisi panel surya dengan akurasi tinggi dan dapat diimplementasikan untuk sistem monitoring otomatis.

---

## 10. REFERENSI

1. Redmon, J., et al. (2016). "You Only Look Once: Unified, Real-Time Object Detection"
2. Ultralytics YOLOv8 Documentation. https://docs.ultralytics.com/
3. Deep Learning for Computer Vision, Stanford CS231n
4. Solar Panel Fault Detection Research Papers (sesuaikan dengan literatur yang digunakan)

---

## APPENDIX A: HYPERPARAMETER TUNING GUIDE

### Jika Accuracy Rendah (<80%)
- ✅ Tingkatkan EPOCHS (50-100)
- ✅ Turunkan LEARNING_RATE (0.0001)
- ✅ Tambahkan data augmentation
- ✅ Gunakan model lebih besar (YOLOv8m atau YOLOv8l)

### Jika Overfitting (Train Acc >> Val Acc)
- ✅ Tambahkan augmentation
- ✅ Turunkan model complexity
- ✅ Tambahkan regularization
- ✅ Perbanyak data training

### Jika Out of Memory
- ✅ Turunkan BATCH_SIZE (8, 4)
- ✅ Turunkan IMG_SIZE (128, 160)
- ✅ Gunakan model lebih kecil (YOLOv8n)

---

## APPENDIX B: TROUBLESHOOTING

### Common Issues
1. **CUDA Out of Memory**: Turunkan batch size atau image size
2. **Low Accuracy**: Tingkatkan epochs, cek data quality
3. **Training Too Slow**: Gunakan GPU, turunkan image size
4. **Model Not Converging**: Adjust learning rate, cek data preprocessing

---

**Dokumen ini merupakan panduan metodologi untuk penelitian klasifikasi kondisi panel surya menggunakan deep learning YOLO.**

*Terakhir diperbarui: Januari 2026*
