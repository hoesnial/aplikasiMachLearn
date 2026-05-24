# 🏥 Aplikasi Deteksi Penyakit - Machine Learning

Aplikasi web berbasis Streamlit untuk mendeteksi/memprediksi berbagai penyakit menggunakan algoritma klasifikasi Machine Learning.

## 📁 Struktur Project

```
aplikasiMachLearn/
├── app.py                          # Entry point aplikasi Streamlit
├── requirements.txt                # Dependencies
├── README.md                       # Dokumentasi
├── dataset/                        # Folder kumpulan dataset
│   ├── healthcare-dataset-stroke-data.csv
│   └── diabetes_dataset.csv        # Dataset diabetes untuk modul XGBoost
├── pages/                          # Halaman-halaman aplikasi
│   ├── __init__.py
│   ├── beranda.py                  # Halaman beranda
│   ├── deteksi_stroke.py           # Deteksi stroke (AKTIF)
│   ├── deteksi_diabetes_praktikum.py # Halaman praktikum XGBoost (preprocessing + CV)
│   ├── deteksi_pasien_treatment.py # Deteksi pasien treatment
│   ├── deteksi_bpjs.py             # Deteksi BPJS
│   ├── deteksi_jantung.py          # Deteksi penyakit jantung
│   ├── deteksi_liver.py            # Deteksi penyakit liver
│   └── deteksi_lainnya.py          # Halaman tambahan
├── utils/                          # Modul utilitas
│   ├── __init__.py
│   ├── preprocessing.py            # Fungsi preprocessing data
│   ├── diabetes_xgb.py             # (legacy) pipeline XGBoost
│   ├── diabetes_xgb_praktikum.py   # Pipeline XGBoost untuk praktikum (preprocessing, IQR, CV)
│   └── models.py                   # Fungsi model ML
└── models/                         # Folder untuk menyimpan model (opsional)
    └── __init__.py
```

## 🚀 Cara Menjalankan

1. Aktifkan virtual environment (Windows PowerShell):

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
& ".\.venv\Scripts\Activate.ps1"
```

2. Pasang dependencies (jika belum):

```powershell
pip install -r requirements.txt
```

3. Jalankan aplikasi (gunakan interpreter venv agar lancar):

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

4. Buka browser di `http://localhost:8501`

## 🤖 Model Klasifikasi Tersedia

| No | Model | Deskripsi |
|----|-------|-----------|
| 1 | Logistic Regression | Model linear untuk klasifikasi biner |
| 2 | Decision Tree | Model berbasis pohon keputusan |
| 3 | Random Forest | Ensemble dari banyak decision tree |
| 4 | W-KNN | KNN dengan bobot jarak (weighted) |
| 5 | XGBoost | Gradient boosting yang powerful |

## 📋 Halaman Deteksi

1. **Deteksi Stroke** ✅ - Healthcare Stroke Dataset
2. **Deteksi Diabetes XGBoost** ✅ - Diabetes Dataset
3. **Pasien Treatment** 🔜 - Dataset belum ditambahkan
4. **BPJS** 🔜 - Dataset belum ditambahkan
5. **Penyakit Jantung** 🔜 - Dataset belum ditambahkan
6. **Penyakit Liver** 🔜 - Dataset belum ditambahkan
7. **Lainnya** 🔜 - Dataset belum ditentukan

## 📝 Cara Menambahkan Dataset Baru

1. Letakkan file CSV di folder `dataset/`
2. Buat fungsi preprocessing di `utils/preprocessing.py`
3. Implementasikan halaman di `pages/deteksi_xxx.py`
4. Ikuti pola yang sudah ada di `deteksi_stroke.py`

## 🩸 Modul XGBoost Diabetes (Praktikum)

Modul praktikum difokuskan pada eksperimen preprocessing, perbandingan metode, dan evaluasi yang siap dipresentasikan.

- File utilitas utama: `utils/diabetes_xgb_praktikum.py`
- Halaman praktikum: `pages/deteksi_diabetes_praktikum.py`
- Dataset: `dataset/diabetes_dataset.csv`
- Model utama: `XGBClassifier`
- Baseline pembanding: Logistic Regression
- Evaluasi: `train_test_split` + `StratifiedKFold` cross-validation
- Fitur tambahan: handling outlier IQR, opsional SMOTE, tuning `RandomizedSearchCV`, export CSV/PNG, interpretasi confusion matrix, dan feature importance

Catatan penting:

- XGBoost tidak terlalu sensitif terhadap scaling karena berbasis decision tree.
- Perbandingan utama ada pada dua dataset: tanpa outlier dan dengan outlier.
- Perbandingan metode klasifikasi ditambahkan melalui baseline Logistic Regression agar alur praktikum lebih lengkap.

## 🔎 Ringkasan Fitur Praktikum

1. Cek dan imputasi missing value.
2. Encoding fitur kategorikal.
3. Deteksi outlier dengan IQR dan visualisasi boxplot.
4. Perbandingan dataset tanpa dan dengan outlier.
5. Cross validation dengan `StratifiedKFold`.
6. Perbandingan baseline Logistic Regression vs XGBoost.
7. Hyperparameter tuning dengan `RandomizedSearchCV`.
8. Analisis imbalance dan opsi SMOTE.
9. Interpretasi feature importance dan confusion matrix.
10. Export hasil evaluasi, classification report, dan confusion matrix.

## 🚀 Cara Menjalankan

1. Aktifkan virtual environment:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
& ".\.venv\Scripts\Activate.ps1"
```

2. Pasang dependency:

```powershell
pip install -r requirements.txt
```

3. Jalankan Streamlit:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

4. Buka browser di `http://localhost:8501`

Jika ingin install XGBoost secara manual:

```powershell
pip install xgboost
```
