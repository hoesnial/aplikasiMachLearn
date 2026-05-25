# 🏥 Aplikasi Deteksi Penyakit - Machine Learning

Aplikasi web berbasis Streamlit untuk mendeteksi/memprediksi berbagai penyakit
menggunakan algoritma klasifikasi Machine Learning.

Halaman aktif saat ini:

| Halaman | Dataset | Base Method |
|---------|---------|-------------|
| ❤️ **Deteksi Penyakit Kardiovaskular** | `cardio_train.csv` (70.000 pasien) | **Extra Trees Classifier** |
| 🩸 **Deteksi Diabetes (XGBoost)** | `diabetes_dataset.csv` | **XGBoost** |
| 🩸 **Deteksi Diabetes (Eksperimen)** | `diabetes_dataset.csv` | XGBoost + analisis eksperimen |

## 📁 Struktur Project

```
aplikasiMachLearn/
├── app.py                                       # Entry point Streamlit
├── requirements.txt                             # Dependencies
├── README.md                                    # Dokumentasi
├── Pembangunan_Model_Preprocessing.ipynb        # Notebook eksperimen Cardiovascular
├── xgboost_pembangunan_model_preprocessing.ipynb# Notebook eksperimen Diabetes
├── dataset/
│   ├── cardio_train.csv                         # Cardiovascular dataset (70k baris)
│   └── diabetes_dataset.csv                     # Diabetes dataset
├── models/
│   ├── diabetes_xgb_model.joblib                # Model XGBoost diabetes (precomputed)
│   └── diabetes_xgb_praktikum.joblib            # Model praktikum diabetes
├── pages/
│   ├── __init__.py
│   ├── beranda.py
│   ├── deteksi_cardiovascular.py                # Aktif (Extra Trees Classifier)
│   ├── deteksi_diabetes.py                      # Aktif (XGBoost eksperimen)
│   ├── deteksi_diabetes_praktikum.py            # Aktif (XGBoost praktikum)
│   ├── deteksi_pasien_treatment.py
│   ├── deteksi_bpjs.py
│   ├── deteksi_jantung.py
│   ├── deteksi_liver.py
│   └── deteksi_lainnya.py
├── utils/
│   ├── __init__.py
│   ├── preprocessing.py                         # Pipeline cardiovascular
│   ├── models.py                                # Wrapper 5 model klasifikasi
│   ├── diabetes_xgb.py                          # Pipeline diabetes (eksperimen)
│   └── diabetes_xgb_praktikum.py                # Pipeline diabetes (praktikum)
└── scripts/
    └── test_bundle.py
```

## 🚀 Cara Menjalankan

### 1. Clone repository

```bash
git clone https://github.com/hoesnial/aplikasiMachLearn.git
cd aplikasiMachLearn
```

### 2. Buat virtual environment (rekomendasi)

```bash
# Linux / macOS
python -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Jalankan aplikasi Streamlit

```bash
streamlit run app.py
```

Aplikasi otomatis terbuka di browser pada `http://localhost:8501`.
Kalau tidak otomatis, buka URL tersebut secara manual.

Jika port 8501 sedang dipakai:

```bash
streamlit run app.py --server.port 8502
```

### 5. (Opsional) Jalankan notebook eksperimen

```bash
jupyter notebook Pembangunan_Model_Preprocessing.ipynb         # Cardiovascular
jupyter notebook xgboost_pembangunan_model_preprocessing.ipynb # Diabetes
```

## 🤖 Model Klasifikasi Tersedia

Halaman cardiovascular mendukung 5 model untuk perbandingan:

| No | Model | Deskripsi |
|----|-------|-----------|
| 1 | Logistic Regression | Model linear |
| 2 | Decision Tree | Pohon keputusan tunggal |
| 3 | **Extra Trees Classifier** ⭐ | Base method cardiovascular |
| 4 | W-KNN | KNN dengan bobot jarak |
| 5 | XGBoost | Gradient boosting |

Halaman diabetes menggunakan **XGBoost** sebagai base method dengan
analisis preprocessing 60 kombinasi.

## ❤️ Halaman Cardiovascular

Dataset `cardio_train.csv` berisi data medis 70.000 pasien:

| Kolom | Tipe | Keterangan |
|-------|------|------------|
| `age` | int | umur dalam **hari** (dikonversi ke tahun) |
| `gender` | int | 1 = perempuan, 2 = laki-laki |
| `height` | int | tinggi badan (cm) |
| `weight` | float | berat badan (kg) |
| `ap_hi` | int | tekanan darah sistolik |
| `ap_lo` | int | tekanan darah diastolik |
| `cholesterol` | int | 1/2/3 (normal / di atas / jauh di atas) |
| `gluc` | int | 1/2/3 (normal / di atas / jauh di atas) |
| `smoke` | int | 0/1 |
| `alco` | int | 0/1 |
| `active` | int | 0/1 |
| `cardio` | int | **target** 0/1 |

Pipeline: drop id → konversi umur → filter outlier fisiologis (BP, height,
weight) → feature engineering (`bmi`, `pulse_pressure`, `map_pressure`,
`bp_category`, `age_group`) → encoding → scaling → resampling.

Eksperimen lengkap (60 kombinasi: 5 model × 3 scaler × 4 resampler) +
5-fold CV + RandomizedSearchCV ada di
`Pembangunan_Model_Preprocessing.ipynb`.

## 🩸 Halaman Diabetes

Dua varian halaman diabetes:

- **XGBoost (praktikum)** — fokus pada deployment cepat dengan model
  yang sudah dilatih (`models/diabetes_xgb_praktikum.joblib`).
- **XGBoost (eksperimen)** — analisis preprocessing 60 kombinasi,
  perbandingan metrik, feature importance, dan ROC curve.

## 🛠️ Troubleshooting

- **`ModuleNotFoundError: streamlit`** → pastikan virtual environment
  aktif dan `pip install -r requirements.txt` sudah dijalankan.
- **`ValueError: No samples will be generated...`** pada notebook → ini
  normal pada data sudah seimbang; loop sudah punya fallback otomatis.
- **App lambat saat training cardio** → di sidebar halaman cardiovascular
  ada slider "Ukuran sampel data" — turunkan untuk eksplorasi cepat.
- **Port 8501 sudah dipakai** → jalankan
  `streamlit run app.py --server.port 8502`.
- **Model diabetes joblib tidak ditemukan** → notebook diabetes akan
  otomatis melatih ulang dan menyimpan model ke `models/` saat pertama
  kali dijalankan.
