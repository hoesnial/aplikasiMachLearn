# 🏥 Aplikasi Deteksi Penyakit - Machine Learning

Aplikasi web berbasis Streamlit untuk mendeteksi/memprediksi berbagai penyakit
menggunakan algoritma klasifikasi Machine Learning.

Halaman aktif saat ini:

| Halaman | Dataset | Base Method |
|---------|---------|-------------|
| ❤️ **Deteksi Penyakit Kardiovaskular** | `cardio_train.csv` (70.000 pasien) | **Extra Trees Classifier** |
| 🩸 **Deteksi Diabetes (XGBoost)** | `diabetes_dataset.csv` | **XGBoost** |
| 🩸 **Deteksi Diabetes (Eksperimen)** | `diabetes_dataset.csv` | XGBoost + analisis eksperimen |
| 💊 **Deteksi Pasien Treatment** | `patient_treatment.csv` | **Logistic Regression** |
| 🏥 **Deteksi Scabies** | `scabies-sapi-diperluas.csv` | **Decision Tree** |
| 🧪 **Deteksi Penyakit Ginjal Kronik** | `penyakit_ginjal_kronik.csv` (400 pasien) | **W-KNN (Weighted KNN)** |

Untuk panduan menambahkan halaman deteksi baru, lihat
[Panduan Menambah Halaman Deteksi Baru](#-panduan-menambah-halaman-deteksi-baru-untuk-developer)
di bawah.

## 📁 Struktur Project

```
aplikasiMachLearn/
├── app.py                                       # Entry point Streamlit
├── requirements.txt                             # Dependencies
├── README.md                                    # Dokumentasi
├── Pembangunan_Model_Preprocessing.ipynb        # Notebook eksperimen Cardiovascular
├── Pembangunan_Model_Preprocessing_Decission_Tree.ipynb # Notebook eksperimen Scabies (Decision Tree)
├── Pembangunan_Model_Preprocessing_CKD_WKNN.ipynb # Notebook eksperimen Ginjal Kronik (W-KNN)
├── LR_Pembangunan_Model_Preprocessing_PasienTreatment.ipynb # Notebook eksperimen Pasien Treatment (Logistic Regression)
├── xgboost_pembangunan_model_preprocessing.ipynb# Notebook eksperimen Diabetes
├── dataset/
│   ├── cardio_train.csv                         # Cardiovascular dataset (70k baris)
│   ├── diabetes_dataset.csv                     # Diabetes dataset
│   ├── patient_treatment.csv                    # Patient treatment dataset
│   ├── penyakit_ginjal_kronik.csv               # Ginjal Kronik dataset (400 baris)
│   └── scabies-sapi-diperluas.csv               # Scabies dataset
├── models/
│   ├── diabetes_xgb_model.joblib                # Model XGBoost diabetes (precomputed)
│   └── diabetes_xgb_praktikum.joblib            # Model praktikum diabetes
├── pages/
│   ├── __init__.py
│   ├── beranda.py
│   ├── deteksi_cardiovascular.py                # Aktif (Extra Trees Classifier)
│   ├── deteksi_diabetes.py                      # Aktif (XGBoost eksperimen)
│   ├── deteksi_diabetes_praktikum.py            # Aktif (XGBoost praktikum)
│   ├── deteksi_ginjal_kronik.py                 # Aktif (W-KNN)
│   ├── deteksi_pasien_treatment.py              # Aktif (Logistic Regression)
│   └── deteksi_scabies.py                       # Aktif (Decision Tree)
├── utils/
│   ├── __init__.py
│   ├── preprocessing.py                         # Pipeline cardiovascular
│   ├── models.py                                # Wrapper 5 model klasifikasi
│   ├── ckd_pipeline.py                          # Pipeline ginjal kronik
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

`utils/models.py` menyediakan 5 model siap pakai (lihat `AVAILABLE_MODELS`):

| Model key | Class | Cocok untuk |
|-----------|-------|-------------|
| `logistic_regression` | `LogisticRegression` | Baseline linear |
| `decision_tree` | `DecisionTreeClassifier` | Interpretable, baseline non-linear |
| `extra_trees` | `ExtraTreesClassifier` | Default cardiovascular ⭐ |
| `wknn` | `KNeighborsClassifier(weights='distance')` | Dataset kecil-menengah |
| `xgboost` | `XGBClassifier` | Default diabetes, biasanya juara |

Helper utama:
- `get_model(model_key)` → instance model
- `train_and_evaluate(model, X_train, X_test, y_train, y_test)` → dict
  metrik (accuracy / precision / recall / F1 / ROC AUC / confusion matrix)

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

---

## 👥 Panduan Menambah Halaman Deteksi Baru (Untuk Developer)

Aplikasi sengaja dirancang **modular** supaya tiap anggota kelompok bisa
menambah halaman deteksi sendiri tanpa konflik dengan halaman lain.
Berikut langkah lengkapnya.

### A. Konvensi & Pola

1. **Satu halaman = satu file** di folder `pages/`, nama mengikuti pola
   `deteksi_<nama_penyakit>.py` dan punya fungsi `show()`.
2. **Logika preprocessing & training** ditempatkan di `utils/` (mis.
   `utils/<dataset>_pipeline.py`) supaya halaman tetap ringan dan bisa
   diuji terpisah.
3. **Dataset** disimpan di `dataset/<nama>.csv` dan **wajib** punya satu
   kolom target biner (0/1) atau dimapping menjadi biner di
   preprocessing.
4. **Model artefak** (kalau di-precompute) disimpan di `models/` dengan
   ekstensi `.joblib`.
5. **Notebook eksperimen** boleh ditambahkan di root dengan pola
   `<dataset>_pembangunan_model_preprocessing.ipynb`.

### B. Langkah Praktis (Checklist)

#### 1. Tambahkan dataset

```bash
cp /path/ke/dataset_anda.csv dataset/<nama>.csv
```

Pastikan ada kolom target biner. Kalau target multi-kelas, mapping ke
biner di pipeline preprocessing.

#### 2. Buat modul preprocessing

Salin `utils/preprocessing.py` sebagai template, lalu buat
`utils/<nama>_pipeline.py`. Modul ini setidaknya menyediakan:

- Konstanta: `TARGET_COL`, `FEATURE_COLS`, label mapping (kalau ada)
- Fungsi `load_and_clean_<nama>_data(filepath)` — load + filter outlier
- Fungsi `prepare_<nama>_data(filepath, scaler_type, apply_smote, ...)`
  → return `(X_train, X_test, y_train, y_test, scaler, encoders, feature_names)`
- (Opsional) `transform_single_record(...)` untuk inference UI form

Pakai `sklearn.preprocessing.StandardScaler/MinMaxScaler/RobustScaler`,
`imblearn.over_sampling.SMOTE` (atau ADASYN/RandomUnderSampler) sesuai
karakteristik dataset.

#### 3. Buat halaman Streamlit

Salin `pages/deteksi_cardiovascular.py` sebagai template, lalu buat
`pages/deteksi_<nama>.py`. Polanya:

```python
"""Halaman Deteksi <Penyakit> (Binary Classification)
Dataset: <nama>.csv
Base method: <model>
"""

import os, sys
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.<nama>_pipeline import prepare_<nama>_data, ...
from utils.models import get_model, train_and_evaluate, AVAILABLE_MODELS


@st.cache_resource
def train_model(model_key, scaler_type="standard", apply_smote=False):
    X_train, X_test, y_train, y_test, *_ = prepare_<nama>_data(
        get_dataset_path(), scaler_type, apply_smote
    )
    model = get_model(model_key)
    metrics = train_and_evaluate(model, X_train, X_test, y_train, y_test)
    return model, metrics, X_test, y_test


def show():
    st.markdown("# 🏥 Deteksi <Penyakit>")
    st.markdown("---")
    # Sidebar opsi (model, scaler, SMOTE)
    # Form input pasien
    # Tombol prediksi → render metrik + hasil
```

#### 4. Daftarkan halaman di `app.py`

Tambahkan dua baris berikut:

```python
# 1) Tambah ke dict pages
pages = {
    ...
    "🏥 Deteksi <Penyakit>": "<key>",
}

# 2) Tambah cabang routing
elif page_key == "<key>":
    from pages import deteksi_<nama>
    deteksi_<nama>.show()
```

Konvensi `key` adalah lowercase tanpa spasi (mis. `stroke`, `kanker`,
`hepatitis`). Buat file baru di `pages/deteksi_<nama>.py` dan daftarkan
di `app.py` mengikuti pola halaman aktif yang sudah ada.

#### 5. (Opsional) Tambah model baru

Kalau ingin algoritma di luar 5 yang ada:

1. Tambahkan key di `AVAILABLE_MODELS` (`utils/models.py`).
2. Tambahkan instance di dict `models` dalam `get_model()`.
3. Pastikan model punya `predict_proba` atau handle fallback `roc_auc=None`.

#### 6. Notebook eksperimen (opsional tapi disarankan)

Pakai `Pembangunan_Model_Preprocessing.ipynb` sebagai template:

- 60 kombinasi (5 model × 3 scaler × 4 resampler)
- 5-fold cross validation
- RandomizedSearchCV untuk tuning
- Header tag fase pipeline (EDA / Preprocessing / Training / Evaluasi /
  Tuning) sudah disiapkan di tiap step

Simpan sebagai `<nama>_pembangunan_model_preprocessing.ipynb` di root.

### C. Checklist Sebelum Push

- [ ] `streamlit run app.py` berjalan tanpa error.
- [ ] Halaman baru muncul di sidebar dan halaman lama tetap berfungsi.
- [ ] Form input prediksi bekerja (tombol predict tidak melempar error).
- [ ] Dataset baru tidak melebihi 100 MB (kalau iya, gunakan
      Git LFS atau host eksternal).
- [ ] Tidak ada credential / data sensitif tertanam di kode.
- [ ] Pakai branch sendiri kalau perubahan besar (mis.
      `feature/deteksi-stroke`), lalu PR ke `main`.

### D. Penamaan & Style

- Bahasa Indonesia untuk teks UI dan komentar dokumentasi.
- Gunakan emoji ringan di header sidebar (❤️🩸🫁💊) supaya konsisten.
- Variabel snake_case, fungsi snake_case, konstanta UPPER_CASE.
- Hindari hardcode path absolut — selalu pakai pola
  `os.path.dirname(os.path.abspath(__file__))`.

---

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
- **`sys.path` error saat import `utils.*`** → pastikan menjalankan dari
  root project (`streamlit run app.py`), bukan dari folder `pages/`.

## 📌 Roadmap Improvement

Tertulis di `Pembangunan_Model_Preprocessing.ipynb` (sel "🗺️ Pemetaan
Step ke Tahap Pipeline ML"):

- Tambah **seleksi fitur eksplisit** (mis. `SelectKBest`,
  feature-importance threshold, drop fitur korelasi rendah).
- Bungkus preprocessing + model dalam `sklearn.Pipeline` /
  `ColumnTransformer` setelah split (best practice anti-leakage).
- Refit `LabelEncoder` hanya pada train set (saat ini di-fit pre-split
  karena kategorinya deterministik).
