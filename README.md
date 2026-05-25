# 🏥 Aplikasi Deteksi Penyakit - Machine Learning

Aplikasi web berbasis Streamlit untuk mendeteksi/memprediksi berbagai penyakit
menggunakan algoritma klasifikasi Machine Learning.

Halaman utama yang aktif saat ini adalah **Deteksi Penyakit Kardiovaskular**
dengan base method **Extra Trees Classifier**.

## 📁 Struktur Project

```
aplikasiMachLearn/
├── app.py                                  # Entry point aplikasi Streamlit
├── requirements.txt                        # Dependencies
├── README.md                               # Dokumentasi
├── Pembangunan_Model_Preprocessing.ipynb   # Notebook eksperimen (60 kombinasi + tuning)
├── dataset/
│   ├── cardio_train.csv                    # Dataset cardiovascular (70.000 baris)
│   └── data_balita.csv                     # (legacy) dataset stunting
├── pages/
│   ├── __init__.py
│   ├── beranda.py                          # Halaman beranda
│   ├── deteksi_cardiovascular.py           # Deteksi kardiovaskular (AKTIF)
│   ├── deteksi_pasien_treatment.py
│   ├── deteksi_bpjs.py
│   ├── deteksi_jantung.py
│   ├── deteksi_liver.py
│   └── deteksi_lainnya.py
├── utils/
│   ├── __init__.py
│   ├── preprocessing.py                    # Pipeline preprocessing dataset cardio
│   └── models.py                           # Wrapper 5 model klasifikasi
└── models/
    └── __init__.py
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

Aplikasi akan otomatis terbuka di browser pada `http://localhost:8501`.
Kalau tidak otomatis, buka URL tersebut secara manual.

### 5. (Opsional) Jalankan notebook eksperimen

```bash
jupyter notebook Pembangunan_Model_Preprocessing.ipynb
# atau
jupyter lab
```

Notebook berisi pipeline lengkap: preprocessing, feature engineering, eksperimen
60 kombinasi (5 model × 3 scaler × 4 resampler), 5-fold cross validation,
hyperparameter tuning, dan analisis pemilihan model terbaik.

## 🤖 Model Klasifikasi Tersedia

| No | Model | Deskripsi |
|----|-------|-----------|
| 1 | Logistic Regression | Model linear untuk klasifikasi |
| 2 | Decision Tree | Model berbasis pohon keputusan |
| 3 | **Extra Trees Classifier** ⭐ | Base method (sesuai instruksi tugas) — ensemble pohon yang sangat acak |
| 4 | W-KNN | KNN dengan bobot jarak (weighted) |
| 5 | XGBoost | Gradient boosting yang powerful |

## 📋 Halaman Deteksi

1. **Deteksi Penyakit Kardiovaskular** ✅ - Dataset `cardio_train.csv` (70.000 pasien)
2. **Pasien Treatment** 🔜 - Dataset belum ditambahkan
3. **BPJS** 🔜 - Dataset belum ditambahkan
4. **Penyakit Jantung** 🔜 - Dataset belum ditambahkan
5. **Penyakit Liver** 🔜 - Dataset belum ditambahkan
6. **Lainnya** 🔜 - Dataset belum ditentukan

## ❤️ Tentang Dataset Cardiovascular (Binary Classification)

Dataset `cardio_train.csv` berisi data medis pasien dengan kolom:

| Kolom | Tipe | Keterangan |
|-------|------|------------|
| `id` | int | identifier (di-drop) |
| `age` | int | umur dalam **hari** (dikonversi ke tahun) |
| `gender` | int | 1 = perempuan, 2 = laki-laki |
| `height` | int | tinggi badan (cm) |
| `weight` | float | berat badan (kg) |
| `ap_hi` | int | tekanan darah sistolik |
| `ap_lo` | int | tekanan darah diastolik |
| `cholesterol` | int | 1 = normal, 2 = di atas normal, 3 = jauh di atas normal |
| `gluc` | int | 1 = normal, 2 = di atas normal, 3 = jauh di atas normal |
| `smoke` | int | 0/1 (perokok) |
| `alco` | int | 0/1 (konsumsi alkohol) |
| `active` | int | 0/1 (aktif berolahraga) |
| `cardio` | int | **target** 0/1 |

Pipeline preprocessing: drop id → konversi umur → filter outlier fisiologis
(BP, height, weight) → feature engineering (`bmi`, `pulse_pressure`,
`map_pressure`, `bp_category`, `age_group`) → encoding → scaling → resampling.

## 🧪 Tahap Pipeline yang Diimplementasi

1. **Data Collection** — load `cardio_train.csv` (70.000 baris).
2. **Data Preprocessing** — drop id, konversi umur ke tahun, filter outlier fisiologis (BP, height, weight), deduplikasi.
3. **EDA** — distribusi target, histogram per kelas, boxplot tekanan darah dan berat badan.
4. **Feature Engineering** — `bmi`, `pulse_pressure`, `map_pressure`, `bp_category`, `age_group`, plus encoding kategorikal.
5. **Split Data** — 80/20 stratified.
6. **Model Training** — 5 algoritma × 3 scaler × 4 resampler = 60 kombinasi.
7. **Evaluation** — Accuracy, Precision, Recall, F1, ROC AUC + 5-fold cross validation.
8. **Hyperparameter Tuning** — RandomizedSearchCV pada Extra Trees Classifier (base method).
9. **Interpretation** — feature importance, confusion matrix, ROC curve, analisis komparatif lima model.

## 📝 Cara Menambahkan Dataset Baru

1. Letakkan file CSV di folder `dataset/`
2. Buat fungsi preprocessing di `utils/preprocessing.py`
3. Implementasikan halaman di `pages/deteksi_xxx.py`
4. Ikuti pola yang sudah ada di `deteksi_cardiovascular.py`

## 🛠️ Troubleshooting

- **`ModuleNotFoundError: streamlit`** → pastikan virtual environment aktif dan `pip install -r requirements.txt` sudah dijalankan.
- **`ValueError: No samples will be generated...`** pada notebook → ini normal pada data sudah seimbang; loop sudah punya fallback otomatis.
- **App lambat saat training** → di sidebar halaman cardiovascular ada slider "Ukuran sampel data" — turunkan untuk eksplorasi cepat.
- **Port 8501 sudah dipakai** → jalankan `streamlit run app.py --server.port 8502`.
