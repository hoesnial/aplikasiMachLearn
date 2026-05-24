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
│   ├── deteksi_diabetes.py         # Deteksi diabetes dengan XGBoost
│   ├── deteksi_pasien_treatment.py # Deteksi pasien treatment
│   ├── deteksi_bpjs.py             # Deteksi BPJS
│   ├── deteksi_jantung.py          # Deteksi penyakit jantung
│   ├── deteksi_liver.py            # Deteksi penyakit liver
│   └── deteksi_lainnya.py          # Halaman tambahan
├── utils/                          # Modul utilitas
│   ├── __init__.py
│   ├── preprocessing.py            # Fungsi preprocessing data
│   ├── diabetes_xgb.py             # Pipeline preprocessing + XGBoost diabetes
│   └── models.py                   # Fungsi model ML
└── models/                         # Folder untuk menyimpan model (opsional)
    └── __init__.py
```

## 🚀 Cara Menjalankan

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Jalankan aplikasi:**
   ```bash
   streamlit run app.py
   ```

3. **Buka browser** di `http://localhost:8501`

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

## 🩸 Modul XGBoost Diabetes

Modul diabetes sudah dipisahkan agar tidak mengganggu halaman lain.

- File utilitas: `utils/diabetes_xgb.py`
- File halaman: `pages/deteksi_diabetes.py`
- Dataset: `dataset/diabetes_dataset.csv`
- Training dilakukan beberapa kali dengan random_state berbeda untuk memilih model terbaik.
- Model terbaik tersimpan otomatis: `models/diabetes_xgb_model.joblib`
- Jumlah percobaan dapat diubah dari sidebar: 5x, 10x, 15x, atau 20x

Jika dependency XGBoost belum ada, install dengan:

```bash
pip install xgboost
```
