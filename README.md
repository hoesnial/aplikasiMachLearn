# 🏥 Aplikasi Deteksi Penyakit - Machine Learning

Aplikasi web berbasis Streamlit untuk mendeteksi/memprediksi berbagai penyakit menggunakan algoritma klasifikasi Machine Learning.

## 📁 Struktur Project

```
aplikasiMachLearn/
├── app.py                          # Entry point aplikasi Streamlit
├── requirements.txt                # Dependencies
├── README.md                       # Dokumentasi
├── dataset/                        # Folder kumpulan dataset
│   └── healthcare-dataset-stroke-data.csv
├── pages/                          # Halaman-halaman aplikasi
│   ├── __init__.py
│   ├── beranda.py                  # Halaman beranda
│   ├── deteksi_stroke.py           # Deteksi stroke (AKTIF)
│   ├── deteksi_pasien_treatment.py # Deteksi pasien treatment
│   ├── deteksi_bpjs.py             # Deteksi BPJS
│   ├── deteksi_jantung.py          # Deteksi penyakit jantung
│   ├── deteksi_liver.py            # Deteksi penyakit liver
│   └── deteksi_lainnya.py          # Halaman tambahan
├── utils/                          # Modul utilitas
│   ├── __init__.py
│   ├── preprocessing.py            # Fungsi preprocessing data
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
2. **Pasien Treatment** 🔜 - Dataset belum ditambahkan
3. **BPJS** 🔜 - Dataset belum ditambahkan
4. **Penyakit Jantung** 🔜 - Dataset belum ditambahkan
5. **Penyakit Liver** 🔜 - Dataset belum ditambahkan
6. **Lainnya** 🔜 - Dataset belum ditentukan

## 📝 Cara Menambahkan Dataset Baru

1. Letakkan file CSV di folder `dataset/`
2. Buat fungsi preprocessing di `utils/preprocessing.py`
3. Implementasikan halaman di `pages/deteksi_xxx.py`
4. Ikuti pola yang sudah ada di `deteksi_stroke.py`
