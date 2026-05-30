"""
Halaman Beranda - Menampilkan informasi umum tentang aplikasi.
"""

import streamlit as st


def show():
    st.markdown('<p class="main-header">🏥 Aplikasi Deteksi Penyakit</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Sistem Prediksi Penyakit Berbasis Machine Learning</p>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    # Deskripsi aplikasi
    st.markdown("""
    ## 👋 Selamat Datang!

    Aplikasi ini merupakan implementasi dari tugas besar **Praktikum Machine Learning** yang bertujuan
    untuk membangun sistem deteksi/prediksi penyakit menggunakan berbagai algoritma klasifikasi.

    ### 🎯 Fitur Utama
    - **Multi-Dataset**: Setiap halaman menggunakan dataset yang berbeda sesuai fokus anggota kelompok
    - **Multi-Model**: Tersedia 5 algoritma klasifikasi yang bisa dipilih dan dibandingkan
    - **Interaktif**: Input data pasien secara langsung untuk mendapatkan prediksi real-time
    - **Evaluasi Model**: Menampilkan metrik evaluasi (Accuracy, Precision, Recall, F1-Score, ROC AUC)
    """)

    st.markdown("---")

    # Daftar halaman deteksi
    st.markdown("### 📋 Daftar Halaman Deteksi")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        #### ❤️ Deteksi Penyakit Kardiovaskular
        Dataset: `cardio_train.csv` (70.000 pasien)
        - Memprediksi risiko penyakit kardiovaskular berdasarkan umur,
          tekanan darah, kolesterol, BMI, dan gaya hidup
        - Base method: **Extra Trees Classifier**
        - **Status: ✅ Aktif**
        """)

        st.markdown("""
        #### 🩸 Deteksi Diabetes (XGBoost)
        Dataset: `diabetes_dataset.csv`
        - Klasifikasi diabetes dengan model XGBoost terlatih
        - Base method: **XGBoost** (precomputed model)
        - **Status: ✅ Aktif**
        """)

    with col2:
        st.markdown("""
        #### 🩸 Deteksi Diabetes (Eksperimen)
        Dataset: `diabetes_dataset.csv`
        - Eksperimen 60 kombinasi preprocessing + XGBoost
        - Base method: **XGBoost**
        - **Status: ✅ Aktif**
        """)

        st.markdown("""
        #### 💊 Deteksi Pasien Treatment
        Dataset: `patient_treatment.csv`
        - Memprediksi kebutuhan treatment pasien
        - Base method: **Logistic Regression**
        - **Status: ✅ Aktif**
        """)

    with col3:
        st.markdown("""
        #### 🏥 Deteksi Scabies
        Dataset: `scabies-sapi-diperluas.csv`
        - Klasifikasi scabies pada sapi
        - Base method: **Decision Tree**
        - **Status: ✅ Aktif**
        """)

        st.markdown("""
        #### 🧪 Deteksi Penyakit Ginjal Kronik
        Dataset: `penyakit_ginjal_kronik.csv` (400 pasien)
        - Memprediksi CKD berbasis 24 fitur klinis (urin, darah, riwayat)
        - Base method: **W-KNN (Weighted K-Nearest Neighbors)**
        - **Status: ✅ Aktif**
        """)

    st.markdown("---")

    # Metode yang digunakan
    st.markdown("### 🤖 Algoritma Klasifikasi yang Digunakan")

    methods_col1, methods_col2 = st.columns(2)

    with methods_col1:
        st.markdown("""
        | No | Algoritma | Deskripsi |
        |----|-----------|-----------|
        | 1 | **Logistic Regression** | Model linear untuk klasifikasi biner |
        | 2 | **Decision Tree** | Model berbasis pohon keputusan |
        | 3 | **Random Forest** | Ensemble dari banyak decision tree |
        """)

    with methods_col2:
        st.markdown("""
        | No | Algoritma | Deskripsi |
        |----|-----------|-----------|
        | 4 | **W-KNN** | KNN dengan bobot jarak (weighted) |
        | 5 | **XGBoost** | Gradient boosting yang powerful |
        """)

    st.markdown("---")

    # Info kelompok
    st.markdown("### 👥 Anggota Kelompok")
    st.info("Informasi anggota kelompok akan ditambahkan di sini.")
