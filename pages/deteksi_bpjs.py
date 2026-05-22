"""
Halaman Deteksi BPJS
Dataset: (Akan ditambahkan)
"""

import streamlit as st


def show():
    st.markdown("# 🏥 Deteksi BPJS")
    st.markdown("---")

    st.warning("🔜 **Halaman ini sedang dalam pengembangan.**")

    st.markdown("""
    ### 📋 Informasi Halaman

    Halaman ini akan digunakan untuk analisis dan prediksi terkait data BPJS
    berdasarkan dataset yang akan ditambahkan oleh anggota kelompok.

    ### 🎯 Rencana Implementasi
    - **Dataset**: BPJS (akan diupload)
    - **Target**: Prediksi klasifikasi terkait BPJS
    - **Metode**: Klasifikasi (Logistic Regression, Decision Tree, Random Forest, W-KNN, XGBoost)

    ### 📝 Langkah Selanjutnya
    1. Upload dataset ke folder `dataset/`
    2. Implementasi preprocessing sesuai karakteristik dataset
    3. Training model dan evaluasi
    4. Pembuatan form input prediksi
    """)

    st.info("Silakan tambahkan dataset dan implementasi untuk halaman ini.")
