"""
Halaman Deteksi Lainnya
Dataset: (Akan ditambahkan)
"""

import streamlit as st


def show():
    st.markdown("# 📋 Deteksi Lainnya")
    st.markdown("---")

    st.warning("🔜 **Halaman ini sedang dalam pengembangan.**")

    st.markdown("""
    ### 📋 Informasi Halaman

    Halaman ini disediakan untuk anggota kelompok yang datasetnya belum ditentukan.
    Akan diimplementasikan setelah dataset tersedia.

    ### 🎯 Rencana Implementasi
    - **Dataset**: (Belum ditentukan)
    - **Target**: Prediksi klasifikasi sesuai dataset
    - **Metode**: Klasifikasi (Logistic Regression, Decision Tree, Random Forest, W-KNN, XGBoost)

    ### 📝 Langkah Selanjutnya
    1. Tentukan dataset yang akan digunakan
    2. Upload dataset ke folder `dataset/`
    3. Implementasi preprocessing sesuai karakteristik dataset
    4. Training model dan evaluasi
    5. Pembuatan form input prediksi
    """)

    st.info("Silakan tentukan dataset dan tambahkan implementasi untuk halaman ini.")
