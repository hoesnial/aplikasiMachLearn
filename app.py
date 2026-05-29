"""
Aplikasi Deteksi Penyakit - Machine Learning
Menggunakan Streamlit sebagai framework web.
Setiap halaman mendeteksi penyakit berdasarkan dataset masing-masing anggota kelompok.
"""

import streamlit as st

# Konfigurasi halaman
st.set_page_config(
    page_title="Aplikasi Deteksi Penyakit - ML",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar navigasi
st.sidebar.title("🏥 Menu Navigasi")
st.sidebar.markdown("---")

# Daftar halaman
pages = {
    "🏠 Beranda": "beranda",
    "🧠 Deteksi Stroke": "stroke",
    "💊 Deteksi Pasien Treatment": "pasien_treatment",
    "🏥 Deteksi Penyakit Scabies": "scabies",
    "❤️ Deteksi Penyakit Jantung": "jantung",
    "🫁 Deteksi Penyakit Liver": "liver",
    "📋 Deteksi Lainnya": "lainnya",
}

selected_page = st.sidebar.radio(
    "Pilih Halaman:",
    list(pages.keys()),
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ Tentang Aplikasi")
st.sidebar.info(
    "Aplikasi ini menggunakan metode klasifikasi Machine Learning "
    "untuk mendeteksi berbagai penyakit berdasarkan dataset medis. "
    "Setiap halaman menggunakan dataset yang berbeda sesuai anggota kelompok."
)
st.sidebar.markdown("---")
st.sidebar.markdown("### 🤖 Model Tersedia")
st.sidebar.markdown("""
1. Logistic Regression
2. Decision Tree
3. Random Forest
4. W-KNN (Weighted KNN)
5. XGBoost
""")

# Routing halaman
page_key = pages[selected_page]

if page_key == "beranda":
    from pages import beranda
    beranda.show()
elif page_key == "stroke":
    from pages import deteksi_stroke
    deteksi_stroke.show()
elif page_key == "pasien_treatment":
    from pages import deteksi_pasien_treatment
    deteksi_pasien_treatment.show()
elif page_key == "scabies":
    from pages import deteksi_scabies
    deteksi_scabies.show()
elif page_key == "jantung":
    from pages import deteksi_jantung
    deteksi_jantung.show()
elif page_key == "liver":
    from pages import deteksi_liver
    deteksi_liver.show()
elif page_key == "lainnya":
    from pages import deteksi_lainnya
    deteksi_lainnya.show()