"""
Halaman Deteksi Stroke
Dataset: healthcare-dataset-stroke-data.csv
Memprediksi risiko stroke berdasarkan data klinis pasien.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

# Tambahkan parent directory ke path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.preprocessing import load_and_clean_stroke_data, prepare_stroke_data
from utils.models import (
    AVAILABLE_MODELS, get_model, train_and_evaluate, get_model_display_name
)


def get_dataset_path():
    """Mendapatkan path dataset stroke."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, "dataset", "healthcare-dataset-stroke-data.csv")


@st.cache_data
def load_data():
    """Load dan cache dataset."""
    filepath = get_dataset_path()
    return load_and_clean_stroke_data(filepath)


@st.cache_resource
def train_model(model_key, scaler_type='standard', apply_smote=True):
    """Train dan cache model."""
    filepath = get_dataset_path()
    X_train, X_test, y_train, y_test, scaler, label_encoders, feature_names = prepare_stroke_data(
        filepath, scaler_type=scaler_type, apply_smote=apply_smote
    )

    model = get_model(model_key)
    metrics = train_and_evaluate(model, X_train, X_test, y_train, y_test)

    return model, scaler, label_encoders, feature_names, metrics, X_test, y_test


def show():
    st.markdown("# 🧠 Deteksi Risiko Stroke")
    st.markdown(
        "Prediksi risiko stroke berdasarkan data klinis pasien menggunakan "
        "algoritma klasifikasi Machine Learning."
    )
    st.markdown("---")

    # Tabs untuk navigasi dalam halaman
    tab1, tab2, tab3, tab4 = st.tabs(["🔮 Prediksi", "📊 Evaluasi Model", "📈 Eksplorasi Data", "🔬 Analisis Pemilihan Model"])

    # Sidebar untuk pemilihan model
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Pengaturan Model")

    # Model terbaik (default) - berdasarkan hasil eksperimen notebook
    best_model_key = "logistic_regression"
    st.sidebar.success(f"🏆 Model Terbaik: **{get_model_display_name(best_model_key)}**")

    # Dropdown pemilihan model
    selected_model_name = st.sidebar.selectbox(
        "Pilih Model Klasifikasi:",
        list(AVAILABLE_MODELS.keys()),
        index=0,  # Default: Logistic Regression (model terbaik)
        help="Pilih algoritma klasifikasi yang ingin digunakan untuk prediksi."
    )
    selected_model_key = AVAILABLE_MODELS[selected_model_name]

    # Opsi preprocessing
    scaler_type = st.sidebar.selectbox(
        "Metode Scaling:",
        ["minmax", "standard", "robust"],
        index=0,  # Default: MinMaxScaler (kombinasi terbaik dari eksperimen)
        help="MinMaxScaler (terbaik), StandardScaler, atau RobustScaler"
    )

    apply_smote = st.sidebar.checkbox("Terapkan SMOTE (Handle Imbalance)", value=True)

    # Train model
    with st.spinner(f"Melatih model {selected_model_name}..."):
        model, scaler, label_encoders, feature_names, metrics, X_test, y_test = train_model(
            selected_model_key, scaler_type, apply_smote
        )

    # ==================== TAB 1: PREDIKSI ====================
    with tab1:
        st.markdown("### 📝 Input Data Pasien")
        st.markdown("Masukkan data pasien di bawah ini untuk memprediksi risiko stroke.")

        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            age = st.number_input("Usia (tahun)", min_value=0.0, max_value=120.0, value=45.0, step=1.0)
            hypertension = st.selectbox("Hipertensi", [0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak")
            heart_disease = st.selectbox("Penyakit Jantung", [0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak")

        with col2:
            ever_married = st.selectbox("Pernah Menikah", ["Yes", "No"])
            work_type = st.selectbox(
                "Tipe Pekerjaan",
                ["Private", "Self-employed", "Govt_job", "children", "Never_worked"]
            )
            residence_type = st.selectbox("Tipe Tempat Tinggal", ["Urban", "Rural"])

        with col3:
            avg_glucose_level = st.number_input(
                "Rata-rata Kadar Glukosa",
                min_value=50.0, max_value=300.0, value=100.0, step=0.1
            )
            bmi = st.number_input(
                "BMI (Body Mass Index)",
                min_value=10.0, max_value=100.0, value=28.0, step=0.1
            )
            smoking_status = st.selectbox(
                "Status Merokok",
                ["never smoked", "formerly smoked", "smokes", "Unknown"]
            )

        st.markdown("---")

        # Tombol prediksi
        if st.button("🔍 Prediksi Risiko Stroke", type="primary", use_container_width=True):
            # Encode input
            input_data = pd.DataFrame({
                'gender': [gender],
                'age': [age],
                'hypertension': [hypertension],
                'heart_disease': [heart_disease],
                'ever_married': [ever_married],
                'work_type': [work_type],
                'Residence_type': [residence_type],
                'avg_glucose_level': [avg_glucose_level],
                'bmi': [bmi],
                'smoking_status': [smoking_status],
            })

            # Encode kategorikal
            categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
            for col in categorical_cols:
                le = label_encoders[col]
                try:
                    input_data[col] = le.transform(input_data[col])
                except ValueError:
                    # Jika value tidak dikenal, gunakan value pertama
                    input_data[col] = 0

            # Scaling
            input_scaled = scaler.transform(input_data)

            # Prediksi
            prediction = model.predict(input_scaled)[0]
            try:
                proba = model.predict_proba(input_scaled)[0]
            except Exception:
                proba = None

            # Tampilkan hasil
            st.markdown("### 📋 Hasil Prediksi")

            if prediction == 1:
                st.error("⚠️ **BERISIKO STROKE**")
                st.markdown(
                    "Berdasarkan data yang dimasukkan, model memprediksi bahwa pasien "
                    "**berisiko mengalami stroke**. Segera konsultasikan dengan dokter."
                )
            else:
                st.success("✅ **TIDAK BERISIKO STROKE**")
                st.markdown(
                    "Berdasarkan data yang dimasukkan, model memprediksi bahwa pasien "
                    "**tidak berisiko mengalami stroke**. Tetap jaga pola hidup sehat."
                )

            if proba is not None:
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Probabilitas Tidak Stroke", f"{proba[0]*100:.2f}%")
                with col_b:
                    st.metric("Probabilitas Stroke", f"{proba[1]*100:.2f}%")

            st.info(f"🤖 Model yang digunakan: **{selected_model_name}**")

    # ==================== TAB 2: EVALUASI MODEL ====================
    with tab2:
        st.markdown(f"### 📊 Evaluasi Model: {selected_model_name}")

        # Metrik utama
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
        with col2:
            st.metric("Precision", f"{metrics['precision']*100:.2f}%")
        with col3:
            st.metric("Recall", f"{metrics['recall']*100:.2f}%")
        with col4:
            st.metric("F1-Score", f"{metrics['f1_score']*100:.2f}%")
        with col5:
            if metrics['roc_auc'] is not None:
                st.metric("ROC AUC", f"{metrics['roc_auc']*100:.2f}%")
            else:
                st.metric("ROC AUC", "N/A")

        st.markdown("---")

        # Confusion Matrix
        st.markdown("#### 🔢 Confusion Matrix")
        cm = metrics['confusion_matrix']
        cm_df = pd.DataFrame(
            cm,
            index=["Actual: No Stroke", "Actual: Stroke"],
            columns=["Predicted: No Stroke", "Predicted: Stroke"]
        )
        st.dataframe(cm_df, use_container_width=True)

        # Classification Report
        st.markdown("#### 📄 Classification Report")
        report = metrics['classification_report']
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.round(4), use_container_width=True)

        st.markdown("---")

        # Perbandingan semua model
        st.markdown("### 🏆 Perbandingan Semua Model")

        if st.button("🔄 Bandingkan Semua Model", use_container_width=True):
            comparison_results = []

            progress_bar = st.progress(0)
            for i, (name, key) in enumerate(AVAILABLE_MODELS.items()):
                with st.spinner(f"Melatih {name}..."):
                    _, _, _, _, m, _, _ = train_model(key, scaler_type, apply_smote)
                    comparison_results.append({
                        "Model": name,
                        "Accuracy": f"{m['accuracy']*100:.2f}%",
                        "Precision": f"{m['precision']*100:.2f}%",
                        "Recall": f"{m['recall']*100:.2f}%",
                        "F1-Score": f"{m['f1_score']*100:.2f}%",
                        "ROC AUC": f"{m['roc_auc']*100:.2f}%" if m['roc_auc'] else "N/A",
                    })
                progress_bar.progress((i + 1) / len(AVAILABLE_MODELS))

            comparison_df = pd.DataFrame(comparison_results)
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    # ==================== TAB 3: EKSPLORASI DATA ====================
    with tab3:
        st.markdown("### 📈 Eksplorasi Dataset Stroke")

        df = load_data()

        # Info dataset
        st.markdown("#### 📋 Informasi Dataset")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Jumlah Data", df.shape[0])
        with col2:
            st.metric("Jumlah Fitur", df.shape[1])
        with col3:
            stroke_pct = (df['stroke'].sum() / len(df)) * 100
            st.metric("Persentase Stroke", f"{stroke_pct:.2f}%")

        st.markdown("---")

        # Preview data
        st.markdown("#### 👀 Preview Data")
        st.dataframe(df.head(20), use_container_width=True)

        # Distribusi target
        st.markdown("#### 🎯 Distribusi Target (Stroke)")
        target_dist = df['stroke'].value_counts()
        col1, col2 = st.columns(2)
        with col1:
            st.bar_chart(target_dist)
        with col2:
            st.markdown(f"""
            - **Tidak Stroke (0)**: {target_dist[0]} data ({target_dist[0]/len(df)*100:.2f}%)
            - **Stroke (1)**: {target_dist[1]} data ({target_dist[1]/len(df)*100:.2f}%)

            Dataset ini sangat **imbalanced** (tidak seimbang), sehingga teknik
            SMOTE digunakan untuk menyeimbangkan data training.
            """)

        # Statistik deskriptif
        st.markdown("#### 📊 Statistik Deskriptif")
        st.dataframe(df.describe().round(2), use_container_width=True)

    # ==================== TAB 4: ANALISIS PEMILIHAN MODEL ====================
    with tab4:
        st.markdown("### 🔬 Analisis Pemilihan Model Terbaik")
        st.markdown("---")

        st.markdown("""
        #### 🏆 Mengapa Logistic Regression Dipilih Sebagai Model Terbaik?

        Berdasarkan **60 eksperimen** yang dilakukan dengan kombinasi 5 model × 3 scaler × 4 resampler,
        hasil menunjukkan bahwa **Logistic Regression** dengan **MinMaxScaler + SMOTE** menghasilkan
        performa terbaik berdasarkan metrik **F1-Score**.

        ---

        #### 📊 Hasil Eksperimen Model Terbaik

        | Metrik | Nilai |
        |--------|-------|
        | **Model** | Logistic Regression |
        | **Scaler** | MinMaxScaler |
        | **Resampler** | SMOTE |
        | **Accuracy** | 0.7407 |
        | **Precision** | 0.1356 |
        | **Recall** | 0.8000 |
        | **F1-Score** | 0.2319 |
        | **AUC-ROC** | 0.8383 |

        ---

        #### ❌ Mengapa Bukan Random Forest?

        Meskipun **Random Forest** merupakan model yang sangat populer dan powerful untuk klasifikasi,
        pada kasus prediksi stroke ini Random Forest **tidak dipilih** sebagai model terbaik karena
        beberapa alasan berikut:

        1. **Bias terhadap kelas mayoritas**: Random Forest cenderung menghasilkan **Accuracy tinggi**
           (>93%) namun dengan **Recall sangat rendah** terhadap kelas stroke (minoritas). Artinya,
           model ini sering gagal mendeteksi pasien yang sebenarnya berisiko stroke.

        2. **F1-Score lebih rendah**: Dalam konteks imbalanced classification, F1-Score adalah metrik
           yang lebih relevan dibanding Accuracy. Random Forest menghasilkan F1-Score yang lebih rendah
           dibanding Logistic Regression pada dataset ini.

        3. **Overfitting pada data training**: Random Forest dengan 100 trees cenderung overfitting
           pada data yang sudah di-resample (SMOTE), sehingga generalisasi ke data test kurang optimal
           untuk kelas minoritas.

        4. **Trade-off Precision vs Recall**: Untuk konteks **screening medis**, kita lebih mengutamakan
           **Recall tinggi** (meminimalkan False Negative) daripada Precision tinggi. Lebih baik
           "salah mendeteksi" seseorang berisiko stroke (False Positive) daripada "melewatkan" pasien
           yang benar-benar berisiko (False Negative). Logistic Regression memberikan Recall **0.80**
           yang jauh lebih baik.

        ---

        #### ✅ Keunggulan Logistic Regression pada Kasus Ini

        1. **Recall tertinggi (0.80)**: Mampu mendeteksi 80% pasien yang benar-benar berisiko stroke.
           Ini sangat penting dalam konteks medis.

        2. **AUC-ROC tinggi (0.8383)**: Menunjukkan kemampuan diskriminasi yang baik antara kelas
           positif dan negatif.

        3. **Interpretabilitas**: Logistic Regression memberikan koefisien yang mudah diinterpretasi,
           sehingga dokter dapat memahami faktor-faktor yang berkontribusi terhadap prediksi.

        4. **Stabilitas**: Model linear cenderung lebih stabil dan tidak mudah overfitting pada
           dataset kecil atau imbalanced.

        5. **Kalibrasi probabilitas**: Logistic Regression menghasilkan probabilitas yang terkalibrasi
           dengan baik, sehingga output probabilitas dapat dipercaya untuk pengambilan keputusan klinis.

        ---

        #### 📋 Perbandingan Random Forest vs Logistic Regression

        | Aspek | Random Forest | Logistic Regression |
        |-------|---------------|---------------------|
        | Accuracy | ✅ Tinggi (~93%) | ⚠️ Lebih rendah (~74%) |
        | Recall (Stroke) | ❌ Rendah (~16-20%) | ✅ Tinggi (~80%) |
        | F1-Score | ❌ Lebih rendah | ✅ Lebih tinggi (0.2319) |
        | AUC-ROC | ⚠️ Kompetitif | ✅ Tertinggi (0.8383) |
        | Interpretabilitas | ❌ Black-box | ✅ Transparan |
        | Risiko Medis | ❌ Banyak False Negative | ✅ Meminimalkan False Negative |

        ---

        #### 🎯 Kesimpulan

        > Dalam konteks **screening medis** untuk prediksi stroke, **Logistic Regression** dengan
        > **MinMaxScaler + SMOTE** adalah pilihan terbaik karena mengutamakan **Recall** (sensitivitas)
        > yang tinggi. Meskipun Random Forest memiliki Accuracy lebih tinggi, metrik tersebut
        > menyesatkan pada dataset yang sangat imbalanced (rasio 1:19). Model yang "akurat" tapi
        > gagal mendeteksi pasien berisiko stroke justru berbahaya dalam konteks klinis.
        >
        > **Kualitas preprocessing memiliki pengaruh yang setara atau bahkan lebih besar dibandingkan
        > pemilihan algoritma.** Dataset yang sama dengan preprocessing berbeda dapat menghasilkan
        > perbedaan performa hingga 20-30%.
        """)

