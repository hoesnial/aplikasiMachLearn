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
    tab1, tab2, tab3 = st.tabs(["🔮 Prediksi", "📊 Evaluasi Model", "📈 Eksplorasi Data"])

    # Sidebar untuk pemilihan model
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Pengaturan Model")

    # Model terbaik (default)
    best_model_key = "random_forest"
    st.sidebar.success(f"🏆 Model Terbaik: **{get_model_display_name(best_model_key)}**")

    # Dropdown pemilihan model
    selected_model_name = st.sidebar.selectbox(
        "Pilih Model Klasifikasi:",
        list(AVAILABLE_MODELS.keys()),
        index=2,  # Default: Random Forest
        help="Pilih algoritma klasifikasi yang ingin digunakan untuk prediksi."
    )
    selected_model_key = AVAILABLE_MODELS[selected_model_name]

    # Opsi preprocessing
    scaler_type = st.sidebar.selectbox(
        "Metode Scaling:",
        ["standard", "minmax", "robust"],
        index=0,
        help="StandardScaler (default), MinMaxScaler, atau RobustScaler"
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
