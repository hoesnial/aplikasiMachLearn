"""
Halaman Deteksi Penyakit Kardiovaskular (Binary Classification)
Dataset: cardio_train.csv

Target:
    cardio = 0 -> Tidak terdeteksi penyakit kardiovaskular
    cardio = 1 -> Terdeteksi penyakit kardiovaskular

Base method utama: Extra Trees Classifier
(sesuai instruksi tugas, divalidasi pada
``Pembangunan_Model_Preprocessing.ipynb``).
"""

import os
import sys

import numpy as np
import pandas as pd
import streamlit as st

# Tambahkan parent directory ke path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.preprocessing import (  # noqa: E402
    load_and_clean_cardio_data,
    prepare_cardio_data,
    transform_single_record,
    CARDIO_TARGET_COL,
    CARDIO_BINARY_NAMES,
    CHOLESTEROL_LABELS,
    GLUC_LABELS,
    GENDER_LABELS,
)
from utils.models import (  # noqa: E402
    AVAILABLE_MODELS,
    get_model,
    train_and_evaluate,
    get_model_display_name,
)

# Untuk performa UI, batasi data training default. Set ke None untuk pakai semua.
DEFAULT_SAMPLE_SIZE = 20000


def get_dataset_path() -> str:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, "dataset", "cardio_train.csv")


@st.cache_data
def load_data() -> pd.DataFrame:
    return load_and_clean_cardio_data(get_dataset_path())


@st.cache_resource
def train_model(
    model_key: str,
    scaler_type: str = "standard",
    apply_smote: bool = False,
    sample_size: int | None = DEFAULT_SAMPLE_SIZE,
):
    filepath = get_dataset_path()
    (
        X_train, X_test, y_train, y_test,
        scaler, label_encoders, feature_names,
    ) = prepare_cardio_data(
        filepath,
        scaler_type=scaler_type,
        apply_smote=apply_smote,
        sample_size=sample_size,
    )

    model = get_model(model_key)
    metrics = train_and_evaluate(model, X_train, X_test, y_train, y_test)

    return (
        model, scaler, label_encoders, feature_names,
        metrics, X_test, y_test,
    )


def _binary_label(idx: int) -> str:
    return CARDIO_BINARY_NAMES.get(int(idx), str(idx)).title()


def show():
    st.markdown("# ❤️ Deteksi Penyakit Kardiovaskular (Binary Classification)")
    st.markdown(
        "Prediksi apakah pasien mengalami **penyakit kardiovaskular** (1) atau "
        "**tidak** (0) berdasarkan umur, jenis kelamin, tinggi/berat badan, "
        "tekanan darah, kolesterol, glukosa, serta gaya hidup. "
        "Dataset: `cardio_train.csv` (70.000 pasien)."
    )
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "🔮 Prediksi",
            "📊 Evaluasi Model",
            "📈 Eksplorasi Data",
            "🔬 Analisis Pemilihan Model",
        ]
    )

    # ===== Sidebar =====
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Pengaturan Model")

    # Default base method = Extra Trees Classifier (sesuai instruksi tugas)
    best_model_key = "extra_trees"
    st.sidebar.success(
        f"🏆 Base Method (sesuai tugas): **{get_model_display_name(best_model_key)}**"
    )

    model_names = list(AVAILABLE_MODELS.keys())
    default_index = model_names.index("Extra Trees")
    selected_model_name = st.sidebar.selectbox(
        "Pilih Model Klasifikasi:",
        model_names,
        index=default_index,
        help="Pilih algoritma klasifikasi yang ingin digunakan untuk prediksi.",
    )
    selected_model_key = AVAILABLE_MODELS[selected_model_name]

    scaler_type = st.sidebar.selectbox(
        "Metode Scaling:",
        ["standard", "minmax", "robust"],
        index=0,
        help="StandardScaler (default), MinMaxScaler, atau RobustScaler.",
    )

    apply_smote = st.sidebar.checkbox(
        "Terapkan SMOTE",
        value=False,
        help="Dataset cardio sudah seimbang ~50/50, SMOTE umumnya tidak perlu.",
    )

    sample_size = st.sidebar.slider(
        "Ukuran sampel data (training)",
        min_value=5000,
        max_value=68000,
        value=DEFAULT_SAMPLE_SIZE,
        step=5000,
        help="Membatasi data agar UI tetap responsif. Set max untuk pakai semua data.",
    )

    with st.spinner(f"Melatih model {selected_model_name}..."):
        (
            model, scaler, label_encoders, feature_names,
            metrics, X_test, y_test,
        ) = train_model(selected_model_key, scaler_type, apply_smote, sample_size)

    # ==================== TAB 1: PREDIKSI ====================
    with tab1:
        st.markdown("### 📝 Input Data Pasien")
        st.markdown(
            "Masukkan data pasien untuk memprediksi risiko penyakit kardiovaskular."
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            age_years = st.number_input(
                "Umur (tahun)", min_value=20, max_value=100, value=50, step=1,
                help="Umur pasien dalam tahun.",
            )
            gender = st.selectbox(
                "Jenis Kelamin",
                options=[1, 2],
                format_func=lambda x: GENDER_LABELS[x],
                index=1,
            )
            height = st.number_input(
                "Tinggi Badan (cm)", min_value=120, max_value=220,
                value=165, step=1,
            )
            weight = st.number_input(
                "Berat Badan (kg)", min_value=30.0, max_value=200.0,
                value=70.0, step=0.5,
            )

        with col2:
            ap_hi = st.number_input(
                "Tekanan Darah Sistolik (ap_hi)", min_value=80, max_value=250,
                value=120, step=1,
                help="Tekanan darah saat jantung berkontraksi.",
            )
            ap_lo = st.number_input(
                "Tekanan Darah Diastolik (ap_lo)", min_value=40, max_value=200,
                value=80, step=1,
                help="Tekanan darah saat jantung relaksasi.",
            )
            cholesterol = st.selectbox(
                "Kolesterol",
                options=[1, 2, 3],
                format_func=lambda x: f"{x} - {CHOLESTEROL_LABELS[x]}",
                index=0,
            )
            gluc = st.selectbox(
                "Glukosa",
                options=[1, 2, 3],
                format_func=lambda x: f"{x} - {GLUC_LABELS[x]}",
                index=0,
            )

        with col3:
            smoke = st.selectbox(
                "Perokok?",
                options=[0, 1],
                format_func=lambda x: "Tidak" if x == 0 else "Ya",
                index=0,
            )
            alco = st.selectbox(
                "Konsumsi Alkohol?",
                options=[0, 1],
                format_func=lambda x: "Tidak" if x == 0 else "Ya",
                index=0,
            )
            active = st.selectbox(
                "Aktif Berolahraga?",
                options=[0, 1],
                format_func=lambda x: "Tidak" if x == 0 else "Ya",
                index=1,
            )

        st.markdown("---")

        if st.button(
            "🔍 Prediksi Risiko Kardiovaskular",
            type="primary",
            use_container_width=True,
        ):
            if ap_hi <= ap_lo:
                st.error(
                    "Tekanan sistolik (ap_hi) harus lebih besar dari "
                    "diastolik (ap_lo). Mohon periksa kembali input."
                )
            else:
                record = {
                    "age_years": age_years,
                    "gender": gender,
                    "height": height,
                    "weight": weight,
                    "ap_hi": ap_hi,
                    "ap_lo": ap_lo,
                    "cholesterol": cholesterol,
                    "gluc": gluc,
                    "smoke": smoke,
                    "alco": alco,
                    "active": active,
                }
                input_scaled = transform_single_record(
                    record, scaler, label_encoders, feature_names,
                )

                pred = int(model.predict(input_scaled)[0])
                try:
                    proba = model.predict_proba(input_scaled)[0]
                except Exception:
                    proba = None

                st.markdown("### 📋 Hasil Prediksi")
                if pred == 1:
                    st.error("⚠️ **Terdeteksi Risiko Penyakit Kardiovaskular**")
                    st.markdown(
                        "Pasien terdeteksi berisiko mengalami penyakit "
                        "kardiovaskular. Disarankan untuk konsultasi ke "
                        "tenaga medis untuk evaluasi lebih lanjut "
                        "(EKG, profil lipid, ekokardiografi)."
                    )
                else:
                    st.success("✅ **Tidak Terdeteksi Penyakit Kardiovaskular**")
                    st.markdown(
                        "Pasien tidak terdeteksi penyakit kardiovaskular "
                        "berdasarkan model. Tetap pertahankan gaya hidup "
                        "sehat: aktif berolahraga, jaga tekanan darah, "
                        "berat badan, kolesterol, dan glukosa darah."
                    )

                if proba is not None:
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Probabilitas Tidak Cardio", f"{proba[0] * 100:.2f}%")
                    with col_b:
                        st.metric("Probabilitas Cardio", f"{proba[1] * 100:.2f}%")

                # Tampilkan fitur turunan untuk transparansi
                bmi = weight / (height / 100) ** 2
                pulse_pressure = ap_hi - ap_lo
                map_pressure = ap_lo + (ap_hi - ap_lo) / 3
                st.markdown("#### 🧮 Fitur Turunan")
                col_x, col_y, col_z = st.columns(3)
                col_x.metric("BMI", f"{bmi:.2f}")
                col_y.metric("Pulse Pressure", f"{pulse_pressure} mmHg")
                col_z.metric("MAP", f"{map_pressure:.1f} mmHg")

                st.info(f"🤖 Model yang digunakan: **{selected_model_name}**")

    # ==================== TAB 2: EVALUASI MODEL ====================
    with tab2:
        st.markdown(f"### 📊 Evaluasi Model: {selected_model_name}")
        st.caption("Binary classification: 0 = tidak cardio, 1 = cardio.")

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Accuracy", f"{metrics['accuracy'] * 100:.2f}%")
        with col2:
            st.metric("Precision", f"{metrics['precision'] * 100:.2f}%")
        with col3:
            st.metric("Recall", f"{metrics['recall'] * 100:.2f}%")
        with col4:
            st.metric("F1-Score", f"{metrics['f1_score'] * 100:.2f}%")
        with col5:
            if metrics["roc_auc"] is not None:
                st.metric("ROC AUC", f"{metrics['roc_auc'] * 100:.2f}%")
            else:
                st.metric("ROC AUC", "N/A")

        st.markdown("---")
        st.markdown("#### 🔢 Confusion Matrix")
        cm = metrics["confusion_matrix"]
        cm_df = pd.DataFrame(
            cm,
            index=["Actual: Tidak Cardio", "Actual: Cardio"],
            columns=["Predicted: Tidak Cardio", "Predicted: Cardio"],
        )
        st.dataframe(cm_df, use_container_width=True)

        st.markdown("#### 📄 Classification Report")
        report = metrics["classification_report"]
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.round(4), use_container_width=True)

        st.markdown("---")
        st.markdown("### 🏆 Perbandingan Semua Model")

        if st.button("🔄 Bandingkan Semua Model", use_container_width=True):
            comparison_results = []
            progress_bar = st.progress(0)
            for i, (name, key) in enumerate(AVAILABLE_MODELS.items()):
                with st.spinner(f"Melatih {name}..."):
                    _, _, _, _, m, _, _ = train_model(
                        key, scaler_type, apply_smote, sample_size
                    )
                    comparison_results.append(
                        {
                            "Model": name,
                            "Accuracy": f"{m['accuracy'] * 100:.2f}%",
                            "Precision": f"{m['precision'] * 100:.2f}%",
                            "Recall": f"{m['recall'] * 100:.2f}%",
                            "F1-Score": f"{m['f1_score'] * 100:.2f}%",
                            "ROC AUC": (
                                f"{m['roc_auc'] * 100:.2f}%"
                                if m["roc_auc"] else "N/A"
                            ),
                        }
                    )
                progress_bar.progress((i + 1) / len(AVAILABLE_MODELS))

            comparison_df = pd.DataFrame(comparison_results)
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    # ==================== TAB 3: EKSPLORASI DATA ====================
    with tab3:
        st.markdown("### 📈 Eksplorasi Dataset Cardiovascular")

        df = load_data()

        st.markdown("#### 📋 Informasi Dataset")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Jumlah Data (clean)", f"{df.shape[0]:,}")
        with col2:
            st.metric(
                "Jumlah Fitur Mentah",
                len([c for c in df.columns if c != CARDIO_TARGET_COL]),
            )
        with col3:
            cardio_pct = df[CARDIO_TARGET_COL].mean() * 100
            st.metric("Persentase Cardio", f"{cardio_pct:.2f}%")

        st.markdown("---")
        st.markdown("#### 👀 Preview Data")
        st.dataframe(df.head(20), use_container_width=True)

        st.markdown("#### 🎯 Distribusi Target")
        target_dist = df[CARDIO_TARGET_COL].value_counts().sort_index()
        target_dist.index = [_binary_label(i) for i in target_dist.index]

        col1, col2 = st.columns(2)
        with col1:
            st.bar_chart(target_dist)
        with col2:
            total = len(df)
            lines = []
            for label, count in target_dist.items():
                lines.append(
                    f"- **{label}**: {count:,} data ({count / total * 100:.2f}%)"
                )
            lines.append("")
            lines.append(
                "Distribusi target relatif seimbang ~50/50, sehingga "
                "resampling (SMOTE/ADASYN) bersifat opsional."
            )
            st.markdown("\n".join(lines))

        st.markdown("#### 📊 Statistik Deskriptif")
        st.dataframe(df.describe().round(2), use_container_width=True)

    # ==================== TAB 4: ANALISIS PEMILIHAN MODEL ====================
    with tab4:
        st.markdown(
            "### 🔬 Analisis Pemilihan Model: Extra Trees Classifier"
        )
        st.markdown("---")

        st.markdown(
            """
        #### 📐 Metodologi
        Pemilihan base method **Extra Trees Classifier** mengikuti
        **instruksi tugas** dan tetap divalidasi melalui eksperimen
        sistematis pada notebook `Pembangunan_Model_Preprocessing.ipynb`:
        **5 model × 3 scaler × 4 resampler = 60 kombinasi**, lalu
        5-fold cross validation pada konfigurasi terbaik dari setiap model.

        Kriteria validasi:
        1. **F1-Score** pada test set — penting pada konteks medis untuk
           menjaga keseimbangan Precision–Recall.
        2. **ROC AUC** pada test set — kemampuan diskriminasi di berbagai
           threshold.
        3. **Stabilitas 5-fold Cross Validation** — deviasi standar kecil =
           generalisasi konsisten.
        4. **Efisiensi training/inference** untuk aplikasi web.

        ---

        #### 🏆 Posisi Setiap Model pada Dataset Cardio

        - **Extra Trees Classifier (base method)**: ensemble pohon dengan
          randomisasi split tinggi. Sangat cocok untuk fitur numerik
          medis dengan boundary non-linear (mis. tekanan darah × umur ×
          BMI). Mendukung feature importance untuk interpretasi klinis.
        - **XGBoost**: gradient boosting kuat pada data tabular medis,
          biasanya bersaing ketat dengan ensemble tree-based.
        - **W-KNN**: KNN dengan bobot jarak, sensitif pada scaling dan
          inference relatif lambat pada 70k baris.
        - **Decision Tree**: tunggal, varians lebih tinggi dibanding
          ensemble.
        - **Logistic Regression**: linear, tertinggal karena interaksi
          antar fitur (umur × tekanan darah) tidak ditangkap eksplisit.

        ---

        #### ✅ Mengapa Extra Trees Classifier?

        1. **Sesuai instruksi tugas** — base method adalah **Extra Trees
           Classifier** (bukan Regressor, karena task ini binary
           classification).
        2. **Performa kompetitif** pada F1 dan ROC AUC, dengan deviasi
           standar CV yang kecil (generalisasi konsisten).
        3. **Cocok untuk relasi non-linear** Tinggi/Berat × Tekanan Darah
           × Umur, yang merupakan jantung dari prediksi cardio.
        4. **Lebih cepat dari Random Forest** karena pemilihan split
           bersifat acak, dan **lebih scalable dari W-KNN** untuk
           inference besar.
        5. **Memberikan feature importance**, mendukung interpretasi
           klinis yang penting di dunia medis.

        ---

        #### 🛠️ Tahap Pipeline yang Diimplementasi

        1. **Data Collection** — load `cardio_train.csv` (70.000 baris).
        2. **Data Preprocessing** — drop id, konversi umur ke tahun,
           filter outlier fisiologis (BP, height, weight), deduplikasi.
        3. **EDA** — distribusi target, histogram per kelas, boxplot
           tekanan darah dan berat badan.
        4. **Feature Engineering** — `bmi`, `pulse_pressure`,
           `map_pressure`, `bp_category`, `age_group`, plus encoding
           kategorikal.
        5. **Split Data** — 80/20 stratified.
        6. **Model Training** — 5 algoritma × 3 scaler × 4 resampler = 60
           kombinasi.
        7. **Evaluation** — Accuracy, Precision, Recall, F1, ROC AUC,
           plus 5-fold CV.
        8. **Hyperparameter Tuning** — RandomizedSearchCV pada Extra
           Trees Classifier (base method).
        9. **Interpretation** — feature importance, confusion matrix,
           ROC curve.

        ---

        #### 🎯 Kesimpulan

        > **Extra Trees Classifier** dipilih sebagai base method sesuai
        > instruksi tugas, dan validasi empiris memperlihatkan model ini
        > kompetitif di seluruh kriteria objektif: F1 test set, ROC AUC
        > test set, mean F1 5-fold CV dengan deviasi standar kecil, dan
        > peningkatan/konsistensi setelah hyperparameter tuning. Hasil
        > ini juga mengkonfirmasi bahwa **kualitas preprocessing dan
        > feature engineering memiliki pengaruh signifikan** terhadap
        > performa setiap algoritma — terutama filter outlier tekanan
        > darah dan fitur turunan `pulse_pressure`, `map_pressure`, dan
        > `bmi` yang sangat relevan secara klinis.
        """
        )
