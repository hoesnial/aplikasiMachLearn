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

    # Model terbaik empiris dari eksperimen 60 kombinasi = XGBoost (ROC AUC & Accuracy tertinggi)
    # Extra Trees adalah base method sesuai instruksi tugas, bukan model terbaik empiris
    best_model_key = "xgboost"
    st.sidebar.success(
        f"🏆 Model Terbaik (empiris): **{get_model_display_name(best_model_key)}**"
    )
    st.sidebar.info(
        "📌 Base method tugas: **Extra Trees Classifier** "
        "(tersedia di dropdown untuk perbandingan)"
    )

    model_names = list(AVAILABLE_MODELS.keys())
    default_index = model_names.index("XGBoost")
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
        st.markdown("### 🔬 Analisis Pemilihan Model Terbaik")
        st.markdown("---")

        st.markdown("""
        #### 📐 Metodologi Eksperimen
        Pemilihan model dilakukan melalui eksperimen sistematis pada notebook
        `Pembangunan_Model_Preprocessing.ipynb`:
        **5 model × 3 scaler × 4 resampler = 60 kombinasi**, diikuti
        5-fold cross validation dan hyperparameter tuning.

        Kriteria evaluasi:
        1. **ROC AUC** — kemampuan diskriminasi di seluruh threshold (metrik
           utama untuk deteksi penyakit).
        2. **F1-Score** — keseimbangan Precision–Recall.
        3. **Accuracy** — proporsi prediksi benar secara keseluruhan.
        4. **Stabilitas CV** — deviasi standar kecil = generalisasi konsisten.
        """)

        st.markdown("---")
        st.markdown("#### 🏆 Hasil Eksperimen Empiris (dari tabel perbandingan)")

        # Tabel hasil aktual dari eksperimen
        empirical_data = {
            "Model": [
                "XGBoost ⭐",
                "Logistic Regression",
                "Extra Trees (base method tugas)",
                "W-KNN (Weighted KNN)",
                "Decision Tree",
            ],
            "Accuracy": ["72.93%", "72.63%", "70.33%", "69.70%", "63.23%"],
            "Precision": ["73.65%", "73.45%", "69.70%", "69.32%", "62.77%"],
            "Recall": ["71.19%", "70.66%", "71.66%", "70.39%", "64.57%"],
            "F1-Score": ["72.40%", "72.03%", "70.67%", "69.85%", "63.66%"],
            "ROC AUC": ["79.72%", "79.47%", "76.06%", "74.32%", "63.20%"],
        }
        st.dataframe(
            pd.DataFrame(empirical_data),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("---")
        st.markdown("""
        #### ✅ Mengapa XGBoost adalah model terbaik empiris?

        Dari tabel di atas, **XGBoost unggul di hampir semua metrik**:

        | Metrik | XGBoost | Extra Trees | Selisih |
        |--------|---------|-------------|---------|
        | ROC AUC | **79.72%** | 76.06% | +3.66 poin |
        | Accuracy | **72.93%** | 70.33% | +2.60 poin |
        | F1-Score | **72.40%** | 70.67% | +1.73 poin |
        | Precision | **73.65%** | 69.70% | +3.95 poin |

        **Alasan teknis XGBoost unggul pada dataset cardio ini:**

        1. **Gradient boosting membangun pohon secara sekuensial** — setiap
           pohon baru fokus memperbaiki kesalahan pohon sebelumnya. Ini
           sangat efektif pada dataset tabular medis dengan banyak fitur
           yang saling berinteraksi (umur × tekanan darah × kolesterol).

        2. **Regularisasi L1/L2 bawaan** — XGBoost punya parameter `reg_alpha`
           dan `reg_lambda` yang mencegah overfitting, sehingga generalisasi
           ke test set lebih baik dibanding Extra Trees yang tidak punya
           regularisasi eksplisit.

        3. **ROC AUC 79.72% vs 76.06%** — selisih 3.66 poin pada ROC AUC
           sangat signifikan dalam konteks medis. Artinya XGBoost jauh lebih
           baik dalam membedakan pasien cardio vs non-cardio di berbagai
           threshold probabilitas.

        4. **Dataset cardio berukuran besar (70k baris)** — XGBoost dioptimasi
           untuk dataset besar dengan implementasi yang efisien (histogram-based
           split finding), sehingga bisa mengeksploitasi lebih banyak pola.

        ---

        #### 📌 Catatan tentang Extra Trees Classifier

        Extra Trees adalah **base method sesuai instruksi tugas**, bukan
        model terbaik secara empiris pada dataset ini. Perbedaan utamanya:

        - Extra Trees memilih **split secara acak** (tidak mencari split
          terbaik), sehingga lebih cepat tapi kurang akurat.
        - Tanpa regularisasi, Extra Trees lebih rentan terhadap noise pada
          fitur yang kurang informatif (mis. `smoke`, `alco`, `active`
          yang korelasinya rendah dengan target).
        - Pada dataset cardio yang sudah seimbang 50/50, keunggulan
          Extra Trees dalam menangani imbalance tidak relevan.

        Extra Trees tetap tersedia di dropdown untuk perbandingan dan
        sebagai pemenuhan instruksi tugas.

        ---

        #### ❌ Mengapa bukan Logistic Regression?

        Logistic Regression sebenarnya sangat kompetitif (ROC AUC 79.47%,
        hampir setara XGBoost). Namun XGBoost dipilih karena:
        - ROC AUC lebih tinggi 0.25 poin
        - Precision lebih tinggi 0.20 poin
        - Mampu menangkap interaksi non-linear antar fitur secara otomatis
          tanpa perlu feature engineering manual tambahan

        ---

        #### 🛠️ Pipeline yang Diimplementasi

        1. **Data Collection** — load `cardio_train.csv` (70.000 baris).
        2. **Preprocessing** — drop id, konversi umur ke tahun, filter
           outlier fisiologis (BP, height, weight), deduplikasi.
        3. **EDA** — distribusi target, histogram per kelas, boxplot.
        4. **Feature Engineering** — `bmi`, `pulse_pressure`, `map_pressure`,
           `bp_category`, `age_group`, plus encoding.
        5. **Split Data** — 80/20 stratified.
        6. **Eksperimen** — 5 model × 3 scaler × 4 resampler = 60 kombinasi.
        7. **Evaluasi** — Accuracy, Precision, Recall, F1, ROC AUC + 5-fold CV.
        8. **Hyperparameter Tuning** — RandomizedSearchCV pada Extra Trees
           (base method tugas) dan XGBoost (model terbaik empiris).
        9. **Interpretasi** — feature importance, confusion matrix, ROC curve.

        ---

        #### 🎯 Kesimpulan

        > **XGBoost adalah model terbaik secara empiris** pada dataset
        > cardiovascular ini, unggul di ROC AUC (79.72%), Accuracy (72.93%),
        > F1-Score (72.40%), dan Precision (73.65%). **Extra Trees Classifier**
        > digunakan sebagai base method sesuai instruksi tugas dan tetap
        > tersedia untuk perbandingan, namun secara objektif performanya
        > di bawah XGBoost dan Logistic Regression pada dataset ini.
        """)
