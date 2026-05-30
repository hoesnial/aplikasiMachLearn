"""Halaman XGBoost untuk klasifikasi diabetes.

Halaman ini difokuskan pada hasil model XGBoost, evaluasi performa,
dan prediksi pasien baru untuk kebutuhan presentasi akademik.
"""

from __future__ import annotations

import os
import sys

import pandas as pd
import streamlit as st
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(page_title="Deteksi Diabetes XGBoost", page_icon="🩸", layout="wide")

from utils.diabetes_xgb_praktikum import (  # noqa: E402
    DATASET_PATH,
    MODEL_PATH,
    build_preprocessor,
    evaluate_pipeline,
    get_best_model,
    load_diabetes_data,
    load_or_train_diabetes_model,
    plot_confusion_matrix,
    predict_diabetes,
    split_feature_types,
    validate_dataset,
    validate_uploaded_file,
)


@st.cache_data
def load_data() -> pd.DataFrame:
    """Load dataset diabetes untuk tab Dataset."""
    return load_diabetes_data()


def _safe_read_uploaded_dataset(uploaded_file) -> pd.DataFrame | None:
    """Baca dataset upload dengan validasi dasar."""
    if uploaded_file is None:
        return None

    validate_uploaded_file(getattr(uploaded_file, "name", None))
    try:
        uploaded_df = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Gagal membaca file upload: {exc}")
        return None

    if uploaded_df.empty:
        st.error("Dataset upload kosong.")
        return None

    try:
        validate_dataset(uploaded_df, ["diabetes"])
    except ValueError as exc:
        st.error(str(exc))
        return None

    return uploaded_df


def _dataset_tab(df: pd.DataFrame) -> None:
    st.markdown("### 📈 Eksplorasi Dataset Diabetes")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Jumlah Data", f"{df.shape[0]:,}")
    with col2:
        st.metric("Jumlah Fitur", f"{df.shape[1] - 1}")
    with col3:
        diabetes_pct = df["diabetes"].mean() * 100 if "diabetes" in df.columns else 0.0
        st.metric("Persentase Diabetes", f"{diabetes_pct:.2f}%")

    st.markdown("#### 👀 Preview Data")
    st.dataframe(df.head(20), width="stretch", hide_index=True)

    st.markdown("#### 🎯 Distribusi Target")
    target_counts = df["diabetes"].value_counts().sort_index()
    target_counts.index = ["Tidak Diabetes" if idx == 0 else "Diabetes" for idx in target_counts.index]

    dist_col1, dist_col2 = st.columns(2)
    with dist_col1:
        st.bar_chart(target_counts)
    with dist_col2:
        total = len(df)
        st.markdown(
            "\n".join(
                [
                    f"- **{label}**: {count:,} data ({count / total * 100:.2f}%)"
                    for label, count in target_counts.items()
                ]
                + [
                    "",
                    "Distribusi target pada dataset diabetes ini masih cukup seimbang, sehingga model XGBoost bisa dilatih tanpa penanganan imbalance tambahan yang agresif.",
                ]
            )
        )

    st.markdown("#### 📊 Statistik Deskriptif")
    st.dataframe(df.describe().round(2), width="stretch")


def _evaluation_tab(bundle: dict) -> None:
    best_model = get_best_model(bundle)
    metrics = best_model["best_model_metrics"]
    confusion_matrix_values = best_model["best_confusion_matrix"]
    report_df = pd.DataFrame(best_model["best_classification_report"]).transpose().round(4)
    selected_dataset_result = bundle.get(best_model["best_result_key"], {})
    selected_dataset_df = selected_dataset_result.get("dataframe", load_data())

    st.markdown("### 📊 Evaluasi Model: XGBoost")
    st.caption(f"Binary classification: 0 = tidak diabetes, 1 = diabetes. Dataset terbaik: {best_model['best_result_key']}")

    metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)
    with metric_col1:
        st.metric("Accuracy", f"{metrics['accuracy'] * 100:.2f}%")
    with metric_col2:
        st.metric("Precision", f"{metrics['precision'] * 100:.2f}%")
    with metric_col3:
        st.metric("Recall", f"{metrics['recall'] * 100:.2f}%")
    with metric_col4:
        st.metric("F1-Score", f"{metrics['f1_score'] * 100:.2f}%")
    with metric_col5:
        st.metric("ROC AUC", f"{metrics['roc_auc'] * 100:.2f}%" if metrics.get("roc_auc") is not None else "N/A")

    st.markdown("---")
    st.markdown("#### 🔢 Confusion Matrix")
    cm_df = pd.DataFrame(
        confusion_matrix_values,
        index=["Actual: Tidak Diabetes", "Actual: Diabetes"],
        columns=["Predicted: Tidak Diabetes", "Predicted: Diabetes"],
    )
    st.dataframe(cm_df, width="stretch")

    st.markdown("#### 📄 Classification Report")
    st.dataframe(report_df, width="stretch")

    st.markdown("---")
    st.markdown("#### 📉 Visualisasi Confusion Matrix")
    st.pyplot(plot_confusion_matrix(confusion_matrix_values, "Confusion Matrix XGBoost"), clear_figure=True)

    st.markdown("---")
    st.markdown("### 🏆 Perbandingan Semua Model")

    if st.button("🔄 Bandingkan Semua Model", use_container_width=True):
        with st.spinner("Menjalankan perbandingan model..."):
            comparison_results = []
            progress_bar = st.progress(0)
            status_placeholder = st.empty()

            model_variants = [
                ("Logistic Regression", LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs", random_state=42), True),
                ("Decision Tree", DecisionTreeClassifier(random_state=42), False),
                ("Extra Trees", ExtraTreesClassifier(n_estimators=200, random_state=42, n_jobs=-1), False),
                ("W-KNN (Weighted KNN)", KNeighborsClassifier(n_neighbors=5, weights="distance"), True),
                ("XGBoost", None, False),
            ]

            feature_columns = [column for column in selected_dataset_df.columns if column != "diabetes"]
            numeric_columns, categorical_columns = split_feature_types(selected_dataset_df[feature_columns])
            X = selected_dataset_df.drop(columns=["diabetes"])
            y = selected_dataset_df["diabetes"].astype(int)

            from sklearn.model_selection import train_test_split

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42,
                stratify=y,
            )

            for index, (model_name, model_instance, needs_scaling) in enumerate(model_variants):
                status_placeholder.info(f"Melatih {model_name}...")

                if model_name == "XGBoost":
                    from utils.diabetes_xgb_praktikum import build_model

                    positive_count = int((y_train == 1).sum())
                    negative_count = int((y_train == 0).sum())
                    scale_pos_weight = (negative_count / positive_count) if positive_count > 0 else 1.0
                    model_instance = build_model(random_state=42, scale_pos_weight=scale_pos_weight)

                preprocessor = build_preprocessor(
                    numeric_columns,
                    categorical_columns,
                    apply_scaling=needs_scaling,
                    encoding="onehot" if model_name in {"Logistic Regression", "W-KNN (Weighted KNN)"} else "ordinal",
                )
                pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model_instance)])
                metrics_result = evaluate_pipeline(pipeline.fit(X_train, y_train), X_test, y_test)

                comparison_results.append(
                    {
                        "Model": model_name,
                        "Accuracy": f"{metrics_result['accuracy'] * 100:.2f}%",
                        "Precision": f"{metrics_result['precision'] * 100:.2f}%",
                        "Recall": f"{metrics_result['recall'] * 100:.2f}%",
                        "F1-Score": f"{metrics_result['f1_score'] * 100:.2f}%",
                        "ROC AUC": f"{metrics_result['roc_auc'] * 100:.2f}%" if metrics_result.get("roc_auc") is not None else "N/A",
                    }
                )
                progress_bar.progress((index + 1) / len(model_variants))

            status_placeholder.empty()
            st.toast("Perbandingan model selesai", icon="✅")
            comparison_df = pd.DataFrame(comparison_results)
            st.dataframe(comparison_df, width="stretch", hide_index=True)


def _prediction_tab(bundle: dict, df: pd.DataFrame) -> None:
    st.markdown("### 📝 Input Data Pasien")
    st.markdown("Masukkan data pasien untuk memprediksi risiko diabetes menggunakan model XGBoost.")

    col1, col2, col3 = st.columns(3)
    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        age = st.number_input("Usia", min_value=0.0, max_value=120.0, value=45.0, step=1.0)
        hypertension = st.selectbox("Hipertensi", [0, 1], format_func=lambda value: "Ya" if value == 1 else "Tidak")
    with col2:
        heart_disease = st.selectbox("Penyakit Jantung", [0, 1], format_func=lambda value: "Ya" if value == 1 else "Tidak")
        smoking_history = st.selectbox("Riwayat Merokok", [str(value) for value in df["smoking_history"].dropna().unique().tolist()])
        bmi = st.number_input("BMI", min_value=10.0, max_value=70.0, value=25.0, step=0.1)
    with col3:
        hba1c_level = st.number_input("HbA1c Level", min_value=3.0, max_value=15.0, value=5.7, step=0.1)
        blood_glucose_level = st.number_input("Blood Glucose Level", min_value=50, max_value=400, value=120, step=1)

    st.markdown("---")

    if st.button("🔍 Prediksi Risiko Diabetes", type="primary", use_container_width=True):
        input_data = {
            "gender": gender,
            "age": age,
            "hypertension": hypertension,
            "heart_disease": heart_disease,
            "smoking_history": smoking_history,
            "bmi": bmi,
            "HbA1c_level": hba1c_level,
            "blood_glucose_level": blood_glucose_level,
        }

        prediction, probability = predict_diabetes(bundle, input_data)
        st.markdown("### 📋 Hasil Prediksi")
        if prediction == 1:
            st.error("⚠️ **Terdeteksi Risiko Diabetes**")
            st.markdown(
                "Pasien diprediksi berisiko diabetes dan memerlukan perhatian lebih lanjut."
            )
        else:
            st.success("✅ **Tidak Terdeteksi Risiko Diabetes**")
            st.markdown(
                "Pasien diprediksi tidak berisiko diabetes berdasarkan model XGBoost."
            )

        prob_col1, prob_col2 = st.columns(2)
        with prob_col1:
            st.metric("Probabilitas Tidak Diabetes", f"{probability[0] * 100:.2f}%")
        with prob_col2:
            st.metric("Probabilitas Diabetes", f"{probability[1] * 100:.2f}%")

        st.info(f"Model yang digunakan: **{best_model_name_from_bundle(bundle)}**")


def _analysis_tab(bundle: dict) -> None:
    best_model = get_best_model(bundle)
    final_dataset_table = bundle.get("final_comparison_table", pd.DataFrame()).copy()
    final_model_table = bundle.get("final_model_comparison_table", pd.DataFrame()).copy()

    st.markdown("### 🔬 Analisis Pemilihan Model Terbaik")
    st.markdown("---")

    st.markdown(
        """
        #### 📐 Metodologi Eksperimen
        Pemilihan model pada halaman ini tetap difokuskan ke keluarga XGBoost, tetapi dibandingkan pada dua versi dataset:
        **tanpa penghapusan outlier** dan **dengan penghapusan outlier**. Setelah itu, model default dan model tuned
        XGBoost dievaluasi untuk memastikan varian yang paling stabil ditampilkan di halaman presentasi.

        Kriteria evaluasi:
        1. **ROC AUC** — kemampuan membedakan pasien diabetes dan non-diabetes.
        2. **F1-Score** — keseimbangan precision dan recall.
        3. **Accuracy** — proporsi prediksi benar secara keseluruhan.
        4. **Stabilitas hasil** — konsistensi performa pada dataset terbaik.
        """
    )

    if not final_dataset_table.empty:
        st.markdown("---")
        st.markdown("#### 🏆 Perbandingan Dataset untuk XGBoost")
        dataset_view = final_dataset_table[["dataset", "accuracy", "precision", "recall", "f1_score", "roc_auc"]].copy()
        for column in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
            dataset_view[column] = (dataset_view[column] * 100).round(2)
        st.dataframe(dataset_view, width="stretch", hide_index=True)

    if not final_model_table.empty:
        st.markdown("---")
        st.markdown("#### 🧪 Perbandingan Varian XGBoost")
        xgb_only = final_model_table[final_model_table["model"].str.contains("XGBoost", na=False)].copy()
        if xgb_only.empty:
            xgb_only = final_model_table.copy()
        for column in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
            if column in xgb_only.columns:
                xgb_only[column] = (xgb_only[column] * 100).round(2)
        st.dataframe(xgb_only, width="stretch", hide_index=True)

    st.markdown("---")
    st.markdown("#### ✅ Kesimpulan Pemilihan")
    st.markdown(
        f"""
        - Model yang dipakai di halaman ini adalah **{best_model['best_model_name']}**.
        - Dataset terbaik yang dipilih oleh pipeline adalah **{best_model['best_result_key']}**.
        - Tujuan halaman ini adalah menampilkan hasil XGBoost secara rapi untuk presentasi, bukan membandingkan banyak algoritma.
        - Struktur tab dibuat mengikuti pola halaman kardiovaskular agar alurnya lebih mudah dipresentasikan.
        """
    )


def best_model_name_from_bundle(bundle: dict) -> str:
    return get_best_model(bundle)["best_model_name"]


def show() -> None:
    """Render halaman XGBoost diabetes."""
    st.markdown("# 🩸 Deteksi Diabetes (XGBoost Binary Classification)")
    st.markdown(
        "Prediksi apakah pasien mengalami **diabetes** (1) atau **tidak** (0) berdasarkan gender, usia, hipertensi, penyakit jantung, riwayat merokok, BMI, HbA1c, dan kadar glukosa darah. Dataset: `diabetes_dataset.csv`."
    )
    st.markdown("---")
    st.caption("Halaman ini disusun dengan pola presentasi yang sama seperti halaman kardiovaskular agar alurnya konsisten saat demo.")

    uploaded_file = st.sidebar.file_uploader("Upload CSV untuk validasi", type=["csv"])

    uploaded_df = _safe_read_uploaded_dataset(uploaded_file)
    df = uploaded_df if uploaded_df is not None else load_data()
    bundle = load_or_train_diabetes_model(apply_scaling=False, use_smote=False)
    best_model = get_best_model(bundle)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Ringkasan Model")
    st.sidebar.success(f"Model terbaik: {best_model['best_model_name']}")
    st.sidebar.caption(f"Accuracy: {best_model['best_accuracy'] * 100:.2f}%")
    st.sidebar.caption(f"F1-score: {best_model['best_f1_score'] * 100:.2f}%")
    st.sidebar.caption(f"ROC AUC: {best_model['best_roc_auc'] * 100:.2f}%")
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Dataset: {os.path.basename(DATASET_PATH)}")
    st.sidebar.caption(f"Model tersimpan: {os.path.basename(MODEL_PATH)}")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🔮 Prediksi",
        "📊 Evaluasi Model",
        "📈 Eksplorasi Data",
        "🔬 Analisis Pemilihan Model",
    ])

    with tab1:
        _prediction_tab(bundle, df)

    with tab2:
        _evaluation_tab(bundle)

    with tab3:
        _dataset_tab(df)

    with tab4:
        _analysis_tab(bundle)
