"""Halaman praktikum XGBoost untuk klasifikasi diabetes."""

from __future__ import annotations

import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.diabetes_xgb_praktikum import (  # noqa: E402
    DATASET_PATH,
    MODEL_PATH,
    dataframe_to_csv_bytes,
    figure_to_png_bytes,
    analyze_imbalance,
    interpret_confusion_matrix,
    interpret_feature_importance,
    load_diabetes_data,
    load_or_train_diabetes_model,
    plot_class_countplot,
    plot_accuracy_comparison,
    plot_boxplot,
    plot_confusion_matrix,
    plot_correlation_heatmap,
    plot_feature_importance,
    plot_numeric_histograms,
    plot_roc_curve,
    predict_diabetes,
    summarize_class_distribution,
    validate_dataset,
    validate_uploaded_file,
    get_best_model,
)


@st.cache_data
def load_data() -> pd.DataFrame:
    """Load dataset diabetes untuk eksplorasi."""
    return load_diabetes_data()


@st.cache_resource
def get_bundle(apply_scaling: bool, use_smote: bool) -> dict:
    """Load model yang sudah dilatih atau latih ulang jika perlu."""
    return load_or_train_diabetes_model(apply_scaling=apply_scaling, use_smote=use_smote)


def _safe_read_uploaded_dataset(uploaded_file) -> pd.DataFrame | None:
    """Baca file upload dengan validasi dasar."""
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


def _format_metric_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def _final_comparison_display_table(bundle: dict) -> pd.DataFrame:
    table = bundle["final_comparison_table"].copy()
    for column in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
        table[column] = (table[column] * 100).round(2)
    table.columns = ["Dataset", "Accuracy", "Precision", "Recall", "F1-score", "ROC AUC"]
    return table


def _final_model_comparison_display_table(bundle: dict) -> pd.DataFrame:
    table = bundle["final_model_comparison_table"].copy()
    for column in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
        table[column] = (table[column] * 100).round(2)
    table.columns = ["Model", "Accuracy", "Precision", "Recall", "F1-score", "ROC AUC"]
    return table
def _render_banner(bundle: dict) -> None:
    st.markdown(
        """
        <div style="
            background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 55%, #38bdf8 100%);
            color: white;
            padding: 24px 26px;
            border-radius: 22px;
            box-shadow: 0 16px 40px rgba(15, 23, 42, 0.22);
            margin-bottom: 18px;
        ">
            <div style="font-size: 2rem; font-weight: 800; margin-bottom: 8px;">🩸 Deteksi Diabetes dengan XGBoost</div>
            <div style="font-size: 1rem; line-height: 1.7; opacity: 0.95;">
                Alur praktikum machine learning lengkap: cek missing value, imputasi, encoding categorical,
                deteksi dan penghapusan outlier dengan IQR, training dua dataset, evaluasi model, visualisasi,
                dan kesimpulan otomatis.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left_col, right_col, extra_col = st.columns([1, 1, 1.3])
    with left_col:
        st.metric("Dataset Asli", f"{bundle['raw_data'].shape[0]} baris")
    with right_col:
        st.metric("Dataset Setelah Outlier Removal", f"{bundle['outlier_removed_data'].shape[0]} baris")
    with extra_col:
        best_result = get_best_model(bundle)
        st.metric("Model Terbaik", best_result["best_dataset_name"])


def _highlight_best_row(frame: pd.DataFrame):
    """Sorot baris dengan accuracy terbaik."""
    best_index = frame["accuracy"].idxmax()

    def style_row(row: pd.Series):
        if row.name == best_index:
            return ["background-color: #dcfce7; font-weight: 700;"] * len(row)
        return [""] * len(row)

    display_frame = frame.copy()
    for column in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
        display_frame[column] = (display_frame[column] * 100).round(2)

    return display_frame.style.apply(style_row, axis=1)


def _render_preprocessing_summary(bundle: dict) -> None:
    st.markdown("### 🧼 Preprocessing Summary")
    summary_col1, summary_col2 = st.columns(2)

    with summary_col1:
        st.markdown("#### Missing Value")
        st.dataframe(bundle["missing_summary"].round(2), use_container_width=True)

    with summary_col2:
        st.markdown("#### IQR Outlier Summary")
        st.dataframe(bundle["outlier_summary"].round(3), use_container_width=True)

    st.info(bundle["scaling_note"])

    st.markdown("#### Analisis Imbalance")
    imbalance_col1, imbalance_col2 = st.columns(2)
    with imbalance_col1:
        st.dataframe(bundle["class_distribution_table"], use_container_width=True, hide_index=True)
    with imbalance_col2:
        st.pyplot(plot_class_countplot(bundle["imputed_data"], "Distribusi Kelas Diabetes"), clear_figure=True)

    imbalance = bundle["imbalance_summary"]
    if imbalance["is_imbalanced"]:
        st.warning(
            f"Dataset terdeteksi imbalanced dengan rasio {imbalance['imbalance_ratio']:.2f}. "
            f"SMOTE {'diaktifkan' if bundle.get('use_smote') else 'tidak diaktifkan'} pada eksperimen ini."
        )
    else:
        st.success(f"Distribusi kelas relatif seimbang dengan rasio {imbalance['imbalance_ratio']:.2f}.")


def _render_data_exploration(bundle: dict) -> None:
    raw_data = bundle["raw_data"]
    imputed_data = bundle["imputed_data"]
    numeric_columns = bundle["no_outlier_result"]["numeric_columns"]

    st.markdown("### 📈 Eksplorasi Data")
    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
    with metrics_col1:
        st.metric("Jumlah Kolom", raw_data.shape[1])
    with metrics_col2:
        st.metric("Jumlah Fitur", len(numeric_columns))
    with metrics_col3:
        diabetes_rate = raw_data["diabetes"].mean() * 100
        st.metric("Persentase Diabetes", f"{diabetes_rate:.2f}%")

    st.markdown("#### Preview Data")
    st.dataframe(raw_data.head(20), use_container_width=True)

    st.markdown("#### Distribusi Kelas")
    st.dataframe(bundle["class_distribution_table"], use_container_width=True, hide_index=True)
    st.pyplot(plot_class_countplot(imputed_data, "Countplot Distribusi Diabetes"), clear_figure=True)

    st.markdown("#### Histogram Distribusi Data")
    st.pyplot(plot_numeric_histograms(imputed_data, numeric_columns, "Histogram Distribusi Fitur Numerik"), clear_figure=True)

    st.markdown("#### Correlation Heatmap")
    st.pyplot(plot_correlation_heatmap(imputed_data, numeric_columns, "Correlation Heatmap"), clear_figure=True)

    st.markdown("#### Boxplot Sebelum dan Sesudah Outlier Removal")
    box_col1, box_col2 = st.columns(2)
    with box_col1:
        st.pyplot(plot_boxplot(imputed_data, numeric_columns, "Boxplot Dataset Tanpa Penghapusan Outlier"), clear_figure=True)
    with box_col2:
        st.pyplot(plot_boxplot(bundle["outlier_removed_data"], numeric_columns, "Boxplot Dataset Dengan Penghapusan Outlier"), clear_figure=True)


def _render_training_comparison(bundle: dict) -> None:
    comparison_table_raw = bundle["comparison_table"].copy()
    comparison_table = _final_comparison_display_table(bundle)
    no_outlier_result = bundle["no_outlier_result"]
    outlier_removed_result = bundle["outlier_removed_result"]

    st.markdown("### 🧪 Training Comparison")
    st.dataframe(comparison_table, use_container_width=True, hide_index=True)
    st.pyplot(plot_accuracy_comparison(comparison_table_raw), clear_figure=True)

    # Tampilkan hasil cross-validation jika tersedia
    if any(col.startswith("cv_") for col in comparison_table_raw.columns):
        st.markdown("#### Cross-Validation (StratifiedKFold) - Rata-rata skor")
        cv_cols = [col for col in comparison_table_raw.columns if col.startswith("cv_") and col.endswith("_mean")]
        display_cv = comparison_table_raw[["dataset"] + cv_cols].copy()
        # ubah nama kolom lebih ramah
        display_cv.columns = ["Dataset"] + [col.replace("cv_", "").replace("_mean", " (mean)") for col in cv_cols]
        st.dataframe(display_cv, use_container_width=True, hide_index=True)

    st.markdown("#### Ringkasan Data Train/Test")
    table_col1, table_col2 = st.columns(2)
    with table_col1:
        st.markdown("**Tanpa Penghapusan Outlier**")
        st.dataframe(
            pd.DataFrame(
                {
                    "Split": ["Train", "Test"],
                    "Rows": [no_outlier_result["train_shape"][0], no_outlier_result["test_shape"][0]],
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
    with table_col2:
        st.markdown("**Dengan Penghapusan Outlier**")
        st.dataframe(
            pd.DataFrame(
                {
                    "Split": ["Train", "Test"],
                    "Rows": [outlier_removed_result["train_shape"][0], outlier_removed_result["test_shape"][0]],
                }
            ),
            use_container_width=True,
            hide_index=True,
        )


def _render_model_comparison(bundle: dict) -> None:
    st.markdown("### 🤝 Perbandingan Metode Klasifikasi")
    st.caption("Baseline Logistic Regression digunakan sebagai pembanding sederhana terhadap XGBoost.")
    model_table = _final_model_comparison_display_table(bundle)
    st.dataframe(model_table, use_container_width=True, hide_index=True)

    fig = plot_accuracy_comparison(
        pd.DataFrame(
            {
                "dataset": model_table["Model"],
                "accuracy": model_table["Accuracy"] / 100,
            }
        )
    )
    st.pyplot(fig, clear_figure=True)

    tuned_params = bundle["tuned_xgb_search"]["best_params"]
    st.markdown("#### Parameter Terbaik Hasil Tuning")
    st.json(tuned_params)


def _render_evaluation(bundle: dict) -> None:
    no_outlier_result = bundle["no_outlier_result"]
    outlier_removed_result = bundle["outlier_removed_result"]
    best_dataset_name = bundle[bundle["best_result_key"]]["dataset_name"]

    st.markdown("### 📊 Evaluasi Model")

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    with metric_col1:
        st.metric(
            "Accuracy Delta",
            f"{(outlier_removed_result['metrics']['accuracy'] - no_outlier_result['metrics']['accuracy']) * 100:.2f}%",
        )
    with metric_col2:
        st.metric(
            "F1 Delta",
            f"{(outlier_removed_result['metrics']['f1_score'] - no_outlier_result['metrics']['f1_score']) * 100:.2f}%",
        )
    with metric_col3:
        st.metric(
            "ROC AUC Delta",
            f"{(outlier_removed_result['metrics']['roc_auc'] - no_outlier_result['metrics']['roc_auc']) * 100:.2f}%",
        )
    with metric_col4:
        st.metric("Best Dataset", best_dataset_name)

    st.markdown("#### Confusion Matrix")
    cm_col1, cm_col2 = st.columns(2)
    with cm_col1:
        st.markdown("**Tanpa Penghapusan Outlier**")
        st.pyplot(
            plot_confusion_matrix(
                no_outlier_result["metrics"]["confusion_matrix"],
                "Confusion Matrix - Tanpa Penghapusan Outlier",
            ),
            clear_figure=True,
        )
    with cm_col2:
        st.markdown("**Dengan Penghapusan Outlier**")
        st.pyplot(
            plot_confusion_matrix(
                outlier_removed_result["metrics"]["confusion_matrix"],
                "Confusion Matrix - Dengan Penghapusan Outlier",
            ),
            clear_figure=True,
        )

    st.markdown("#### Interpretasi Confusion Matrix")
    confusion_summary = bundle["best_confusion_summary"]
    st.write(
        f"True Positive = {confusion_summary['true_positive']}, True Negative = {confusion_summary['true_negative']}, "
        f"False Positive = {confusion_summary['false_positive']}, False Negative = {confusion_summary['false_negative']}."
    )
    st.info(bundle["best_confusion_interpretation"])

    st.markdown("#### ROC Curve")
    roc_col1, roc_col2 = st.columns(2)
    with roc_col1:
        st.pyplot(
            plot_roc_curve(
                no_outlier_result["y_test"],
                no_outlier_result["y_prob"],
                "ROC Curve - Tanpa Penghapusan Outlier",
            ),
            clear_figure=True,
        )
    with roc_col2:
        st.pyplot(
            plot_roc_curve(
                outlier_removed_result["y_test"],
                outlier_removed_result["y_prob"],
                "ROC Curve - Dengan Penghapusan Outlier",
            ),
            clear_figure=True,
        )

    st.markdown("#### Tabel Evaluasi")
    evaluation_table = _final_comparison_display_table(bundle)
    st.dataframe(evaluation_table, use_container_width=True, hide_index=True)

    st.markdown("#### Classification Report")
    report_col1, report_col2 = st.columns(2)
    with report_col1:
        st.markdown("**Tanpa Penghapusan Outlier**")
        st.dataframe(
            pd.DataFrame(no_outlier_result["metrics"]["classification_report"]).transpose().round(4),
            use_container_width=True,
        )
    with report_col2:
        st.markdown("**Dengan Penghapusan Outlier**")
        st.dataframe(
            pd.DataFrame(outlier_removed_result["metrics"]["classification_report"]).transpose().round(4),
            use_container_width=True,
        )

    st.markdown("#### Perbandingan Baseline vs XGBoost")
    st.dataframe(_final_model_comparison_display_table(bundle), use_container_width=True, hide_index=True)


def _render_feature_importance(bundle: dict) -> None:
    feature_importance_df = bundle["best_feature_importance"].copy()

    st.markdown("### 🔎 Feature Importance")
    if feature_importance_df.empty:
        st.info("Feature importance tidak tersedia untuk baseline Logistic Regression. Bagian ini ditampilkan untuk model XGBoost.")
        return

    top_five = feature_importance_df.head(5).copy()
    top_five["importance_pct"] = (top_five["importance"] * 100).round(2)
    st.dataframe(top_five[["feature", "importance_pct"]], use_container_width=True, hide_index=True)
    st.pyplot(
        plot_feature_importance(feature_importance_df, f"Feature Importance - {bundle['best_model_name']}"),
        clear_figure=True,
    )

    st.markdown("#### Interpretasi Akademik Sederhana")
    for item in bundle["best_feature_interpretation"][:5]:
        st.write(f"- {item}")


def _render_exports(bundle: dict) -> None:
    st.markdown("### 📦 Export Hasil")

    export_col1, export_col2, export_col3 = st.columns(3)
    with export_col1:
        st.download_button(
            label="Download Tabel Evaluasi CSV",
            data=dataframe_to_csv_bytes(_final_comparison_display_table(bundle)),
            file_name="diabetes_evaluation_summary.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with export_col2:
        report_df = pd.DataFrame(bundle["best_metrics"]["classification_report"]).transpose().round(4)
        st.download_button(
            label="Download Classification Report CSV",
            data=dataframe_to_csv_bytes(report_df),
            file_name="diabetes_classification_report.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with export_col3:
        fig = plot_confusion_matrix(bundle["best_metrics"]["confusion_matrix"], "Confusion Matrix - Export")
        st.download_button(
            label="Download Confusion Matrix PNG",
            data=figure_to_png_bytes(fig),
            file_name="diabetes_confusion_matrix.png",
            mime="image/png",
            use_container_width=True,
        )


def _render_conclusion(bundle: dict) -> None:
    no_outlier_accuracy = bundle["no_outlier_result"]["metrics"]["accuracy"]
    outlier_removed_accuracy = bundle["outlier_removed_result"]["metrics"]["accuracy"]
    accuracy_delta = outlier_removed_accuracy - no_outlier_accuracy

    st.markdown("### 🧾 Kesimpulan Otomatis")

    if accuracy_delta > 0:
        st.success(bundle["conclusion"])
    elif accuracy_delta < 0:
        st.warning(bundle["conclusion"])
    else:
        st.info(bundle["conclusion"])

    st.markdown("#### Kesimpulan Akhir")
    conclusion_points = [
        f"Model terbaik keseluruhan: {bundle['best_model_name']}",
        f"Dataset terbaik untuk XGBoost: {bundle[bundle['best_result_key']]['dataset_name']}",
        f"Pengaruh preprocessing: {'membantu' if accuracy_delta > 0 else 'memberi perubahan terbatas' if accuracy_delta == 0 else 'tidak selalu meningkatkan performa'}.",
        f"Pengaruh outlier: {'penghapusan outlier memperbaiki performa' if accuracy_delta > 0 else 'XGBoost cukup tahan terhadap outlier' if accuracy_delta <= 0 else ''}",
        "Pengaruh cross validation: hasil CV membantu memeriksa stabilitas model di beberapa fold data.",
        "Alasan memilih XGBoost: model tree-based ini kuat pada data tabular dan tidak terlalu sensitif terhadap scaling.",
    ]
    for item in conclusion_points:
        st.write(f"- {item}")

    conclusion_table = pd.DataFrame(
        {
            "Metric": ["Accuracy", "Precision", "Recall", "F1-score", "ROC AUC"],
            "Tanpa Outlier Removal": [
                no_outlier_accuracy,
                bundle["no_outlier_result"]["metrics"]["precision"],
                bundle["no_outlier_result"]["metrics"]["recall"],
                bundle["no_outlier_result"]["metrics"]["f1_score"],
                bundle["no_outlier_result"]["metrics"]["roc_auc"],
            ],
            "Dengan Outlier Removal": [
                outlier_removed_accuracy,
                bundle["outlier_removed_result"]["metrics"]["precision"],
                bundle["outlier_removed_result"]["metrics"]["recall"],
                bundle["outlier_removed_result"]["metrics"]["f1_score"],
                bundle["outlier_removed_result"]["metrics"]["roc_auc"],
            ],
        }
    )
    for column in ["Tanpa Outlier Removal", "Dengan Outlier Removal"]:
        conclusion_table[column] = (conclusion_table[column] * 100).round(2)

    st.dataframe(conclusion_table, use_container_width=True, hide_index=True)

    st.markdown(
        """
        <div style="
            background: linear-gradient(135deg, #eff6ff 0%, #f8fafc 100%);
            border: 1px solid #bfdbfe;
            border-radius: 16px;
            padding: 16px 18px;
            margin-top: 10px;
            color: #0f172a;
        ">
            <strong>Ringkasan Praktikum:</strong> preprocessing lengkap membantu menyiapkan data lebih bersih untuk eksperimen,
            sementara evaluasi pada dua dataset memberikan pembanding yang jelas tentang dampak penghapusan outlier terhadap performa XGBoost.
        </div>
        """,
        unsafe_allow_html=True,
    )


def _prediction_section(bundle: dict, df: pd.DataFrame) -> None:
    smoking_options = [str(value) for value in df["smoking_history"].dropna().unique().tolist()]
    preferred_order = ["No Info", "never", "former", "current", "not current", "ever"]
    ordered_smoking_options = [value for value in preferred_order if value in smoking_options]
    ordered_smoking_options.extend(value for value in smoking_options if value not in ordered_smoking_options)

    st.markdown("### 🔮 Prediksi Risiko Diabetes")
    st.markdown("Gunakan model terbaik yang dipilih otomatis dari perbandingan dua dataset.")

    with st.form("diabetes_prediction_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            age = st.number_input("Usia", min_value=0.0, max_value=120.0, value=45.0, step=1.0)
            hypertension = st.selectbox("Hipertensi", [0, 1], format_func=lambda value: "Ya" if value == 1 else "Tidak")
        with col2:
            heart_disease = st.selectbox("Penyakit Jantung", [0, 1], format_func=lambda value: "Ya" if value == 1 else "Tidak")
            smoking_history = st.selectbox("Riwayat Merokok", ordered_smoking_options)
            bmi = st.number_input("BMI", min_value=10.0, max_value=70.0, value=25.0, step=0.1)
        with col3:
            hba1c_level = st.number_input("HbA1c Level", min_value=3.0, max_value=15.0, value=5.7, step=0.1)
            blood_glucose_level = st.number_input("Blood Glucose Level", min_value=50, max_value=400, value=120, step=1)

        submitted = st.form_submit_button("🔍 Prediksi")

    if submitted:
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
        result_label = "BERISIKO DIABETES" if prediction == 1 else "TIDAK BERISIKO DIABETES"

        if prediction == 1:
            st.error(f"⚠️ {result_label}")
        else:
            st.success(f"✅ {result_label}")

        prob_non_diabetes = float(probability[0])
        prob_diabetes = float(probability[1])
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric("Probabilitas Tidak Diabetes", f"{prob_non_diabetes * 100:.2f}%")
        with metric_col2:
            st.metric("Probabilitas Diabetes", f"{prob_diabetes * 100:.2f}%")

        st.caption(
            f"Prediksi memakai model terbaik dari {bundle['best_result_key'].replace('_', ' ')}. "
            f"Model tersimpan di {MODEL_PATH}."
        )


def show() -> None:
    """Render halaman utama deteksi diabetes."""
    st.markdown("# 🩸 Deteksi Diabetes XGBoost")
    st.markdown(
        "Halaman ini mengikuti alur praktikum machine learning dengan preprocessing lengkap, "
        "handling outlier menggunakan IQR, dan perbandingan performa sebelum dan sesudah preprocessing."
    )
    st.markdown("---")

    apply_scaling = st.sidebar.checkbox(
        "Uji StandardScaler (opsional)",
        value=False,
        help="XGBoost tidak terlalu bergantung pada scaling, tetapi eksperimen ini tetap disediakan.",
    )
    use_smote = st.sidebar.checkbox(
        "Aktifkan SMOTE jika imbalance",
        value=True,
        help="Opsi ini akan aktif hanya bila dataset terdeteksi imbalance.",
    )
    uploaded_file = st.sidebar.file_uploader("Upload CSV untuk validasi", type=["csv"])
    st.sidebar.caption("StandardScaler bersifat opsional untuk model tree-based seperti XGBoost.")
    st.sidebar.caption(f"Dataset: {DATASET_PATH}")
    st.sidebar.caption(f"Model tersimpan: {MODEL_PATH}")

    uploaded_df = _safe_read_uploaded_dataset(uploaded_file)
    df = uploaded_df if uploaded_df is not None else load_data()
    bundle = get_bundle(apply_scaling, use_smote)
    best_model = get_best_model(bundle)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Ringkasan Model")
    st.sidebar.success(f"Model terbaik: {best_model['best_model_name']}")
    st.sidebar.caption(f"Dataset terbaik untuk eksperimen XGBoost: {best_model['best_dataset_name']}")
    st.sidebar.caption(f"Accuracy: {best_model['best_accuracy'] * 100:.2f}%")
    st.sidebar.caption(f"F1-score: {best_model['best_f1_score'] * 100:.2f}%")
    st.sidebar.caption(f"ROC AUC: {best_model['best_roc_auc'] * 100:.2f}%")
    st.sidebar.caption(bundle["tuning_note"])

    _render_banner(bundle)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["🔮 Prediksi", "🧼 Preprocessing", "📈 Eksplorasi Data", "📊 Evaluasi & Perbandingan", "🧠 Feature Importance", "📦 Export & Kesimpulan"]
    )

    with tab1:
        _prediction_section(bundle, df)

    with tab2:
        _render_preprocessing_summary(bundle)

    with tab3:
        _render_data_exploration(bundle)

    with tab4:
        _render_training_comparison(bundle)
        _render_model_comparison(bundle)
        _render_evaluation(bundle)

    with tab5:
        _render_feature_importance(bundle)

    with tab6:
        _render_exports(bundle)
        _render_conclusion(bundle)
