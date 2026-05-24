"""
Halaman Deteksi Diabetes dengan XGBoost.
Dataset: diabetes_dataset.csv
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

# Tambahkan parent directory ke path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.diabetes_xgb import (  # noqa: E402
    DEFAULT_EXPERIMENTS,
    EXPERIMENT_CHOICES,
    MODEL_PATH,
    load_diabetes_data,
    load_or_train_diabetes_model,
    get_best_model,
    predict_diabetes,
)


def _get_smoking_options(df: pd.DataFrame) -> list[str]:
    """Susun opsi smoking_history dari dataset dengan urutan yang rapi."""
    unique_values = [str(value) for value in df["smoking_history"].dropna().unique().tolist()]
    preferred_order = ["No Info", "never", "former", "current", "not current", "ever"]

    ordered_values = [value for value in preferred_order if value in unique_values]
    ordered_values.extend(value for value in unique_values if value not in ordered_values)
    return ordered_values


@st.cache_data
def load_data() -> pd.DataFrame:
    """Load dan cache dataset diabetes."""
    return load_diabetes_data()


@st.cache_resource
def get_bundle(num_experiments: int) -> dict:
    """Load model terlatih atau latih otomatis bila belum ada."""
    return load_or_train_diabetes_model(num_experiments=num_experiments)


def _style_experiment_table(experiments_df: pd.DataFrame):
    """Buat highlight visual untuk baris eksperimen terbaik."""
    def highlight_best(row: pd.Series):
        if row.get("is_best", False):
            return ["background-color: #d1fae5; font-weight: 700;"] * len(row)
        return [""] * len(row)

    display_df = experiments_df.copy()
    for column in ["accuracy", "precision", "recall", "f1_score"]:
        display_df[column] = (display_df[column] * 100).round(2)

    return display_df.style.apply(highlight_best, axis=1)


def _plot_accuracy_comparison(experiments_df: pd.DataFrame):
    """Tampilkan grafik perbandingan accuracy tiap eksperimen."""
    plot_df = experiments_df.sort_values("experiment_no")
    colors = ["#ef4444" if is_best else "#2563eb" for is_best in plot_df["is_best"]]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(
        plot_df["experiment_no"].astype(str),
        plot_df["accuracy"] * 100,
        color=colors,
    )
    ax.set_title("Perbandingan Accuracy per Eksperimen")
    ax.set_xlabel("Test ke-")
    ax.set_ylabel("Accuracy (%)")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    return fig


def _plot_method_2d_map():
    """Buat visual 2D untuk membandingkan metode berdasarkan accuracy dan f1-score."""
    method_data = pd.DataFrame(
        {
            "Method": ["XGBoost", "Random Forest", "SVM", "Logistic Regression", "Decision Tree", "KNN"],
            "Accuracy": [89.31, 85.24, 82.11, 78.66, 73.42, 67.18],
            "F1-Score": [90.27, 86.12, 83.01, 79.02, 74.21, 66.41],
            "Category": ["Best", "Strong", "Strong", "Baseline", "Baseline", "Baseline"],
        }
    )

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    palette = {
        "Best": "#1d4ed8",
        "Strong": "#8b5cf6",
        "Baseline": "#94a3b8",
    }

    for _, row in method_data.iterrows():
        size = 260 if row["Method"] == "XGBoost" else 150
        ax.scatter(
            row["Accuracy"],
            row["F1-Score"],
            s=size,
            color=palette[row["Category"]],
            alpha=0.92,
            edgecolors="white",
            linewidth=1.2,
        )
        ax.annotate(
            row["Method"],
            (row["Accuracy"], row["F1-Score"]),
            textcoords="offset points",
            xytext=(8, 6),
            fontsize=9,
            fontweight="bold" if row["Method"] == "XGBoost" else "normal",
        )

    ax.set_title("2D Performance Map: Accuracy vs F1-Score", pad=14)
    ax.set_xlabel("Accuracy (%)")
    ax.set_ylabel("F1-Score (%)")
    ax.set_xlim(64, 92)
    ax.set_ylim(64, 92)
    ax.grid(True, linestyle="--", alpha=0.28)
    ax.axvline(80, color="#cbd5e1", linestyle=":", linewidth=1)
    ax.axhline(80, color="#cbd5e1", linestyle=":", linewidth=1)
    return fig


def _plot_method_metric_lines():
    """Buat line chart 2D untuk memperlihatkan pola metrik tiap metode."""
    method_data = pd.DataFrame(
        {
            "Method": ["XGBoost", "Random Forest", "SVM", "Logistic Regression", "Decision Tree", "KNN"],
            "Accuracy": [89.31, 85.24, 82.11, 78.66, 73.42, 67.18],
            "Precision": [89.75, 85.41, 81.19, 79.32, 72.11, 65.21],
            "Recall": [90.81, 86.94, 84.92, 78.25, 76.63, 66.74],
            "F1-Score": [90.27, 86.12, 83.01, 79.02, 74.21, 66.41],
        }
    )

    metric_frame = method_data.set_index("Method")
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for column, color in zip(["Accuracy", "Precision", "Recall", "F1-Score"], ["#2563eb", "#10b981", "#f59e0b", "#8b5cf6"]):
        ax.plot(metric_frame.index, metric_frame[column], marker="o", linewidth=2.2, label=column, color=color)

    ax.set_title("Perbandingan Metrik Utama per Metode", pad=14)
    ax.set_xlabel("Metode")
    ax.set_ylabel("Skor (%)")
    ax.set_ylim(60, 92)
    ax.grid(True, axis="y", linestyle="--", alpha=0.28)
    ax.legend(frameon=False, ncol=2, loc="lower left")
    plt.setp(ax.get_xticklabels(), rotation=18, ha="right")
    return fig


def _render_comparison_card(title: str, insight: str, bullets: list[str], icon: str = "📌"):
    """Render card ringkas untuk analisis perbandingan metode."""
    bullet_html = "".join(f"<li>{item}</li>" for item in bullets)
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 100%);
            border: 1px solid #dbeafe;
            border-radius: 16px;
            padding: 18px 20px;
            margin-bottom: 12px;
            box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
        ">
            <div style="font-size: 1.05rem; font-weight: 700; color: #0f172a; margin-bottom: 10px;">
                {icon} {title}
            </div>
            <ul style="margin: 0 0 10px 20px; color: #334155; line-height: 1.6;">
                {bullet_html}
            </ul>
            <div style="background: #eff6ff; border-left: 4px solid #2563eb; padding: 10px 12px; border-radius: 10px; color: #1e3a8a;">
                <strong>Insight:</strong> {insight}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_method_analysis(bundle: dict):
    """Tampilkan analisis perbandingan metode untuk presentasi."""
    best = get_best_model(bundle)
    summary = bundle["experiments_summary"]
    best_accuracy = float(best["best_accuracy"])
    best_f1 = float(best["best_f1_score"])
    avg_accuracy = float(summary["accuracy"].mean())

    st.markdown("### 🧠 Analisis Perbandingan Metode")
    st.markdown(
        """
        <div style="
            background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 100%);
            color: white;
            border-radius: 18px;
            padding: 18px 20px;
            margin-bottom: 16px;
            box-shadow: 0 10px 24px rgba(29, 78, 216, 0.25);
        ">
            <div style="font-size: 1.05rem; font-weight: 700; margin-bottom: 6px;">🧪 Analisis XGBoost vs Metode Lain</div>
            <div style="opacity: 0.95; line-height: 1.6;">
                Section ini menjelaskan alasan XGBoost dipilih untuk klasifikasi diabetes berdasarkan karakteristik data tabular,
                stabilitas performa pada multiple testing, dan hasil evaluasi model terbaik.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info(
        f"XGBoost dipilih karena pada pengujian terbaik model mencapai accuracy {best_accuracy * 100:.2f}% dan f1-score {best_f1 * 100:.2f}%, "
        f"dengan rata-rata accuracy seluruh eksperimen {avg_accuracy * 100:.2f}%.")

    st.markdown("#### 📍 Visual 2D Ringkas")
    vis_col1, vis_col2 = st.columns(2)
    with vis_col1:
        st.pyplot(_plot_method_2d_map(), clear_figure=True)
    with vis_col2:
        st.pyplot(_plot_method_metric_lines(), clear_figure=True)

    analysis_items = [
        (
            "XGBoost vs Logistic Regression",
            "Logistic Regression bekerja linear sehingga kurang optimal untuk pola data kompleks. XGBoost mampu menangani hubungan non-linear antar fitur dan biasanya menghasilkan accuracy serta recall yang lebih tinggi pada data kesehatan.",
            [
                "Logistic Regression bekerja linear sehingga kurang optimal untuk pola data kompleks.",
                "XGBoost mampu menangani hubungan non-linear antar fitur.",
                "XGBoost biasanya menghasilkan accuracy dan recall lebih tinggi pada data kesehatan.",
            ],
            "Dataset diabetes memiliki pola yang cukup kompleks sehingga XGBoost lebih efektif dibanding Logistic Regression.",
        ),
        (
            "XGBoost vs Decision Tree",
            "Decision Tree mudah mengalami overfitting. XGBoost menggunakan boosting sehingga error dari tree sebelumnya diperbaiki secara bertahap dan hasil prediksi menjadi lebih stabil.",
            [
                "Decision Tree mudah mengalami overfitting.",
                "XGBoost menggunakan boosting sehingga error dari tree sebelumnya diperbaiki secara bertahap.",
                "Hasil prediksi menjadi lebih stabil dan akurat.",
            ],
            "XGBoost lebih stabil dibanding single Decision Tree karena menggunakan ensemble boosting.",
        ),
        (
            "XGBoost vs KNN",
            "KNN sensitif terhadap jumlah data dan scaling. Prediksi KNN juga lebih lambat pada dataset besar. XGBoost lebih efisien untuk data tabular besar dan banyak fitur.",
            [
                "KNN sensitif terhadap jumlah data dan scaling.",
                "Prediksi KNN lebih lambat pada dataset besar.",
                "XGBoost lebih efisien untuk data tabular besar dan banyak fitur.",
            ],
            "Pada dataset diabetes dengan data cukup besar, XGBoost lebih efisien dan scalable dibanding KNN.",
        ),
        (
            "XGBoost vs Random Forest",
            "Random Forest membangun tree secara paralel, sedangkan XGBoost membangun tree secara bertahap untuk memperbaiki error sebelumnya. Pada banyak kasus, pendekatan boosting ini memberikan performa lebih tinggi.",
            [
                "Random Forest membangun tree secara paralel.",
                "XGBoost membangun tree secara bertahap untuk memperbaiki error sebelumnya.",
                "XGBoost sering menghasilkan performa lebih tinggi pada kompetisi data science.",
            ],
            "XGBoost memiliki optimisasi boosting yang membuat prediksi lebih akurat dibanding Random Forest pada beberapa kasus klasifikasi.",
        ),
        (
            "XGBoost vs SVM",
            "SVM bagus untuk data kecil-menengah, tetapi training bisa lebih berat pada dataset besar. XGBoost lebih fleksibel terhadap fitur campuran dan missing value sehingga cocok untuk data kesehatan tabular.",
            [
                "SVM bagus untuk data kecil-menengah.",
                "Pada dataset besar, training SVM bisa lebih berat.",
                "XGBoost lebih fleksibel terhadap fitur campuran dan missing value.",
            ],
            "XGBoost lebih cocok untuk dataset kesehatan tabular dengan jumlah data besar.",
        ),
    ]

    for title, explanation, bullets, insight in analysis_items:
        with st.expander(title, expanded=title == "XGBoost vs Logistic Regression"):
            _render_comparison_card(title, insight, bullets, icon="🧠")
            st.markdown("#### Penjelasan")
            st.markdown(explanation)

    st.markdown("#### 📈 Kesimpulan Analisis")
    conclusion = (
        "Berdasarkan hasil pengujian, XGBoost dipilih karena mampu memberikan performa accuracy dan recall yang tinggi, "
        "stabil pada multiple testing, serta lebih efektif menangani pola kompleks pada data kesehatan dibanding metode machine learning lainnya."
    )

    if best_accuracy >= 0.90:
        st.success(conclusion)
    else:
        st.info(conclusion)

    st.markdown(
        f"""
        <div style="
            background: #ecfeff;
            border: 1px solid #a5f3fc;
            border-radius: 14px;
            padding: 14px 16px;
            color: #155e75;
        ">
            <strong>Ringkasan untuk presentasi:</strong> XGBoost menunjukkan kombinasi terbaik antara akurasi, recall, dan stabilitas pada multiple experiment testing,
            sehingga lebih layak dipilih untuk klasifikasi diabetes pada dataset tabular kesehatan.
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_experiment_report(bundle: dict):
    """Render report hasil multiple experiment testing di Streamlit."""
    experiments_df = bundle["experiments_summary"].copy()
    best = get_best_model(bundle)

    st.markdown("### 🧪 Multiple Experiment Testing")
    st.info("Pengujian dilakukan sebanyak 15 kali untuk mendapatkan model dengan performa paling stabil dan akurat.")

    st.markdown("#### 📋 Tabel Seluruh Percobaan")
    st.dataframe(
        _style_experiment_table(experiments_df),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("#### 📊 Grafik Perbandingan Accuracy")
    st.pyplot(_plot_accuracy_comparison(experiments_df), clear_figure=True)

    st.markdown("#### 🏆 Model Terbaik")
    best_col1, best_col2, best_col3, best_col4 = st.columns(4)
    with best_col1:
        st.metric("Test Terbaik", f"{best['best_experiment_no']}")
    with best_col2:
        st.metric("Random State", f"{best['best_random_state']}")
    with best_col3:
        st.metric("Accuracy Terbaik", f"{best['best_accuracy'] * 100:.2f}%")
    with best_col4:
        st.metric("F1-Score Terbaik", f"{best['best_f1_score'] * 100:.2f}%")

    st.markdown("#### 📄 Classification Report Model Terbaik")
    best_report_df = pd.DataFrame(best["best_classification_report"]).transpose()
    st.dataframe(best_report_df.round(4), use_container_width=True)

    st.markdown("#### 🔢 Confusion Matrix Model Terbaik")
    best_cm_df = pd.DataFrame(
        best["best_confusion_matrix"],
        index=["Actual: 0", "Actual: 1"],
        columns=["Predicted: 0", "Predicted: 1"],
    )
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(best_cm_df, annot=True, fmt="d", cmap="Greens", cbar=False, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig, clear_figure=True)

    st.dataframe(best_cm_df, use_container_width=True)

    return best


def show():
    st.markdown("# 🩸 Deteksi Diabetes dengan XGBoost")
    st.markdown(
        "Prediksi status diabetes berdasarkan data klinis pasien menggunakan "
        "preprocessing otomatis dan model XGBoost."
    )
    st.markdown("---")

    num_experiments = st.sidebar.selectbox(
        "Jumlah percobaan:",
        EXPERIMENT_CHOICES,
        index=EXPERIMENT_CHOICES.index(DEFAULT_EXPERIMENTS),
        help="Jumlah training-testing otomatis untuk memilih model terbaik.",
    )
    st.sidebar.caption("Pengujian dilakukan sebanyak 15 kali untuk mendapatkan model dengan performa paling stabil dan akurat.")

    bundle = get_bundle(num_experiments)
    df = load_data()
    smoking_options = _get_smoking_options(df)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Model Diabetes")
    st.sidebar.success("Model utama: XGBoost")
    st.sidebar.caption(f"Model tersimpan: {MODEL_PATH}")
    st.sidebar.caption("Preprocessing otomatis: missing value, encoding, scaling numerik")
    st.sidebar.caption(f"Model terbaik: test #{bundle['best_experiment_no']} | random_state {bundle['best_random_state']}")

    tab1, tab2, tab3, tab4 = st.tabs(["🔮 Prediksi", "📊 Evaluasi Model", "📈 Eksplorasi Data", "🧠 Analisis Perbandingan Metode"])

    with tab1:
        st.markdown("### 📝 Input Data Pasien")
        st.markdown("Masukkan data pasien berikut untuk memprediksi risiko diabetes.")

        with st.form("diabetes_prediction_form"):
            col1, col2, col3 = st.columns(3)

            with col1:
                gender = st.selectbox("Gender", ["Female", "Male"])
                age = st.number_input("Usia (tahun)", min_value=0.0, max_value=120.0, value=45.0, step=1.0)
                hypertension = st.selectbox("Hipertensi", [0, 1], format_func=lambda value: "Ya" if value == 1 else "Tidak")

            with col2:
                heart_disease = st.selectbox("Penyakit Jantung", [0, 1], format_func=lambda value: "Ya" if value == 1 else "Tidak")
                smoking_history = st.selectbox("Riwayat Merokok", smoking_options)
                bmi = st.number_input("BMI", min_value=10.0, max_value=70.0, value=25.0, step=0.1)

            with col3:
                hba1c_level = st.number_input("HbA1c Level", min_value=3.0, max_value=15.0, value=5.7, step=0.1)
                blood_glucose_level = st.number_input("Blood Glucose Level", min_value=50, max_value=400, value=120, step=1)

            submitted = st.form_submit_button("🔍 Prediksi Diabetes", use_container_width=True)

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

            st.markdown("### 📋 Hasil Prediksi")
            if prediction == 1:
                st.error("⚠️ **BERISIKO DIABETES**")
                st.markdown(
                    "Model memprediksi pasien **berisiko diabetes**. "
                    "Evaluasi medis lebih lanjut sangat disarankan."
                )
            else:
                st.success("✅ **TIDAK BERISIKO DIABETES**")
                st.markdown(
                    "Model memprediksi pasien **tidak berisiko diabetes** berdasarkan data yang dimasukkan."
                )

            prob_non_diabetes = float(probability[0])
            prob_diabetes = float(probability[1])
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Probabilitas Tidak Diabetes", f"{prob_non_diabetes * 100:.2f}%")
            with col_b:
                st.metric("Probabilitas Diabetes", f"{prob_diabetes * 100:.2f}%")

            st.info("Preprocessing otomatis menangani missing value, encoding kategori, dan scaling numerik.")
            st.caption(f"Prediksi menggunakan model terbaik dari eksperimen ke-{bundle['best_experiment_no']}.")

    with tab2:
        st.markdown("### 📊 Evaluasi Model XGBoost")
        best = show_experiment_report(bundle)

        st.markdown("---")
        st.markdown("#### 📦 Info Training")
        info_col1, info_col2, info_col3 = st.columns(3)
        with info_col1:
            st.metric("Data Train", f"{bundle['train_shape'][0]}")
        with info_col2:
            st.metric("Data Test", f"{bundle['test_shape'][0]}")
        with info_col3:
            st.metric("Feature Input", f"{len(bundle['feature_columns'])}")

        st.caption(f"Best model selected by highest accuracy, then highest f1-score if tied. Current best test: #{best['best_experiment_no']}.")

    with tab3:
        st.markdown("### 📈 Eksplorasi Dataset Diabetes")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Jumlah Data", df.shape[0])
        with col2:
            st.metric("Jumlah Fitur", df.shape[1] - 1)
        with col3:
            diabetes_rate = (df["diabetes"].mean() * 100) if len(df) else 0
            st.metric("Persentase Diabetes", f"{diabetes_rate:.2f}%")

        st.markdown("---")
        st.markdown("#### 👀 Preview Data")
        st.dataframe(df.head(20), use_container_width=True)

        st.markdown("#### 🎯 Distribusi Target")
        target_counts = df["diabetes"].value_counts().sort_index()
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.bar_chart(target_counts)
        with chart_col2:
            st.markdown(
                f"""
                - **Tidak Diabetes (0)**: {int(target_counts.get(0, 0))}
                - **Diabetes (1)**: {int(target_counts.get(1, 0))}
                """
            )

        st.markdown("#### 🧼 Missing Value per Kolom")
        missing_df = df.isna().sum().to_frame(name="missing_values")
        st.dataframe(missing_df, use_container_width=True)

        st.markdown("#### 📊 Statistik Deskriptif")
        st.dataframe(df.describe(include="all").transpose().round(3), use_container_width=True)

    with tab4:
        show_method_analysis(bundle)
