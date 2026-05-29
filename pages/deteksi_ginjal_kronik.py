"""Halaman Deteksi Penyakit Ginjal Kronik (Chronic Kidney Disease / CKD).

Dataset: ``dataset/penyakit_ginjal_kronik.csv`` (400 pasien, 24 fitur klinis).

Target:
    target = 0 -> Tidak terdeteksi penyakit ginjal kronik
    target = 1 -> Terdeteksi penyakit ginjal kronik (CKD)

Base method utama: **W-KNN (Weighted K-Nearest Neighbors)**.

Eksperimen lengkap (perbandingan model, scaler, dan tuning) ada di
``Pembangunan_Model_Preprocessing_CKD_WKNN.ipynb``.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Tambahkan parent dir ke path agar `from utils...` bekerja kalau page
# di-import langsung.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.ckd_pipeline import (  # noqa: E402
    CKD_BINARY_OPTIONS,
    CKD_NUMERIC_RANGES,
    CKD_FEATURE_COLS,
    CKD_NUMERIC_COLS,
    CKD_BINARY_CATEGORICAL_COLS,
    CKD_TARGET_COL,
    CKD_TARGET_LABELS,
    load_and_clean_ckd_data,
    prepare_ckd_data,
    transform_single_record,
)
from utils.models import (  # noqa: E402
    AVAILABLE_MODELS,
    get_model,
    train_and_evaluate,
)


# ----------------------- Helpers ---------------------------------------------

def _dataset_path() -> str:
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, "dataset", "penyakit_ginjal_kronik.csv")


@st.cache_data
def _load_raw() -> pd.DataFrame:
    return load_and_clean_ckd_data(_dataset_path())


@st.cache_resource
def _train_model(
    model_key: str,
    scaler_type: str,
    apply_smote: bool,
    knn_k: int = 5,
):
    X_train, X_test, y_train, y_test, scaler, encoders, imputers, feats = (
        prepare_ckd_data(
            _dataset_path(),
            scaler_type=scaler_type,
            apply_smote=apply_smote,
        )
    )

    model = get_model(model_key)
    # Override n_neighbors kalau model wknn
    if model_key == "wknn":
        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier(
            n_neighbors=knn_k, weights="distance", n_jobs=-1
        )

    metrics = train_and_evaluate(model, X_train, X_test, y_train, y_test)
    return {
        "model": model,
        "scaler": scaler,
        "encoders": encoders,
        "imputers": imputers,
        "feature_names": feats,
        "metrics": metrics,
        "X_test": X_test,
        "y_test": y_test,
    }


def _format_pct(x: float) -> str:
    return f"{x*100:.2f}%" if x is not None else "—"


# ----------------------- UI Sections -----------------------------------------

def _render_header():
    st.markdown("# 🧪 Deteksi Penyakit Ginjal Kronik (CKD)")
    st.markdown(
        "Dataset: **400 pasien**, target binary "
        "(`ckd` vs `notckd`). Base method: **W-KNN**."
    )
    st.markdown("---")


def _render_dataset_overview(df_raw: pd.DataFrame):
    st.subheader("📊 Ringkasan Dataset")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Pasien", f"{len(df_raw):,}")
    n_ckd = int((df_raw[CKD_TARGET_COL] == 1).sum())
    c2.metric("Terdeteksi CKD", f"{n_ckd}")
    c3.metric("Tidak CKD", f"{len(df_raw) - n_ckd}")

    with st.expander("🔍 Lihat 10 baris pertama dataset (setelah cleaning)"):
        st.dataframe(df_raw.head(10), use_container_width=True)

    with st.expander("ℹ️ Catatan preprocessing"):
        st.markdown(
            """
- **Whitespace** di nilai string dibersihkan (mis. ``' yes'`` → ``'yes'``).
- Kolom numerik yang tersimpan sebagai string (``MCV``,
  ``seldarahputih``, ``seldarahmerah.1``) dikonversi paksa ke float.
- **Missing values** diimputasi: median (numerik), modus (kategorikal).
- Target ``klasifikasi`` dimapping: `'ckd'` → 1, `'notckd'` → 0.
- Kolom duplikat ``seldarahmerah`` (kategori) & ``seldarahmerah.1``
  (numerik) di-rename jadi ``seldarahmerah_kat`` &
  ``seldarahmerah_count`` agar tidak ambigu.
            """
        )


def _render_sidebar() -> dict:
    st.sidebar.markdown("### ⚙️ Konfigurasi Model")

    model_label = st.sidebar.selectbox(
        "Pilih Model",
        list(AVAILABLE_MODELS.keys()),
        index=list(AVAILABLE_MODELS.keys()).index("W-KNN (Weighted KNN)"),
    )
    model_key = AVAILABLE_MODELS[model_label]

    scaler_type = st.sidebar.selectbox(
        "Scaler", ["minmax", "standard", "robust"], index=0
    )

    apply_smote = st.sidebar.checkbox(
        "Apply SMOTE (oversample CKD minority)", value=False,
        help="Dataset 250/150, sudah cukup seimbang. SMOTE opsional.",
    )

    knn_k = 5
    if model_key == "wknn":
        knn_k = st.sidebar.slider(
            "Jumlah tetangga (k) untuk WKNN", min_value=1,
            max_value=21, value=5, step=2,
        )

    return {
        "model_key": model_key,
        "model_label": model_label,
        "scaler_type": scaler_type,
        "apply_smote": apply_smote,
        "knn_k": knn_k,
    }


def _render_metrics(bundle: dict, cfg: dict):
    metrics = bundle["metrics"]
    st.subheader(f"📈 Performa Model — {cfg['model_label']}")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", _format_pct(metrics["accuracy"]))
    c2.metric("Precision", _format_pct(metrics["precision"]))
    c3.metric("Recall", _format_pct(metrics["recall"]))
    c4.metric("F1-Score", _format_pct(metrics["f1_score"]))
    c5.metric("ROC AUC", _format_pct(metrics["roc_auc"]))


def _render_eval_charts(bundle: dict, cfg: dict):
    metrics = bundle["metrics"]
    cm = metrics["confusion_matrix"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Confusion matrix
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Tidak CKD", "CKD"],
        yticklabels=["Tidak CKD", "CKD"],
        ax=axes[0],
    )
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")
    axes[0].set_title(f"Confusion Matrix — {cfg['model_label']}")

    # ROC curve
    if metrics["roc_auc"] is not None:
        from sklearn.metrics import roc_curve
        try:
            y_proba = bundle["model"].predict_proba(bundle["X_test"])[:, 1]
            fpr, tpr, _ = roc_curve(bundle["y_test"], y_proba)
            axes[1].plot(fpr, tpr, label=f"AUC = {metrics['roc_auc']:.4f}",
                         linewidth=2)
            axes[1].plot([0, 1], [0, 1], "k--", alpha=0.5)
            axes[1].set_xlabel("False Positive Rate")
            axes[1].set_ylabel("True Positive Rate")
            axes[1].set_title("ROC Curve")
            axes[1].legend(loc="lower right")
        except Exception:
            axes[1].axis("off")
    else:
        axes[1].axis("off")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _render_form_predict(bundle: dict, cfg: dict):
    st.subheader("🩺 Form Prediksi Pasien")
    st.markdown(
        "Isi data klinis pasien di bawah. Klik **Prediksi** untuk lihat "
        "hasil klasifikasi."
    )

    with st.form("ckd_predict_form"):
        col_left, col_right = st.columns(2)

        record: dict = {}

        # Numeric inputs (split between two columns)
        numeric_items = list(CKD_NUMERIC_RANGES.items())
        half = len(numeric_items) // 2 + 1
        for i, (col, (mn, mx, default)) in enumerate(numeric_items):
            target_col = col_left if i < half else col_right
            record[col] = target_col.number_input(
                _label_for(col), min_value=float(mn), max_value=float(mx),
                value=float(default), step=_step_for(col),
            )

        # Categorical inputs
        st.markdown("---")
        st.markdown("**Riwayat & Pemeriksaan Kategorikal**")
        cat_cols = st.columns(3)
        for i, (col, opts) in enumerate(CKD_BINARY_OPTIONS.items()):
            target_col = cat_cols[i % 3]
            record[col] = target_col.selectbox(_label_for(col), opts, index=0)

        submitted = st.form_submit_button("🔮 Prediksi")

    if not submitted:
        return

    # Inference
    X = transform_single_record(
        record,
        scaler=bundle["scaler"],
        encoders=bundle["encoders"],
        imputers=bundle["imputers"],
        feature_names=bundle["feature_names"],
    )
    model = bundle["model"]
    pred = int(model.predict(X)[0])
    try:
        proba = float(model.predict_proba(X)[0, 1])
    except Exception:
        proba = None

    label = CKD_TARGET_LABELS[pred]
    st.markdown("---")
    if pred == 1:
        st.error(f"### ⚠️ Hasil: **{label}**")
    else:
        st.success(f"### ✅ Hasil: **{label}**")

    if proba is not None:
        st.progress(min(max(proba, 0.0), 1.0))
        st.caption(f"Probabilitas terdeteksi CKD: **{proba*100:.2f}%**")

    st.info(
        "⚠️ Hasil ini bersifat indikatif berdasarkan model statistik. "
        "Konsultasikan ke tenaga medis untuk diagnosis pasti."
    )


# ----------------------- Label helpers ---------------------------------------

_LABEL_OVERRIDES = {
    "umur": "Umur (tahun)",
    "tekanandarah": "Tekanan Darah (mmHg)",
    "gravitas": "Berat Jenis Urin (Specific Gravity)",
    "albumin": "Albumin (skala 0-5)",
    "sugar": "Sugar dalam Urin (skala 0-5)",
    "gds": "Gula Darah Sewaktu (mg/dL)",
    "ureum": "Ureum (mg/dL)",
    "kreatinin": "Kreatinin (mg/dL)",
    "natrium": "Natrium (mEq/L)",
    "kalium": "Kalium (mEq/L)",
    "hemoglobin": "Hemoglobin (g/dL)",
    "mcv": "MCV (fL)",
    "seldarahputih": "Sel Darah Putih (cells/cmm)",
    "seldarahmerah_count": "Sel Darah Merah (juta/cmm)",
    "seldarahmerah_kat": "Sel Darah Merah (kategori)",
    "pussel": "Pus Cell",
    "puscell": "Pus Cell Clumps",
    "bakteri": "Bakteri",
    "hipertensi": "Riwayat Hipertensi",
    "diabetes": "Riwayat Diabetes Melitus",
    "cad": "Coronary Artery Disease",
    "nafsumakan": "Nafsu Makan",
    "edema": "Edema (Pembengkakan)",
    "anemia": "Anemia",
}


def _label_for(col: str) -> str:
    return _LABEL_OVERRIDES.get(col, col)


def _step_for(col: str) -> float:
    if col == "gravitas":
        return 0.001
    if col in ("seldarahputih",):
        return 100.0
    if col in ("kreatinin", "hemoglobin", "seldarahmerah_count"):
        return 0.1
    return 1.0


# ----------------------- Entry point -----------------------------------------

def show():
    _render_header()
    cfg = _render_sidebar()
    df_raw = _load_raw()
    _render_dataset_overview(df_raw)

    with st.spinner("Melatih model..."):
        bundle = _train_model(
            cfg["model_key"], cfg["scaler_type"],
            cfg["apply_smote"], knn_k=cfg["knn_k"],
        )

    _render_metrics(bundle, cfg)
    _render_eval_charts(bundle, cfg)
    _render_form_predict(bundle, cfg)
