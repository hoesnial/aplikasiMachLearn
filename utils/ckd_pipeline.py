"""Pipeline preprocessing untuk dataset Penyakit Ginjal Kronik (CKD).

Dataset: ``dataset/penyakit_ginjal_kronik.csv`` (400 baris, 26 kolom).

Target:
    klasifikasi = 'ckd'    -> 1 (terdeteksi penyakit ginjal kronik)
    klasifikasi = 'notckd' -> 0 (tidak terdeteksi)

Base method utama: **W-KNN (Weighted K-Nearest Neighbors)**.

Karakteristik dataset & masalah yang ditangani:
    - **Banyak missing values** di hampir semua kolom (5%–38%).
    - **Whitespace tersembunyi** di nilai string (mis. ``' yes'``,
      ``'\\tno'``) yang membuat kategori sama dianggap berbeda.
    - **Kolom numerik tersimpan sebagai string** karena whitespace
      (``MCV``, ``seldarahputih``, ``seldarahmerah.1``) — perlu di-coerce
      ke numerik.
    - **Label target tidak konsisten** (``'ckd'`` vs ``'ckd\\t'``).
    - **Duplikat nama kolom**: ``seldarahmerah`` (kategori) dan
      ``seldarahmerah.1`` (numerik float) — kita rename agar lebih jelas.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler, StandardScaler

# ===== Konstanta dataset =====
CKD_TARGET_COL = "target"  # nama setelah dimapping ke 0/1

# Mapping kolom numerik (perlu coerce ke float)
CKD_NUMERIC_COLS = [
    "umur", "tekanandarah", "gravitas", "albumin", "sugar",
    "gds", "ureum", "kreatinin", "natrium", "kalium",
    "hemoglobin", "mcv", "seldarahputih", "seldarahmerah_count",
]

# Kolom kategorikal binary (yes/no, normal/abnormal, present/notpresent, good/poor)
CKD_BINARY_CATEGORICAL_COLS = [
    "seldarahmerah_kat", "pussel", "puscell", "bakteri",
    "hipertensi", "diabetes", "cad", "nafsumakan", "edema", "anemia",
]

CKD_FEATURE_COLS = CKD_NUMERIC_COLS + CKD_BINARY_CATEGORICAL_COLS

CKD_TARGET_LABELS = {0: "Tidak CKD", 1: "Terdeteksi CKD"}

# Untuk UI form: opsi pilihan & rentang fisiologis
CKD_BINARY_OPTIONS = {
    "seldarahmerah_kat": ("normal", "abnormal"),
    "pussel": ("normal", "abnormal"),
    "puscell": ("notpresent", "present"),
    "bakteri": ("notpresent", "present"),
    "hipertensi": ("no", "yes"),
    "diabetes": ("no", "yes"),
    "cad": ("no", "yes"),
    "nafsumakan": ("good", "poor"),
    "edema": ("no", "yes"),
    "anemia": ("no", "yes"),
}

CKD_NUMERIC_RANGES = {
    "umur": (1.0, 100.0, 50.0),  # min, max, default
    "tekanandarah": (50.0, 200.0, 80.0),
    "gravitas": (1.000, 1.030, 1.020),
    "albumin": (0.0, 5.0, 1.0),
    "sugar": (0.0, 5.0, 0.0),
    "gds": (50.0, 500.0, 120.0),
    "ureum": (10.0, 400.0, 50.0),
    "kreatinin": (0.4, 80.0, 1.2),
    "natrium": (100.0, 170.0, 137.0),
    "kalium": (2.0, 50.0, 4.5),
    "hemoglobin": (3.0, 18.0, 13.0),
    "mcv": (15.0, 60.0, 41.0),
    "seldarahputih": (2000.0, 30000.0, 8000.0),
    "seldarahmerah_count": (2.0, 8.0, 5.0),
}


# ----------------------------- Cleaning --------------------------------------

def _strip_str(x):
    """Bersihkan whitespace dan tab dari string."""
    if isinstance(x, str):
        return x.strip().lower()
    return x


def _to_numeric_safe(series: pd.Series) -> pd.Series:
    """Konversi series ke numerik, nilai non-numerik jadi NaN."""
    if series.dtype == object:
        series = series.map(_strip_str)
    return pd.to_numeric(series, errors="coerce")


def load_and_clean_ckd_data(filepath: str) -> pd.DataFrame:
    """Load dataset CKD dan lakukan pembersihan dasar.

    Langkah:
        1. Drop kolom ``id``.
        2. Strip whitespace dari semua kolom string.
        3. Rename ``seldarahmerah`` (kategori) -> ``seldarahmerah_kat``,
           ``seldarahmerah.1`` (numerik float) -> ``seldarahmerah_count``.
        4. Lowercase nama kolom (``MCV`` -> ``mcv``).
        5. Coerce kolom numerik yang tersimpan sebagai string.
        6. Mapping target ``klasifikasi`` -> 0/1 (drop baris kalau invalid).
    """
    df = pd.read_csv(filepath)

    # 1) Drop id
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    # 2) Rename duplikat & lowercase kolom
    df = df.rename(
        columns={
            "seldarahmerah": "seldarahmerah_kat",
            "seldarahmerah.1": "seldarahmerah_count",
            "MCV": "mcv",
        }
    )

    # 3) Strip whitespace di semua kolom object
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].map(_strip_str)

    # 4) Coerce kolom numerik
    for c in CKD_NUMERIC_COLS:
        if c in df.columns:
            df[c] = _to_numeric_safe(df[c])

    # 5) Normalisasi target (ckd -> 1, notckd -> 0)
    df["klasifikasi"] = df["klasifikasi"].map(_strip_str)
    target_map = {"ckd": 1, "notckd": 0}
    df[CKD_TARGET_COL] = df["klasifikasi"].map(target_map)
    df = df.drop(columns=["klasifikasi"])
    df = df.dropna(subset=[CKD_TARGET_COL]).reset_index(drop=True)
    df[CKD_TARGET_COL] = df[CKD_TARGET_COL].astype(int)

    return df


# ----------------------------- Imputasi --------------------------------------

def impute_missing(
    df: pd.DataFrame,
    numeric_cols: List[str] | None = None,
    categorical_cols: List[str] | None = None,
) -> Tuple[pd.DataFrame, Dict[str, SimpleImputer]]:
    """Imputasi missing values:
        - Numerik: median.
        - Kategorikal: mode (most frequent).

    Returns
    -------
    df : pd.DataFrame
        DataFrame dengan missing values terisi.
    imputers : dict
        Mapping ``{kolom: imputer}`` agar bisa dipakai ulang untuk inference.
    """
    numeric_cols = numeric_cols or CKD_NUMERIC_COLS
    categorical_cols = categorical_cols or CKD_BINARY_CATEGORICAL_COLS

    df = df.copy()
    imputers: Dict[str, SimpleImputer] = {}

    for c in numeric_cols:
        if c in df.columns:
            imp = SimpleImputer(strategy="median")
            df[c] = imp.fit_transform(df[[c]]).ravel()
            imputers[c] = imp

    for c in categorical_cols:
        if c in df.columns:
            imp = SimpleImputer(strategy="most_frequent")
            df[c] = imp.fit_transform(df[[c]]).ravel()
            imputers[c] = imp

    return df, imputers


# ----------------------------- Encoding --------------------------------------

def encode_categoricals(
    df: pd.DataFrame,
    categorical_cols: List[str] | None = None,
) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """LabelEncode kolom binary kategorikal."""
    categorical_cols = categorical_cols or CKD_BINARY_CATEGORICAL_COLS

    df = df.copy()
    encoders: Dict[str, LabelEncoder] = {}

    for c in categorical_cols:
        if c in df.columns:
            le = LabelEncoder()
            df[c] = le.fit_transform(df[c].astype(str))
            encoders[c] = le

    return df, encoders


# ----------------------------- Pipeline --------------------------------------

def _get_scaler(scaler_type: str):
    scaler_type = scaler_type.lower()
    if scaler_type == "standard":
        return StandardScaler()
    if scaler_type == "minmax":
        return MinMaxScaler()
    if scaler_type == "robust":
        return RobustScaler()
    raise ValueError(f"scaler_type tidak dikenal: {scaler_type}")


def prepare_ckd_data(
    filepath: str,
    test_size: float = 0.2,
    scaler_type: str = "minmax",
    apply_smote: bool = False,
    random_state: int = 42,
):
    """Pipeline lengkap CKD: load -> clean -> impute -> encode -> split ->
    scale -> (opsional) SMOTE.

    Returns
    -------
    X_train, X_test, y_train, y_test : np.ndarray
    scaler : fitted scaler
    encoders : dict[col] -> LabelEncoder
    imputers : dict[col] -> SimpleImputer
    feature_names : list[str]
    """
    df = load_and_clean_ckd_data(filepath)
    df, imputers = impute_missing(df)
    df, encoders = encode_categoricals(df)

    X = df[CKD_FEATURE_COLS].astype(float).values
    y = df[CKD_TARGET_COL].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = _get_scaler(scaler_type)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if apply_smote:
        try:
            sm = SMOTE(random_state=random_state)
            X_train, y_train = sm.fit_resample(X_train, y_train)
        except ValueError:
            # Data sudah seimbang atau terlalu sedikit untuk SMOTE
            pass

    return (
        X_train, X_test, y_train, y_test,
        scaler, encoders, imputers, CKD_FEATURE_COLS,
    )


# ----------------------------- Inference --------------------------------------

def transform_single_record(
    record: Dict[str, float | str],
    scaler,
    encoders: Dict[str, LabelEncoder],
    imputers: Dict[str, SimpleImputer],
    feature_names: List[str] | None = None,
) -> np.ndarray:
    """Transform 1 record dari form input ke vektor siap predict.

    Parameters
    ----------
    record : dict
        Mapping ``{feature: value}``. Nilai kategorikal pakai string asli
        (mis. ``'yes'``), nilai numerik pakai float/int.
    """
    feature_names = feature_names or CKD_FEATURE_COLS

    df = pd.DataFrame([record], columns=feature_names)

    # Pastikan kolom numerik bertipe float
    for c in CKD_NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Imputasi pakai imputer yang sudah di-fit (untuk handle missing)
    for c, imp in imputers.items():
        if c in df.columns:
            df[c] = imp.transform(df[[c]]).ravel()

    # Encode kategorikal
    for c, le in encoders.items():
        if c in df.columns:
            val = str(df[c].iloc[0]).strip().lower()
            if val in le.classes_:
                df[c] = le.transform([val])
            else:
                # fallback ke kelas pertama (most common)
                df[c] = le.transform([le.classes_[0]])

    X = df[feature_names].astype(float).values
    return scaler.transform(X)


__all__ = [
    "CKD_TARGET_COL",
    "CKD_TARGET_LABELS",
    "CKD_NUMERIC_COLS",
    "CKD_BINARY_CATEGORICAL_COLS",
    "CKD_FEATURE_COLS",
    "CKD_BINARY_OPTIONS",
    "CKD_NUMERIC_RANGES",
    "load_and_clean_ckd_data",
    "impute_missing",
    "encode_categoricals",
    "prepare_ckd_data",
    "transform_single_record",
]
