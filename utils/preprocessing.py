"""
Modul utilitas untuk preprocessing data.
Berisi fungsi-fungsi umum yang digunakan di berbagai halaman deteksi.
Fokus utama: Dataset Cardiovascular Disease (binary classification).

Target:
    cardio = 0 -> Tidak terdeteksi penyakit kardiovaskular
    cardio = 1 -> Terdeteksi penyakit kardiovaskular

Pipeline lengkap mencakup feature engineering:
    - bmi              = weight / (height/100)^2
    - pulse_pressure   = ap_hi - ap_lo
    - map_pressure     = ap_lo + (ap_hi - ap_lo) / 3
    - bp_category      (normal / elevated / stage1 / stage2)
    - age_group        (<40 / 40-49 / 50-59 / 60+)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


# ===== Konstanta dataset cardiovascular =====
CARDIO_TARGET_COL = "cardio"

CARDIO_RAW_FEATURES = [
    "age_years", "gender", "height", "weight",
    "ap_hi", "ap_lo", "cholesterol", "gluc",
    "smoke", "alco", "active",
]
CARDIO_ENGINEERED_FEATURES = [
    "bmi", "pulse_pressure", "map_pressure",
    "bp_category", "age_group",
]
CARDIO_FEATURE_COLS = CARDIO_RAW_FEATURES + CARDIO_ENGINEERED_FEATURES
CARDIO_CATEGORICAL_COLS = ["bp_category", "age_group"]

CARDIO_BINARY_NAMES = {0: "tidak cardio", 1: "cardio"}

# Mapping label klinis untuk variabel ordinal cardio dataset
CHOLESTEROL_LABELS = {1: "Normal", 2: "Di atas normal", 3: "Jauh di atas normal"}
GLUC_LABELS = {1: "Normal", 2: "Di atas normal", 3: "Jauh di atas normal"}
GENDER_LABELS = {1: "Perempuan", 2: "Laki-laki"}


def _bp_category(ap_hi: float, ap_lo: float) -> str:
    """Kategorisasi tekanan darah (American Heart Association sederhana)."""
    if ap_hi < 120 and ap_lo < 80:
        return "normal"
    if ap_hi < 130 and ap_lo < 80:
        return "elevated"
    if ap_hi < 140 or ap_lo < 90:
        return "stage1"
    return "stage2"


def _age_group(age_years: float) -> str:
    """Kelompokkan umur ke 4 grup."""
    if age_years < 40:
        return "<40"
    if age_years < 50:
        return "40-49"
    if age_years < 60:
        return "50-59"
    return "60+"


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tambahkan fitur turunan ke df (BMI, pulse_pressure, map_pressure,
    bp_category, age_group)."""
    df = df.copy()
    df["bmi"] = df["weight"] / (df["height"] / 100) ** 2
    df["pulse_pressure"] = df["ap_hi"] - df["ap_lo"]
    df["map_pressure"] = df["ap_lo"] + (df["ap_hi"] - df["ap_lo"]) / 3
    df["bp_category"] = df.apply(
        lambda r: _bp_category(r["ap_hi"], r["ap_lo"]), axis=1
    )
    df["age_group"] = df["age_years"].apply(_age_group)
    return df


def load_and_clean_cardio_data(filepath: str) -> pd.DataFrame:
    """
    Load dan bersihkan dataset cardiovascular.
    - Pisahkan menggunakan ``;`` (format dataset asli).
    - Drop kolom ``id``.
    - Konversi ``age`` (hari) ke ``age_years``.
    - Filter outlier fisiologis pada tekanan darah, tinggi, dan berat.
    - Hapus duplikat.
    """
    df = pd.read_csv(filepath, sep=";")

    if "id" in df.columns:
        df = df.drop(columns=["id"])

    if "age_years" not in df.columns and "age" in df.columns:
        df["age_years"] = (df["age"] / 365.25).round(1)

    # Filter outlier fisiologis
    mask = (
        df["ap_hi"].between(80, 250)
        & df["ap_lo"].between(40, 200)
        & (df["ap_hi"] > df["ap_lo"])
        & df["height"].between(120, 220)
        & df["weight"].between(30, 200)
    )
    df = df[mask].copy()

    if "age" in df.columns:
        df = df.drop(columns=["age"])

    df = df.dropna(subset=CARDIO_RAW_FEATURES + [CARDIO_TARGET_COL])
    df = df.drop_duplicates().reset_index(drop=True)
    df[CARDIO_TARGET_COL] = df[CARDIO_TARGET_COL].astype(int)

    return df


def encode_cardio_features(df: pd.DataFrame):
    """
    Tambahkan engineered features, encode kolom kategorikal, dan return X, y.
    Returns:
        (X, y, feature_names, label_encoders)
    """
    df_fe = add_engineered_features(df)

    label_encoders: dict[str, LabelEncoder] = {}
    for col in CARDIO_CATEGORICAL_COLS:
        le = LabelEncoder()
        df_fe[col] = le.fit_transform(df_fe[col].astype(str))
        label_encoders[col] = le

    X = df_fe[CARDIO_FEATURE_COLS]
    y = df_fe[CARDIO_TARGET_COL].astype(int)
    feature_names = X.columns.tolist()

    return X, y, feature_names, label_encoders


def prepare_cardio_data(
    filepath: str,
    test_size: float = 0.2,
    apply_smote: bool = False,
    scaler_type: str = "standard",
    sample_size: int | None = None,
):
    """
    Pipeline lengkap preprocessing dataset cardio (binary).

    Parameter ``sample_size`` (opsional) berguna pada UI Streamlit untuk
    membatasi training data agar startup tetap responsif (W-KNN/Random
    Forest dapat lambat pada 70k baris). Set ``None`` untuk gunakan semua.

    Returns:
        X_train, X_test, y_train, y_test,
        scaler, label_encoders, feature_names
    """
    df = load_and_clean_cardio_data(filepath)

    if sample_size is not None and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    X, y, feature_names, label_encoders = encode_cardio_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    if scaler_type == "standard":
        scaler = StandardScaler()
    elif scaler_type == "minmax":
        scaler = MinMaxScaler()
    elif scaler_type == "robust":
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if apply_smote:
        smote = SMOTE(random_state=42)
        X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)

    return (
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        scaler,
        label_encoders,
        feature_names,
    )


def transform_single_record(
    record: dict,
    scaler,
    label_encoders: dict,
    feature_names: list[str],
) -> np.ndarray:
    """
    Transformasi satu input prediksi (dict berisi raw features) menggunakan
    pipeline preprocessing yang sama dengan training.

    record harus berisi kunci raw cardio:
        age_years, gender (1/2), height, weight,
        ap_hi, ap_lo, cholesterol (1/2/3), gluc (1/2/3),
        smoke (0/1), alco (0/1), active (0/1)
    """
    df = pd.DataFrame([record])
    df = add_engineered_features(df)

    for col in CARDIO_CATEGORICAL_COLS:
        le = label_encoders[col]
        try:
            df[col] = le.transform(df[col].astype(str))
        except ValueError:
            df[col] = 0

    df = df[feature_names]
    return scaler.transform(df)
