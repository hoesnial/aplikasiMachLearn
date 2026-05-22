"""
Modul utilitas untuk preprocessing data.
Berisi fungsi-fungsi umum yang digunakan di berbagai halaman deteksi.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


def load_and_clean_stroke_data(filepath: str) -> pd.DataFrame:
    """
    Load dan bersihkan dataset stroke.
    - Hapus kolom 'id'
    - Imputasi BMI dengan median
    - Hapus gender='Other'
    - Hapus duplikat
    """
    df = pd.read_csv(filepath)

    # Hapus kolom id
    if 'id' in df.columns:
        df = df.drop('id', axis=1)

    # Imputasi BMI dengan median (handle 'N/A' string juga)
    df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')
    median_bmi = df['bmi'].median()
    df['bmi'] = df['bmi'].fillna(median_bmi)

    # Hapus gender='Other'
    df = df[df['gender'] != 'Other']

    # Hapus duplikat
    df = df.drop_duplicates()

    return df


def encode_stroke_features(df: pd.DataFrame) -> tuple:
    """
    Encode fitur kategorikal dan pisahkan X, y untuk dataset stroke.
    Returns: (X, y, feature_names, label_encoders)
    """
    df_encoded = df.copy()
    label_encoders = {}

    categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']

    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        label_encoders[col] = le

    X = df_encoded.drop('stroke', axis=1)
    y = df_encoded['stroke']
    feature_names = X.columns.tolist()

    return X, y, feature_names, label_encoders


def prepare_stroke_data(filepath: str, test_size: float = 0.2, apply_smote: bool = True, scaler_type: str = 'standard'):
    """
    Pipeline lengkap preprocessing dataset stroke.
    Returns: (X_train, X_test, y_train, y_test, scaler, label_encoders, feature_names)
    """
    df = load_and_clean_stroke_data(filepath)
    X, y, feature_names, label_encoders = encode_stroke_features(df)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    # Scaling
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    elif scaler_type == 'robust':
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # SMOTE untuk handle imbalance
    if apply_smote:
        smote = SMOTE(random_state=42)
        X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, label_encoders, feature_names
