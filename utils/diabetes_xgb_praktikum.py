"""Pipeline XGBoost diabetes untuk praktikum machine learning.

Modul ini menyediakan alur lengkap:
- load dan cek missing value
- handling missing value
- encoding categorical
- deteksi outlier dengan IQR
- training dua dataset: tanpa dan dengan penghapusan outlier
- evaluasi model dengan accuracy, precision, recall, F1, confusion matrix, dan ROC AUC
- feature importance
- visualisasi dan kesimpulan otomatis
"""

from __future__ import annotations

import io
import os
from typing import Any

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
except ImportError:  # pragma: no cover - optional dependency
    SMOTE = None  # type: ignore[assignment]
    ImbPipeline = Pipeline  # type: ignore[assignment]

try:
    from xgboost import XGBClassifier
except ImportError as exc:  # pragma: no cover - handled in Streamlit
    XGBClassifier = None  # type: ignore[assignment]
    XGBOOST_IMPORT_ERROR = exc
else:
    XGBOOST_IMPORT_ERROR = None


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "diabetes_dataset.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "diabetes_xgb_praktikum.joblib")
TARGET_COLUMN = "diabetes"
RANDOM_STATE = 42
TEST_SIZE = 0.2
BUNDLE_VERSION = 2
IMBALANCE_RATIO_THRESHOLD = 1.25


def load_diabetes_data(filepath: str = DATASET_PATH) -> pd.DataFrame:
    """Load dataset dan rapikan nilai kosong, tipe data, dan duplikat."""
    df = pd.read_csv(filepath)
    df = df.replace(r"^\s*$", np.nan, regex=True)
    df = df.drop_duplicates().reset_index(drop=True)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Kolom target '{TARGET_COLUMN}' tidak ditemukan.")

    numeric_columns = ["age", "hypertension", "heart_disease", "bmi", "HbA1c_level", "blood_glucose_level", TARGET_COLUMN]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    categorical_columns = [column for column in df.columns if column not in numeric_columns]
    for column in categorical_columns:
        df[column] = df[column].replace(["", "None", "none", "nan", "NaN"], np.nan)

    df = df.dropna(subset=[TARGET_COLUMN])
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
    return df


def validate_dataset(df: pd.DataFrame, required_columns: list[str] | None = None) -> None:
    """Validasi dasar dataset sebelum training."""
    if df is None or df.empty:
        raise ValueError("Dataset kosong. Silakan upload atau pilih dataset yang berisi data.")

    required = required_columns or [TARGET_COLUMN]
    missing_columns = [column for column in required if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Kolom tidak ditemukan: {', '.join(missing_columns)}")


def validate_uploaded_file(filename: str | None) -> None:
    """Validasi ekstensi file upload."""
    if not filename:
        raise ValueError("File upload tidak valid atau belum dipilih.")

    if not filename.lower().endswith(".csv"):
        raise ValueError("File upload tidak valid. Gunakan file CSV.")


def split_feature_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Pisahkan kolom numerik dan kategorikal."""
    categorical_columns = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    numeric_columns = [column for column in X.columns if column not in categorical_columns]
    return numeric_columns, categorical_columns


def analyze_imbalance(df: pd.DataFrame) -> dict[str, Any]:
    """Analisis distribusi kelas dan status imbalance."""
    validate_dataset(df, [TARGET_COLUMN])
    class_counts = df[TARGET_COLUMN].value_counts().sort_index()
    if len(class_counts) < 2:
        raise ValueError("Dataset harus memiliki minimal dua kelas untuk klasifikasi.")

    majority = class_counts.max()
    minority = class_counts.min()
    ratio = float(majority / minority) if minority > 0 else float("inf")

    return {
        "class_counts": class_counts,
        "majority_class": int(class_counts.idxmax()),
        "minority_class": int(class_counts.idxmin()),
        "imbalance_ratio": ratio,
        "is_imbalanced": ratio >= IMBALANCE_RATIO_THRESHOLD,
    }


def summarize_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Ringkas jumlah missing value per kolom."""
    summary = pd.DataFrame({"missing_values": df.isna().sum()})
    summary["missing_percentage"] = (summary["missing_values"] / len(df) * 100).round(2)
    return summary.sort_values("missing_values", ascending=False)


def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Imputasi missing value untuk kolom numerik dan kategorikal."""
    cleaned = df.copy()
    feature_columns = [column for column in cleaned.columns if column != TARGET_COLUMN]
    numeric_columns, categorical_columns = split_feature_types(cleaned[feature_columns])

    for column in numeric_columns:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
        median_value = cleaned[column].median()
        cleaned[column] = cleaned[column].fillna(median_value)

    for column in categorical_columns:
        mode_values = cleaned[column].mode(dropna=True)
        fallback_value = mode_values.iloc[0] if not mode_values.empty else "Unknown"
        cleaned[column] = cleaned[column].fillna(fallback_value).astype(str)

    return cleaned


def calculate_iqr_summary(df: pd.DataFrame, numeric_columns: list[str]) -> pd.DataFrame:
    """Hitung Q1, Q3, IQR, dan jumlah outlier per kolom."""
    rows: list[dict[str, Any]] = []

    for column in numeric_columns:
        series = pd.to_numeric(df[column], errors="coerce")
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outlier_mask = (series < lower_bound) | (series > upper_bound)
        outlier_count = int(outlier_mask.sum())
        rows.append(
            {
                "feature": column,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "outlier_count": outlier_count,
                "outlier_percentage": round((outlier_count / len(series)) * 100, 2) if len(series) else 0.0,
            }
        )

    return pd.DataFrame(rows)


def remove_outliers_iqr(df: pd.DataFrame, numeric_columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hapus baris yang berada di luar batas IQR pada kolom numerik."""
    summary = calculate_iqr_summary(df, numeric_columns)
    mask = pd.Series(True, index=df.index)

    for _, row in summary.iterrows():
        column = row["feature"]
        series = pd.to_numeric(df[column], errors="coerce")
        lower_bound = row["lower_bound"]
        upper_bound = row["upper_bound"]
        column_mask = series.between(lower_bound, upper_bound) | series.isna()
        mask &= column_mask

    filtered_df = df.loc[mask].reset_index(drop=True)
    return filtered_df, summary


def build_preprocessor(
    numeric_columns: list[str],
    categorical_columns: list[str],
    apply_scaling: bool = False,
    encoding: str = "ordinal",
) -> ColumnTransformer:
    """Bangun preprocessing untuk fitur numerik dan kategorikal."""
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if apply_scaling:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(steps=numeric_steps)

    if encoding == "onehot":
        categorical_encoder: Any = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    else:
        categorical_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", categorical_encoder),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )


def build_model(random_state: int, scale_pos_weight: float) -> XGBClassifier:
    """Bangun XGBClassifier dengan parameter dasar yang stabil."""
    if XGBClassifier is None:
        raise ImportError("xgboost belum terpasang. Jalankan: pip install xgboost") from XGBOOST_IMPORT_ERROR

    return XGBClassifier(
        n_estimators=250,
        learning_rate=0.05,
        max_depth=4,
        random_state=random_state,
        subsample=0.9,
        colsample_bytree=0.9,
        n_jobs=-1,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
    )


def build_baseline_model(random_state: int) -> LogisticRegression:
    """Bangun baseline model sederhana untuk perbandingan metode."""
    return LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=random_state,
    )


def build_model_pipeline(
    model_name: str,
    numeric_columns: list[str],
    categorical_columns: list[str],
    apply_scaling: bool,
    random_state: int,
    scale_pos_weight: float = 1.0,
    use_smote: bool = False,
) -> ImbPipeline | Pipeline:
    """Bangun pipeline model untuk XGBoost atau baseline."""
    model_name = model_name.lower()
    if model_name == "xgb":
        preprocessor = build_preprocessor(numeric_columns, categorical_columns, apply_scaling=apply_scaling, encoding="ordinal")
        model = build_model(random_state=random_state, scale_pos_weight=scale_pos_weight)
    elif model_name == "baseline":
        preprocessor = build_preprocessor(numeric_columns, categorical_columns, apply_scaling=True, encoding="onehot")
        model = build_baseline_model(random_state=random_state)
    else:
        raise ValueError(f"model_name tidak dikenal: {model_name}")

    steps: list[tuple[str, Any]] = [("preprocessor", preprocessor)]
    if use_smote and SMOTE is not None:
        steps.append(("smote", SMOTE(random_state=random_state)))
    steps.append(("model", model))

    if use_smote and SMOTE is not None:
        return ImbPipeline(steps=steps)
    return Pipeline(steps=steps)


def _normalize_xgb_params(params: dict[str, Any] | None) -> dict[str, Any]:
    """Ubah parameter search CV menjadi parameter model XGBoost."""
    if not params:
        return {}
    normalized: dict[str, Any] = {}
    for key, value in params.items():
        if key.startswith("model__"):
            normalized[key.replace("model__", "")] = value
    return normalized


def _clean_feature_name(feature_name: str) -> str:
    if feature_name.startswith("numeric__"):
        return feature_name.replace("numeric__", "")
    if feature_name.startswith("categorical__"):
        return feature_name.replace("categorical__", "")
    return feature_name


def build_feature_importance_table(pipeline: Pipeline) -> pd.DataFrame:
    """Ambil ranking feature importance dari model XGBoost."""
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    importances = pipeline.named_steps["model"].feature_importances_
    importance_df = pd.DataFrame(
        {
            "feature": [_clean_feature_name(name) for name in feature_names],
            "importance": importances,
        }
    ).sort_values("importance", ascending=False)
    return importance_df.reset_index(drop=True)


def interpret_feature_importance(importance_df: pd.DataFrame, top_n: int = 5) -> list[str]:
    """Buat interpretasi akademik sederhana dari fitur paling penting."""
    if importance_df.empty:
        return ["Belum tersedia feature importance untuk diinterpretasikan."]

    top_features = importance_df.head(top_n)
    interpretations: list[str] = []
    for _, row in top_features.iterrows():
        feature = row["feature"]
        score = float(row["importance"])
        interpretations.append(
            f"Fitur {feature} memiliki kontribusi relatif tinggi ({score:.3f}) terhadap keputusan model, sehingga perubahan nilai pada fitur ini cenderung memengaruhi probabilitas prediksi diabetes."
        )
    return interpretations


def summarize_confusion_matrix(confusion_matrix_values: np.ndarray) -> dict[str, int]:
    """Ubah confusion matrix menjadi TP, TN, FP, FN."""
    tn, fp, fn, tp = confusion_matrix_values.ravel()
    return {
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def interpret_confusion_matrix(confusion_summary: dict[str, int]) -> str:
    """Berikan interpretasi medis sederhana dari confusion matrix."""
    return (
        f"False Negative berjumlah {confusion_summary['false_negative']} dan perlu diperhatikan karena pasien diabetes dapat tidak terdeteksi. "
        f"False Positive berjumlah {confusion_summary['false_positive']}, yang berarti pemeriksaan lanjutan mungkin dilakukan meskipun hasil awal model menunjukkan risiko. "
        f"True Positive = {confusion_summary['true_positive']} dan True Negative = {confusion_summary['true_negative']} menunjukkan prediksi yang tepat." 
    )


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Konversi DataFrame menjadi bytes CSV untuk download."""
    return df.to_csv(index=False).encode("utf-8")


def text_to_bytes(text: str) -> bytes:
    """Konversi teks ke bytes UTF-8."""
    return text.encode("utf-8")


def figure_to_png_bytes(fig: plt.Figure) -> bytes:
    """Simpan figure matplotlib ke bytes PNG."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=200, bbox_inches="tight")
    buffer.seek(0)
    return buffer.getvalue()


def summarize_class_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Buat tabel distribusi kelas untuk analisis imbalance."""
    class_counts = df[TARGET_COLUMN].value_counts().sort_index()
    return pd.DataFrame(
        {
            "class": class_counts.index.astype(int),
            "count": class_counts.values,
            "percentage": (class_counts.values / class_counts.values.sum() * 100).round(2),
        }
    )


def evaluate_pipeline(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    """Hitung metrik evaluasi dan keluaran prediksi."""
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    try:
        roc_auc = roc_auc_score(y_test, y_prob)
    except ValueError:
        roc_auc = float("nan")

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc,
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


def train_and_evaluate_model(
    df: pd.DataFrame,
    model_name: str,
    apply_scaling: bool,
    random_state: int = RANDOM_STATE,
    test_size: float = TEST_SIZE,
    use_smote: bool = False,
    model_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Latih satu model dan kembalikan metrik evaluasi lengkap."""
    validate_dataset(df, [TARGET_COLUMN])
    working_df = df.copy().reset_index(drop=True)
    feature_columns = [column for column in working_df.columns if column != TARGET_COLUMN]
    numeric_columns, categorical_columns = split_feature_types(working_df[feature_columns])

    X = working_df.drop(columns=[TARGET_COLUMN])
    y = working_df[TARGET_COLUMN].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    positive_count = int((y_train == 1).sum())
    negative_count = int((y_train == 0).sum())
    scale_pos_weight = (negative_count / positive_count) if positive_count > 0 else 1.0

    if model_name.lower() == "xgb":
        preprocessor = build_preprocessor(numeric_columns, categorical_columns, apply_scaling=apply_scaling, encoding="ordinal")
        model = build_model(random_state=random_state, scale_pos_weight=scale_pos_weight)
    elif model_name.lower() == "baseline":
        preprocessor = build_preprocessor(numeric_columns, categorical_columns, apply_scaling=True, encoding="onehot")
        model = build_baseline_model(random_state=random_state)
    else:
        raise ValueError(f"model_name tidak dikenal: {model_name}")

    steps: list[tuple[str, Any]] = [("preprocessor", preprocessor)]
    if use_smote and SMOTE is not None:
        steps.append(("smote", SMOTE(random_state=random_state)))
    steps.append(("model", model))
    pipeline = ImbPipeline(steps=steps) if (use_smote and SMOTE is not None) else Pipeline(steps=steps)

    if model_name.lower() == "xgb":
        normalized_params = _normalize_xgb_params(model_params)
        if normalized_params:
            pipeline.named_steps["model"].set_params(**normalized_params)

    try:
        pipeline.fit(X_train, y_train)
        metrics = evaluate_pipeline(pipeline, X_test, y_test)
    except Exception as exc:
        raise RuntimeError(f"Gagal training model {model_name}: {exc}") from exc

    result = {
        "model_name": model_name,
        "dataframe": working_df,
        "feature_columns": feature_columns,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "pipeline": pipeline,
        "metrics": metrics,
        "train_shape": X_train.shape,
        "test_shape": X_test.shape,
        "class_distribution": y.value_counts().sort_index().to_dict(),
        "train_distribution": y_train.value_counts().sort_index().to_dict(),
        "test_distribution": y_test.value_counts().sort_index().to_dict(),
        "y_test": y_test.reset_index(drop=True),
        "y_prob": metrics["y_prob"],
        "use_smote": use_smote,
    }

    if model_name.lower() == "xgb":
        result["feature_importance"] = build_feature_importance_table(pipeline)
    else:
        result["feature_importance"] = pd.DataFrame(columns=["feature", "importance"])

    result["confusion_summary"] = summarize_confusion_matrix(metrics["confusion_matrix"])
    result["confusion_interpretation"] = interpret_confusion_matrix(result["confusion_summary"])
    return result


def tune_xgb_model(
    df: pd.DataFrame,
    apply_scaling: bool,
    random_state: int = RANDOM_STATE,
    cv: int = 5,
    n_iter: int = 12,
) -> dict[str, Any]:
    """Optimasi hyperparameter XGBoost menggunakan RandomizedSearchCV."""
    validate_dataset(df, [TARGET_COLUMN])
    feature_columns = [column for column in df.columns if column != TARGET_COLUMN]
    numeric_columns, categorical_columns = split_feature_types(df[feature_columns])

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].astype(int)

    positive_count = int((y == 1).sum())
    negative_count = int((y == 0).sum())
    scale_pos_weight = (negative_count / positive_count) if positive_count > 0 else 1.0

    pipeline = build_model_pipeline(
        model_name="xgb",
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        apply_scaling=apply_scaling,
        random_state=random_state,
        use_smote=False,
    )

    param_distributions = {
        "model__n_estimators": [100, 150, 200, 250, 300],
        "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "model__max_depth": [3, 4, 5, 6],
        "model__subsample": [0.7, 0.8, 0.9, 1.0],
    }

    pipeline.named_steps["model"].set_params(scale_pos_weight=scale_pos_weight)
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state),
        random_state=random_state,
        n_jobs=-1,
        verbose=0,
    )

    search.fit(X, y)
    best_pipeline = search.best_estimator_

    return {
        "best_pipeline": best_pipeline,
        "best_params": search.best_params_,
        "best_model_params": _normalize_xgb_params(search.best_params_),
        "best_score": float(search.best_score_),
        "feature_columns": feature_columns,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
    }


def run_cross_validation(
    df: pd.DataFrame,
    numeric_columns: list[str],
    categorical_columns: list[str],
    apply_scaling: bool = False,
    cv: int = 5,
    random_state: int = RANDOM_STATE,
) -> dict[str, float]:
    """Jalankan StratifiedKFold cross-validation dan kembalikan mean skor."""
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].astype(int)

    positive_count = int((y == 1).sum())
    negative_count = int((y == 0).sum())
    scale_pos_weight = (negative_count / positive_count) if positive_count > 0 else 1.0

    preprocessor = build_preprocessor(numeric_columns, categorical_columns, apply_scaling=apply_scaling)
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", build_model(random_state=random_state, scale_pos_weight=scale_pos_weight)),
        ]
    )

    cv_split = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    try:
        scores = cross_validate(pipeline, X, y, cv=cv_split, scoring=scoring, n_jobs=-1, error_score="raise")
    except Exception:
        # Jika scoring ROC AUC gagal karena kelas tunggal pada fold, coba tanpa roc_auc
        scoring = {k: v for k, v in scoring.items() if k != "roc_auc"}
        scores = cross_validate(pipeline, X, y, cv=cv_split, scoring=scoring, n_jobs=-1, error_score="raise")

    results = {}
    for key in scores:
        if key.startswith("test_"):
            short = key.replace("test_", "")
            results[f"cv_{short}_mean"] = float(np.nanmean(scores[key]))
            results[f"cv_{short}_std"] = float(np.nanstd(scores[key]))

    return results


def plot_numeric_histograms(df: pd.DataFrame, numeric_columns: list[str], title: str) -> plt.Figure:
    """Tampilkan histogram distribusi data numerik."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()
    plot_columns = numeric_columns[: len(axes)]

    for index, column in enumerate(plot_columns):
        sns.histplot(df[column], kde=True, bins=20, ax=axes[index], color="#2563eb")
        axes[index].set_title(column)
        axes[index].set_xlabel(column)
        axes[index].set_ylabel("Frequency")

    for index in range(len(plot_columns), len(axes)):
        axes[index].axis("off")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, numeric_columns: list[str], title: str) -> plt.Figure:
    """Tampilkan correlation heatmap."""
    fig, ax = plt.subplots(figsize=(10, 7))
    correlation_matrix = df[numeric_columns].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title(title, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_boxplot(df: pd.DataFrame, numeric_columns: list[str], title: str) -> plt.Figure:
    """Tampilkan boxplot untuk mendeteksi outlier."""
    melted = df[numeric_columns].melt(var_name="feature", value_name="value")
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.boxplot(data=melted, x="feature", y="value", ax=ax, color="#93c5fd")
    ax.set_title(title, fontweight="bold")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    return fig


def plot_confusion_matrix(confusion_matrix_values: np.ndarray, title: str) -> plt.Figure:
    """Tampilkan confusion matrix."""
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    cm_df = pd.DataFrame(
        confusion_matrix_values,
        index=["Actual 0", "Actual 1"],
        columns=["Predicted 0", "Predicted 1"],
    )
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Greens", cbar=False, ax=ax)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    return fig


def plot_roc_curve(y_test: pd.Series, y_prob: np.ndarray, title: str) -> plt.Figure:
    """Tampilkan ROC curve."""
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label="ROC Curve", color="#ef4444", linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="#64748b", label="Random Guess")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(frameon=False)
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_feature_importance(feature_importance_df: pd.DataFrame, title: str) -> plt.Figure:
    """Tampilkan feature importance dalam bentuk bar chart."""
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    top_features = feature_importance_df.head(10).sort_values("importance", ascending=True)
    ax.barh(top_features["feature"], top_features["importance"], color="#1d4ed8")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    return fig


def plot_accuracy_comparison(comparison_table: pd.DataFrame) -> plt.Figure:
    """Tampilkan perbandingan accuracy sebelum dan sesudah preprocessing."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(comparison_table["dataset"], comparison_table["accuracy"] * 100, color=["#2563eb", "#ef4444"])
    ax.set_title("Perbandingan Accuracy", fontweight="bold")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Dataset")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_class_countplot(df: pd.DataFrame, title: str) -> plt.Figure:
    """Tampilkan countplot distribusi kelas target."""
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x=TARGET_COLUMN, data=df, palette=["#2563eb", "#ef4444"], ax=ax)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    fig.tight_layout()
    return fig


def _build_variant_result(
    dataset_name: str,
    df: pd.DataFrame,
    apply_scaling: bool,
    test_size: float,
    random_state: int,
    use_smote: bool = False,
) -> dict[str, Any]:
    """Latih dan evaluasi satu variasi dataset."""
    working_df = df.copy().reset_index(drop=True)
    feature_columns = [column for column in working_df.columns if column != TARGET_COLUMN]
    numeric_columns, categorical_columns = split_feature_types(working_df[feature_columns])

    X = working_df.drop(columns=[TARGET_COLUMN])
    y = working_df[TARGET_COLUMN].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    positive_count = int((y_train == 1).sum())
    negative_count = int((y_train == 0).sum())
    scale_pos_weight = (negative_count / positive_count) if positive_count > 0 else 1.0

    pipeline = build_model_pipeline(
        model_name="xgb",
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        apply_scaling=apply_scaling,
        random_state=random_state,
        scale_pos_weight=scale_pos_weight,
        use_smote=use_smote,
    )
    pipeline.fit(X_train, y_train)
    metrics = evaluate_pipeline(pipeline, X_test, y_test)
    feature_importance_df = build_feature_importance_table(pipeline)

    result = {
        "dataset_name": dataset_name,
        "dataframe": working_df,
        "feature_columns": feature_columns,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "pipeline": pipeline,
        "metrics": metrics,
        "feature_importance": feature_importance_df,
        "train_shape": X_train.shape,
        "test_shape": X_test.shape,
        "class_distribution": y.value_counts().sort_index().to_dict(),
        "train_distribution": y_train.value_counts().sort_index().to_dict(),
        "test_distribution": y_test.value_counts().sort_index().to_dict(),
        "y_test": y_test.reset_index(drop=True),
        "y_prob": metrics["y_prob"],
    }
    result["confusion_summary"] = summarize_confusion_matrix(metrics["confusion_matrix"])
    result["confusion_interpretation"] = interpret_confusion_matrix(result["confusion_summary"])
    return result


def build_conclusion(no_outlier_result: dict[str, Any], outlier_removed_result: dict[str, Any]) -> str:
    """Buat kesimpulan otomatis berdasarkan perbandingan accuracy."""
    accuracy_without = float(no_outlier_result["metrics"]["accuracy"])
    accuracy_with_removal = float(outlier_removed_result["metrics"]["accuracy"])
    delta = accuracy_with_removal - accuracy_without

    if abs(delta) < 0.01:
        return (
            "XGBoost menunjukkan ketahanan terhadap outlier sehingga penghapusan outlier tidak selalu meningkatkan performa model. "
            "Pada eksperimen ini, perbedaan accuracy antar dataset relatif kecil, sehingga informasi pada outlier kemungkinan masih relevan untuk pola prediksi."
        )

    if delta > 0:
        return (
            "Berdasarkan hasil eksperimen, preprocessing dan handling outlier memberikan pengaruh terhadap performa model XGBoost. "
            "Model menunjukkan peningkatan accuracy setelah data dibersihkan sehingga preprocessing membantu model mengenali pola data dengan lebih baik."
        )

    return (
        "XGBoost menunjukkan ketahanan terhadap outlier sehingga penghapusan outlier tidak selalu meningkatkan performa model. "
        "Pada eksperimen ini, accuracy menurun setelah penghapusan outlier, yang mengindikasikan bahwa outlier mungkin mengandung informasi penting dan penghapusan data menyebabkan informasi berkurang."
    )


def build_comparison_bundle(
    apply_scaling: bool = False,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    use_smote: bool = False,
    save_model: bool = True,
) -> dict[str, Any]:
    """Bangun bundle lengkap untuk dataset tanpa dan dengan penghapusan outlier."""
    raw_df = load_diabetes_data()
    validate_dataset(raw_df, [TARGET_COLUMN])
    missing_summary = summarize_missing_values(raw_df)
    imputed_df = impute_missing_values(raw_df)

    imbalance_summary = analyze_imbalance(imputed_df)
    class_distribution_table = summarize_class_distribution(imputed_df)
    use_smote_effective = bool(use_smote and imbalance_summary["is_imbalanced"])

    feature_columns = [column for column in imputed_df.columns if column != TARGET_COLUMN]
    numeric_columns, categorical_columns = split_feature_types(imputed_df[feature_columns])
    outlier_summary = calculate_iqr_summary(imputed_df, numeric_columns)
    outlier_removed_df, _ = remove_outliers_iqr(imputed_df, numeric_columns)

    no_outlier_result = _build_variant_result(
        dataset_name="Tanpa Penghapusan Outlier",
        df=imputed_df,
        apply_scaling=apply_scaling,
        test_size=test_size,
        random_state=random_state,
        use_smote=use_smote_effective,
    )
    outlier_removed_result = _build_variant_result(
        dataset_name="Dengan Penghapusan Outlier",
        df=outlier_removed_df,
        apply_scaling=apply_scaling,
        test_size=test_size,
        random_state=random_state,
        use_smote=use_smote_effective,
    )

    # Cross-validation untuk kedua variasi (StratifiedKFold)
    cv_no = run_cross_validation(imputed_df, numeric_columns, categorical_columns, apply_scaling=apply_scaling, cv=5, random_state=random_state)
    cv_removed = run_cross_validation(outlier_removed_df, numeric_columns, categorical_columns, apply_scaling=apply_scaling, cv=5, random_state=random_state)

    comparison_table = pd.DataFrame(
        [
            {
                "dataset": no_outlier_result["dataset_name"],
                "rows": len(no_outlier_result["dataframe"]),
                "accuracy": no_outlier_result["metrics"]["accuracy"],
                "precision": no_outlier_result["metrics"]["precision"],
                "recall": no_outlier_result["metrics"]["recall"],
                "f1_score": no_outlier_result["metrics"]["f1_score"],
                "roc_auc": no_outlier_result["metrics"]["roc_auc"],
                **{k: v for k, v in cv_no.items()},
            },
            {
                "dataset": outlier_removed_result["dataset_name"],
                "rows": len(outlier_removed_result["dataframe"]),
                "accuracy": outlier_removed_result["metrics"]["accuracy"],
                "precision": outlier_removed_result["metrics"]["precision"],
                "recall": outlier_removed_result["metrics"]["recall"],
                "f1_score": outlier_removed_result["metrics"]["f1_score"],
                "roc_auc": outlier_removed_result["metrics"]["roc_auc"],
                **{k: v for k, v in cv_removed.items()},
            },
        ]
    )
    comparison_table["accuracy_pct"] = (comparison_table["accuracy"] * 100).round(2)
    comparison_table["precision_pct"] = (comparison_table["precision"] * 100).round(2)
    comparison_table["recall_pct"] = (comparison_table["recall"] * 100).round(2)
    comparison_table["f1_score_pct"] = (comparison_table["f1_score"] * 100).round(2)
    comparison_table["roc_auc_pct"] = (comparison_table["roc_auc"] * 100).round(2)
    # Tambahkan kolom CV (mean) jika tersedia
    for col in list(comparison_table.columns):
        if col.startswith("cv_") and col.endswith("_mean"):
            comparison_table[f"{col}_pct"] = (comparison_table[col] * 100).round(2)

    best_result = no_outlier_result
    if outlier_removed_result["metrics"]["accuracy"] > no_outlier_result["metrics"]["accuracy"]:
        best_result = outlier_removed_result
    elif outlier_removed_result["metrics"]["accuracy"] == no_outlier_result["metrics"]["accuracy"]:
        if outlier_removed_result["metrics"]["f1_score"] > no_outlier_result["metrics"]["f1_score"]:
            best_result = outlier_removed_result

    selected_dataset_result = best_result
    selected_dataset_df = selected_dataset_result["dataframe"]
    baseline_result = train_and_evaluate_model(
        selected_dataset_df,
        model_name="baseline",
        apply_scaling=True,
        random_state=random_state,
        test_size=test_size,
        use_smote=use_smote_effective,
    )
    xgb_selected_result = train_and_evaluate_model(
        selected_dataset_df,
        model_name="xgb",
        apply_scaling=apply_scaling,
        random_state=random_state,
        test_size=test_size,
        use_smote=use_smote_effective,
    )
    tuned_search = tune_xgb_model(
        selected_dataset_df,
        apply_scaling=apply_scaling,
        random_state=random_state,
        cv=5,
        n_iter=12,
    )
    tuned_result = train_and_evaluate_model(
        selected_dataset_df,
        model_name="xgb",
        apply_scaling=apply_scaling,
        random_state=random_state,
        test_size=test_size,
        use_smote=use_smote_effective,
        model_params=tuned_search["best_model_params"],
    )

    final_comparison_table = pd.DataFrame(
        [
            {
                "dataset": no_outlier_result["dataset_name"],
                "accuracy": no_outlier_result["metrics"]["accuracy"],
                "precision": no_outlier_result["metrics"]["precision"],
                "recall": no_outlier_result["metrics"]["recall"],
                "f1_score": no_outlier_result["metrics"]["f1_score"],
                "roc_auc": no_outlier_result["metrics"]["roc_auc"],
            },
            {
                "dataset": outlier_removed_result["dataset_name"],
                "accuracy": outlier_removed_result["metrics"]["accuracy"],
                "precision": outlier_removed_result["metrics"]["precision"],
                "recall": outlier_removed_result["metrics"]["recall"],
                "f1_score": outlier_removed_result["metrics"]["f1_score"],
                "roc_auc": outlier_removed_result["metrics"]["roc_auc"],
            },
        ]
    )

    final_model_comparison_table = pd.DataFrame(
        [
            {
                "model": "Baseline Logistic Regression",
                "accuracy": baseline_result["metrics"]["accuracy"],
                "precision": baseline_result["metrics"]["precision"],
                "recall": baseline_result["metrics"]["recall"],
                "f1_score": baseline_result["metrics"]["f1_score"],
                "roc_auc": baseline_result["metrics"]["roc_auc"],
            },
            {
                "model": "XGBoost Default",
                "accuracy": xgb_selected_result["metrics"]["accuracy"],
                "precision": xgb_selected_result["metrics"]["precision"],
                "recall": xgb_selected_result["metrics"]["recall"],
                "f1_score": xgb_selected_result["metrics"]["f1_score"],
                "roc_auc": xgb_selected_result["metrics"]["roc_auc"],
            },
            {
                "model": "XGBoost Tuned",
                "accuracy": tuned_result["metrics"]["accuracy"],
                "precision": tuned_result["metrics"]["precision"],
                "recall": tuned_result["metrics"]["recall"],
                "f1_score": tuned_result["metrics"]["f1_score"],
                "roc_auc": tuned_result["metrics"]["roc_auc"],
            },
        ]
    )

    best_model_row = final_model_comparison_table.sort_values(by=["accuracy", "f1_score"], ascending=False).iloc[0]
    best_model_name = str(best_model_row["model"])
    if best_model_name == "Baseline Logistic Regression":
        best_model_result = baseline_result
    elif best_model_name == "XGBoost Tuned":
        best_model_result = tuned_result
    else:
        best_model_result = xgb_selected_result

    conclusion = build_conclusion(no_outlier_result, outlier_removed_result)
    conclusion = (
        conclusion
        + " Baseline Logistic Regression digunakan sebagai pembanding sederhana, sedangkan tuning hyperparameter membantu mengevaluasi potensi peningkatan performa XGBoost.")

    feature_importance_interpretation = interpret_feature_importance(selected_dataset_result["feature_importance"], top_n=5)

    best_confusion_summary = selected_dataset_result["confusion_summary"]
    best_confusion_interpretation = selected_dataset_result["confusion_interpretation"]

    final_best_tuned = tuned_result if tuned_result["metrics"]["accuracy"] >= xgb_selected_result["metrics"]["accuracy"] else xgb_selected_result

    bundle = {
        "bundle_version": BUNDLE_VERSION,
        "apply_scaling": apply_scaling,
        "use_smote": use_smote_effective,
        "raw_data": raw_df,
        "imputed_data": imputed_df,
        "missing_summary": missing_summary,
        "imbalance_summary": imbalance_summary,
        "class_distribution_table": class_distribution_table,
        "outlier_summary": outlier_summary,
        "outlier_removed_data": outlier_removed_df,
        "comparison_table": final_comparison_table,
        "final_comparison_table": final_comparison_table,
        "final_model_comparison_table": final_model_comparison_table,
        "best_model_name": best_model_name,
        "no_outlier_result": no_outlier_result,
        "outlier_removed_result": outlier_removed_result,
        "best_result_key": "outlier_removed_result" if best_result is outlier_removed_result else "no_outlier_result",
        "best_pipeline": best_model_result["pipeline"],
        "best_metrics": best_model_result["metrics"],
        "best_feature_importance": best_model_result.get("feature_importance", pd.DataFrame(columns=["feature", "importance"])),
        "best_feature_interpretation": interpret_feature_importance(best_model_result.get("feature_importance", pd.DataFrame(columns=["feature", "importance"])), top_n=5),
        "best_feature_columns": best_model_result["feature_columns"],
        "best_numeric_columns": best_model_result["numeric_columns"],
        "best_categorical_columns": best_model_result["categorical_columns"],
        "best_train_shape": best_model_result["train_shape"],
        "best_test_shape": best_model_result["test_shape"],
        "best_class_distribution": best_model_result["class_distribution"],
        "best_train_distribution": best_model_result["train_distribution"],
        "best_test_distribution": best_model_result["test_distribution"],
        "best_confusion_summary": best_model_result["confusion_summary"],
        "best_confusion_interpretation": best_model_result["confusion_interpretation"],
        "tuned_xgb_result": tuned_result,
        "tuned_xgb_search": tuned_search,
        "baseline_result": baseline_result,
        "xgb_selected_result": xgb_selected_result,
        "conclusion": conclusion,
        "scaling_note": "StandardScaler bersifat opsional untuk XGBoost. Tree-based model ini tidak terlalu bergantung pada scaling, tetapi eksperimen tetap disediakan untuk praktikum.",
        "imbalance_note": (
            f"Distribusi kelas menunjukkan rasio imbalance sebesar {imbalance_summary['imbalance_ratio']:.2f}. "
            + (
                "SMOTE diaktifkan untuk eksperimen ini karena dataset terdeteksi imbalanced."
                if use_smote_effective
                else "SMOTE tidak diaktifkan; eksperimen dilakukan pada distribusi data asli."
            )
        ),
        "tuning_note": "RandomizedSearchCV digunakan untuk mengoptimalkan n_estimators, learning_rate, max_depth, dan subsample.",
    }

    if save_model:
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(bundle, MODEL_PATH)

    return bundle


def load_or_train_diabetes_model(
    force_retrain: bool = False,
    apply_scaling: bool = False,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    use_smote: bool = False,
) -> dict[str, Any]:
    """Load model dari disk atau latih ulang jika belum tersedia."""
    if not force_retrain and os.path.exists(MODEL_PATH):
        try:
            loaded_bundle = joblib.load(MODEL_PATH)
            if (
                loaded_bundle.get("bundle_version") == BUNDLE_VERSION
                and loaded_bundle.get("apply_scaling") == apply_scaling
                and loaded_bundle.get("use_smote", False) == use_smote
            ):
                return loaded_bundle
        except Exception:
            pass

    return build_comparison_bundle(
        apply_scaling=apply_scaling,
        test_size=test_size,
        random_state=random_state,
        use_smote=use_smote,
        save_model=True,
    )


def get_best_model(bundle: dict[str, Any]) -> dict[str, Any]:
    """Ambil ringkasan model terbaik untuk ditampilkan di halaman Streamlit."""
    best_result = bundle[bundle["best_result_key"]]
    best_model_metrics = bundle["best_metrics"]
    return {
        "pipeline": bundle["best_pipeline"],
        "best_model_name": bundle.get("best_model_name", "Unknown"),
        "best_result_key": bundle["best_result_key"],
        "best_dataset_name": best_result["dataset_name"],
        "best_accuracy": best_model_metrics["accuracy"],
        "best_precision": best_model_metrics["precision"],
        "best_recall": best_model_metrics["recall"],
        "best_f1_score": best_model_metrics["f1_score"],
        "best_roc_auc": best_model_metrics["roc_auc"],
        "best_confusion_matrix": best_model_metrics["confusion_matrix"],
        "best_classification_report": best_model_metrics["classification_report"],
        "best_feature_importance": bundle["best_feature_importance"],
        "best_y_test": best_result["y_test"],
        "best_y_prob": best_result["y_prob"],
        "best_model_metrics": best_model_metrics,
        "best_model_confusion_summary": bundle.get("best_confusion_summary", {}),
        "best_model_confusion_interpretation": bundle.get("best_confusion_interpretation", ""),
    }


def predict_diabetes(bundle: dict[str, Any], input_data: dict[str, Any]) -> tuple[int, np.ndarray]:
    """Prediksi satu pasien menggunakan pipeline terbaik."""
    pipeline = bundle["best_pipeline"]
    input_frame = pd.DataFrame([input_data])
    prediction = int(pipeline.predict(input_frame)[0])
    probability = pipeline.predict_proba(input_frame)[0]
    return prediction, probability
