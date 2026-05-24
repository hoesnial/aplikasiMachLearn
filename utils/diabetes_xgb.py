"""
Modul utilitas untuk model XGBoost diabetes.
Berisi preprocessing otomatis, training, evaluasi, penyimpanan model,
dan helper prediksi untuk halaman Streamlit.
"""

from __future__ import annotations

import os
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBClassifier
except ImportError as exc:  # pragma: no cover - handled at runtime in Streamlit
    XGBClassifier = None  # type: ignore[assignment]
    XGBOOST_IMPORT_ERROR = exc
else:
    XGBOOST_IMPORT_ERROR = None


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "diabetes_dataset.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "diabetes_xgb_model.joblib")
TARGET_COLUMN = "diabetes"
DEFAULT_EXPERIMENTS = 15
EXPERIMENT_CHOICES = [5, 10, 15, 20]


def load_diabetes_data(filepath: str = DATASET_PATH) -> pd.DataFrame:
    """Load dataset diabetes dan lakukan pembersihan dasar."""
    df = pd.read_csv(filepath)
    df = df.replace(r"^\s*$", np.nan, regex=True)
    df = df.drop_duplicates()

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Kolom target '{TARGET_COLUMN}' tidak ditemukan.")

    df = df.dropna(subset=[TARGET_COLUMN])
    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    df = df.dropna(subset=[TARGET_COLUMN])
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    return df


def _build_preprocessor(X: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    """Bangun preprocessing untuk kolom numerik dan kategorikal."""
    categorical_columns = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_columns = [column for column in X.columns if column not in categorical_columns]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_columns),
            ("categorical", categorical_transformer, categorical_columns),
        ],
        remainder="drop",
    )

    return preprocessor, numeric_columns, categorical_columns


def _calculate_metrics(y_test: pd.Series, y_pred: np.ndarray) -> dict[str, Any]:
    """Hitung metrik evaluasi utama untuk klasifikasi biner."""
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(
            y_test,
            y_pred,
            output_dict=True,
            zero_division=0,
        ),
    }


def _build_model(scale_pos_weight: float = 1.0):
    """Bangun instance XGBClassifier dengan parameter yang stabil."""
    if XGBClassifier is None:
        raise ImportError(
            "xgboost belum terpasang. Jalankan: pip install xgboost"
        ) from XGBOOST_IMPORT_ERROR

    return XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
    )


def _build_bundle(
    pipeline: Pipeline,
    feature_columns: list[str],
    numeric_columns: list[str],
    categorical_columns: list[str],
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    y_pred: np.ndarray,
) -> dict[str, Any]:
    """Buat ringkasan hasil training yang siap disimpan dan dipakai ulang."""
    metrics = _calculate_metrics(y_test, y_pred)

    return {
        "pipeline": pipeline,
        "metrics": metrics,
        "feature_columns": feature_columns,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "train_shape": x_train.shape,
        "test_shape": x_test.shape,
        "class_distribution": y_train.value_counts().to_dict(),
        "test_distribution": y_test.value_counts().to_dict(),
    }


def _build_experiment_summary(
    experiment_no: int,
    random_state: int,
    result: dict[str, Any],
    is_best: bool = False,
) -> dict[str, Any]:
    """Ambil ringkasan eksperimen yang cocok ditampilkan di tabel Streamlit."""
    metrics = result["metrics"]
    return {
        "experiment_no": experiment_no,
        "random_state": random_state,
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1_score"],
        "is_best": is_best,
    }


def _is_better_candidate(candidate: dict[str, Any], current_best: dict[str, Any] | None) -> bool:
    """Bandingkan kandidat berdasarkan accuracy lalu f1-score."""
    if current_best is None:
        return True

    candidate_metrics = candidate["metrics"]
    best_metrics = current_best["metrics"]

    if candidate_metrics["accuracy"] > best_metrics["accuracy"]:
        return True
    if candidate_metrics["accuracy"] < best_metrics["accuracy"]:
        return False

    return candidate_metrics["f1_score"] > best_metrics["f1_score"]


def _extract_pipeline(bundle: dict[str, Any]):
    """Ambil pipeline yang dipakai untuk prediksi dari bundle hasil training."""
    if "best_pipeline" in bundle:
        return bundle["best_pipeline"]
    return bundle["pipeline"]


def _build_single_experiment(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int,
    test_size: float,
) -> dict[str, Any]:
    """Latih satu eksperimen XGBoost dengan random_state tertentu."""
    x_train, x_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    preprocessor, numeric_columns, categorical_columns = _build_preprocessor(X)
    positive_count = int((y_train == 1).sum())
    negative_count = int((y_train == 0).sum())
    scale_pos_weight = (negative_count / positive_count) if positive_count > 0 else 1.0

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", _build_model(scale_pos_weight=scale_pos_weight)),
        ]
    )

    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_test)

    result = _build_bundle(
        pipeline=pipeline,
        feature_columns=X.columns.tolist(),
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        y_pred=y_pred,
    )

    return result


def run_multiple_experiments(
    num_experiments: int = DEFAULT_EXPERIMENTS,
    test_size: float = 0.2,
    base_random_state: int = 42,
    save_model: bool = True,
) -> dict[str, Any]:
    """Latih XGBoost beberapa kali dan pilih model terbaik.

    Model terbaik ditentukan berdasarkan accuracy tertinggi.
    Jika accuracy sama, dipilih f1-score tertinggi.
    """
    if num_experiments < 1:
        raise ValueError("num_experiments harus bernilai minimal 1.")

    df = load_diabetes_data()
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    experiments: list[dict[str, Any]] = []
    best_result: dict[str, Any] | None = None
    best_experiment_no = 0
    best_random_state = 0

    for experiment_no in range(1, num_experiments + 1):
        random_state = base_random_state + experiment_no - 1
        result = _build_single_experiment(X, y, random_state=random_state, test_size=test_size)
        result["experiment_no"] = experiment_no
        result["random_state"] = random_state
        experiments.append(result)

        if _is_better_candidate(result, best_result):
            best_result = result
            best_experiment_no = experiment_no
            best_random_state = random_state

    if best_result is None:
        raise RuntimeError("Gagal mendapatkan hasil eksperimen terbaik.")

    experiments_summary = pd.DataFrame(
        [
            _build_experiment_summary(
                experiment_no=item["experiment_no"],
                random_state=item["random_state"],
                result=item,
                is_best=item["experiment_no"] == best_experiment_no,
            )
            for item in experiments
        ]
    )
    experiments_summary["accuracy_pct"] = experiments_summary["accuracy"] * 100
    experiments_summary["precision_pct"] = experiments_summary["precision"] * 100
    experiments_summary["recall_pct"] = experiments_summary["recall"] * 100
    experiments_summary["f1_score_pct"] = experiments_summary["f1_score"] * 100

    best_payload = {
        "best_pipeline": best_result["pipeline"],
        "best_metrics": best_result["metrics"],
        "best_experiment_no": best_experiment_no,
        "best_random_state": best_random_state,
        "best_accuracy": best_result["metrics"]["accuracy"],
        "best_f1_score": best_result["metrics"]["f1_score"],
        "best_confusion_matrix": best_result["metrics"]["confusion_matrix"],
        "best_classification_report": best_result["metrics"]["classification_report"],
        "experiments_summary": experiments_summary,
        "num_experiments": num_experiments,
        "test_size": test_size,
        "base_random_state": base_random_state,
        "feature_columns": best_result["feature_columns"],
        "numeric_columns": best_result["numeric_columns"],
        "categorical_columns": best_result["categorical_columns"],
        "train_shape": best_result["train_shape"],
        "test_shape": best_result["test_shape"],
        "class_distribution": best_result["class_distribution"],
        "test_distribution": best_result["test_distribution"],
    }

    if save_model:
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(best_payload, MODEL_PATH)

    return best_payload


def train_diabetes_xgb_model(
    test_size: float = 0.2,
    random_state: int = 42,
    save_model: bool = True,
) -> dict[str, Any]:
    """Backward-compatible wrapper untuk tetap mendukung pemanggilan lama.

    Fungsi ini sekarang menjalankan multiple experiment testing dengan satu
    random_state sebagai baseline sehingga hasilnya tetap berupa payload best model.
    """
    return run_multiple_experiments(
        num_experiments=1,
        test_size=test_size,
        base_random_state=random_state,
        save_model=save_model,
    )


def get_best_model(experiment_bundle: dict[str, Any]) -> dict[str, Any]:
    """Ambil informasi model terbaik dari hasil multiple experiment."""
    return {
        "pipeline": _extract_pipeline(experiment_bundle),
        "best_experiment_no": experiment_bundle["best_experiment_no"],
        "best_random_state": experiment_bundle["best_random_state"],
        "best_accuracy": experiment_bundle["best_accuracy"],
        "best_f1_score": experiment_bundle["best_f1_score"],
        "best_confusion_matrix": experiment_bundle["best_confusion_matrix"],
        "best_classification_report": experiment_bundle["best_classification_report"],
    }


def load_or_train_diabetes_model(
    force_retrain: bool = False,
    num_experiments: int = DEFAULT_EXPERIMENTS,
) -> dict[str, Any]:
    """Load model dari disk atau latih ulang jika belum tersedia."""
    if not force_retrain and os.path.exists(MODEL_PATH):
        try:
            loaded_bundle = joblib.load(MODEL_PATH)
            if "best_pipeline" in loaded_bundle:
                return loaded_bundle
        except Exception:
            pass

    return run_multiple_experiments(num_experiments=num_experiments, save_model=True)


def predict_diabetes(bundle: dict[str, Any], input_data: dict[str, Any]) -> tuple[int, np.ndarray]:
    """Prediksi satu baris input diabetes menggunakan bundle model yang tersimpan."""
    pipeline = _extract_pipeline(bundle)
    input_frame = pd.DataFrame([input_data])

    prediction = int(pipeline.predict(input_frame)[0])
    probability = pipeline.predict_proba(input_frame)[0]

    return prediction, probability
