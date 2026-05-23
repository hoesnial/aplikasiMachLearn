"""
Modul utilitas untuk model Machine Learning.
Berisi fungsi untuk membuat dan melatih model klasifikasi.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_auc_score
)
import numpy as np


# Daftar model yang tersedia
AVAILABLE_MODELS = {
    "Logistic Regression": "logistic_regression",
    "Decision Tree": "decision_tree",
    "Random Forest": "random_forest",
    "W-KNN (Weighted KNN)": "wknn",
    "XGBoost": "xgboost",
}

# Model terbaik default per dataset
BEST_MODEL_PER_DATASET = {
    "stroke": "logistic_regression",
    "pasien_treatment": "random_forest",
    "bpjs": "random_forest",
    "jantung": "random_forest",
    "liver": "random_forest",
    "lainnya": "random_forest",
}


def get_model(model_key: str):
    """
    Mengembalikan instance model berdasarkan key.
    """
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "decision_tree": DecisionTreeClassifier(random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "wknn": KNeighborsClassifier(n_neighbors=5, weights='distance'),
        "xgboost": XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        ),
    }
    return models.get(model_key)


def train_and_evaluate(model, X_train, X_test, y_train, y_test):
    """
    Melatih model dan mengembalikan metrik evaluasi.
    Returns: dict dengan metrik evaluasi
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Probabilitas untuk ROC AUC
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
    except Exception:
        roc_auc = None

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc,
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "y_pred": y_pred,
    }

    return metrics


def get_model_display_name(model_key: str) -> str:
    """Mendapatkan nama tampilan model dari key."""
    for name, key in AVAILABLE_MODELS.items():
        if key == model_key:
            return name
    return model_key
