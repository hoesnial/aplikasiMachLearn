"""
Modul utilitas untuk model Machine Learning (binary classification).
Base model utama untuk eksperimen: Extra Trees Classifier.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_auc_score
)


# Daftar model yang tersedia
AVAILABLE_MODELS = {
    "Logistic Regression": "logistic_regression",
    "Decision Tree": "decision_tree",
    "Extra Trees": "extra_trees",
    "W-KNN (Weighted KNN)": "wknn",
    "XGBoost": "xgboost",
}

# Model default per dataset (dapat diubah berdasarkan hasil eksperimen)
BEST_MODEL_PER_DATASET = {
    "cardio": "extra_trees",
    "pasien_treatment": "extra_trees",
    "bpjs": "extra_trees",
    "jantung": "extra_trees",
    "liver": "extra_trees",
    "lainnya": "extra_trees",
}


def get_model(model_key: str):
    """Mengembalikan instance model berdasarkan key."""
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "decision_tree": DecisionTreeClassifier(random_state=42),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=200, random_state=42, n_jobs=-1
        ),
        "wknn": KNeighborsClassifier(n_neighbors=5, weights='distance'),
        "xgboost": XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss',
        ),
    }
    return models.get(model_key)


def train_and_evaluate(model, X_train, X_test, y_train, y_test):
    """
    Melatih model klasifikasi biner dan mengembalikan metrik evaluasi.
    Returns: dict dengan metrik evaluasi
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Probabilitas untuk ROC AUC
    roc_auc = None
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
        "classification_report": classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        ),
        "y_pred": y_pred,
    }

    return metrics


def get_model_display_name(model_key: str) -> str:
    """Mendapatkan nama tampilan model dari key."""
    for name, key in AVAILABLE_MODELS.items():
        if key == model_key:
            return name
    return model_key
