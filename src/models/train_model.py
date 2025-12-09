"""Utility helpers for model training and evaluation."""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

from .decision_tree_model import features

def split_data(df: pd.DataFrame):
    """Split dataframe into train/validation/test sets."""
    prediction_column = "Churn"

    X = df.drop(prediction_column, axis=1)
    y = df[prediction_column]
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=123, stratify=y_temp
    )
    return X_train[features], X_val[features], X_test[features], y_train, y_val, y_test


def plot_roc_curve(y_true, y_score, label: str) -> float:
    """Plot a ROC curve and return the AUC."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {label} (AUC={auc:.3f})")
    plt.legend()
    plt.tight_layout()
    plt.show()
    return auc

def plot_all_roc_curve(X_test, y_test, models):
    """Plot a ROC curve and return the AUC."""
    plt.figure()
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {"All Models"}")

    for model in models:
        test_prob = model["model"].predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, test_prob)
        auc = roc_auc_score(y_test, test_prob)
        plt.plot(fpr, tpr, label=f"{model["name"]}: AUC = {auc:.3f}")
        plt.plot([0, 1], [0, 1], "--", color=model["color"][:-1].lower())

    plt.legend()
    plt.tight_layout()
    plt.show()

def output_metrics(X_test, y_test, models):
    """
    Output the metrics for each model.
    """
    print(X_test, y_test)
    for model in models:
        test_prob = model["model"].predict(X_test)
        accuracy = metrics.accuracy_score(y_test, test_prob)
        recall = metrics.recall_score(y_test, test_prob)
        precision = metrics.precision_score(y_test, test_prob)
        [[TN, FP], _] = metrics.confusion_matrix(y_test, test_prob) 
        specificity = TN / (TN + FP)
        f1_score = metrics.f1_score(y_test, test_prob)

        print(f"---------{model["name"]}---------")
        print("---------Testing performance---------")
        print(f"{accuracy=}")
        print(f"{recall=}")
        print(f"{precision=}")
        print(f"{specificity=}")
        print(f"{f1_score=}")
