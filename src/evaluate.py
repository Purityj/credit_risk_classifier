"""
Evaluation utilities for classification models in this project.

Centralized here because every model we build (baseline, neural net,
anything after) needs to be judged by the SAME set of metrics, computed
the SAME way, to be fairly comparable to each other.

Why these specific metrics and not accuracy:
Given the ~78/22 class imbalance established in EDA, a model that
predicts "no default" for everyone would score ~78% accuracy while
being useless. Precision, recall, F1, and ROC-AUC all account for
performance on the minority (default) class specifically, which is
the class that actually matters for this problem.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_classifier(y_true, y_pred, y_pred_proba, model_name: str = "Model") -> dict:
    """
    Compute the standard metric set for a binary classifier and print
    a readable summary.

    Parameters
    ----------
    y_true : array-like of true labels (0/1)
    y_pred : array-like of predicted labels (0/1) — i.e. after
        thresholding at 0.5
    y_pred_proba : array-like of predicted probabilities for class 1 —
        needed for ROC-AUC, which evaluates ranking quality across all
        thresholds, not just the default 0.5 cutoff
    model_name : label used in the printed output, so results from
        different models are easy to tell apart when comparing

    Returns
    -------
    dict of the four headline metrics, so results can be collected
    across models into a comparison table later.
    """
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred_proba)

    print(f"=== {model_name} ===")
    print(f"Precision: {precision:.4f}  (of predicted defaulters, how many actually defaulted)")
    print(f"Recall:    {recall:.4f}  (of actual defaulters, how many did we catch)")
    print(f"F1:        {f1:.4f}  (harmonic mean of precision and recall)")
    print(f"ROC-AUC:   {roc_auc:.4f}  (ranking quality across all thresholds)")
    print()
    print(classification_report(y_true, y_pred, target_names=["No Default", "Default"]))

    return {
        "model": model_name,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
    }


def plot_confusion_matrix(y_true, y_pred, model_name: str = "Model"):
    """Plot a confusion matrix with readable class labels."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Default", "Default"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix — {model_name}")
    plt.show()


def plot_roc_curve(y_true, y_pred_proba, model_name: str = "Model"):
    """Plot the ROC curve for a single model."""
    RocCurveDisplay.from_predictions(y_true, y_pred_proba, name=model_name)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
    plt.title(f"ROC Curve — {model_name}")
    plt.legend()
    plt.show()