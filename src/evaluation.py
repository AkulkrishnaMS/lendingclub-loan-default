"""Shared metrics and plotting helpers."""
from __future__ import annotations
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report, confusion_matrix,
                             mean_squared_error, r2_score, roc_auc_score, roc_curve)

from .config import RESULTS_DIR


def save_json(obj, name: str):
    RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    with open(RESULTS_DIR / name, "w") as f:
        json.dump(obj, f, indent=2, default=float)


def classification_metrics(model, splits: dict, tag: str) -> dict:
    """splits = {'train': (X, y), 'val': (X, y), 'test': (X, y)}. Prints reports; saves confusion matrix + ROC (test)."""
    out = {}
    for name, (X, y) in splits.items():
        pred, proba = model.predict(X), model.predict_proba(X)[:, 1]
        rep = classification_report(y, pred, target_names=["Default", "Paid Off"], output_dict=True, zero_division=0)
        out[name] = {"accuracy": rep["accuracy"], "roc_auc": roc_auc_score(y, proba),
                     "default": rep["Default"], "paid_off": rep["Paid Off"], "weighted_avg": rep["weighted avg"]}
        print(f"\n=== {tag} | {name} ===")
        print(confusion_matrix(y, pred))
        print(classification_report(y, pred, target_names=["Default", "Paid Off"], zero_division=0))

    Xt, yt = splits["test"]
    RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    ConfusionMatrixDisplay.from_predictions(yt, model.predict(Xt), display_labels=["Default", "Paid Off"], cmap="Blues")
    plt.title(f"{tag} - test confusion matrix"); plt.savefig(RESULTS_DIR / f"{tag}_confusion_matrix.png", dpi=150); plt.close()
    fpr, tpr, _ = roc_curve(yt, model.predict_proba(Xt)[:, 1])
    plt.plot(fpr, tpr, label=f"AUC={out['test']['roc_auc']:.3f}"); plt.plot([0, 1], [0, 1], "--", c="grey")
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.title(f"{tag} - test ROC"); plt.legend()
    plt.savefig(RESULTS_DIR / f"{tag}_roc.png", dpi=150); plt.close()
    save_json(out, f"{tag}_metrics.json")
    return out


def regression_metrics(y_true, y_pred) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    return {"mse": mse, "rmse": float(np.sqrt(mse)), "r2": r2_score(y_true, y_pred)}
