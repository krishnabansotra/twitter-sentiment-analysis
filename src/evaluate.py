"""Metrics, confusion matrices, and model comparison."""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, classification_report)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "models", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def evaluate(y_true, y_pred, name: str, labels=None):
    """Print + save metrics and confusion matrix for one model."""
    labels = labels or sorted(set(y_true) | set(y_pred))
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    print(f"\n=== {name} ===")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(max(5, len(labels) * 0.9), max(4, len(labels) * 0.7)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"cm_{name}.png"), dpi=150)
    plt.close()

    res = {"model": name, "accuracy": round(acc, 4), "precision": round(p, 4),
           "recall": round(r, 4), "f1_macro": round(f1, 4)}
    with open(os.path.join(RESULTS_DIR, f"{name}.json"), "w") as f:
        json.dump(res, f, indent=2)
    return res


def compare_models() -> pd.DataFrame:
    """Combine all saved result JSONs into a comparison table."""
    rows = []
    for fn in os.listdir(RESULTS_DIR):
        if fn.endswith(".json") and not fn.startswith("issue_"):
            with open(os.path.join(RESULTS_DIR, fn)) as f:
                rows.append(json.load(f))
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("f1_macro", ascending=False).reset_index(drop=True)
        df.to_csv(os.path.join(RESULTS_DIR, "comparison.csv"), index=False)
    return df


if __name__ == "__main__":
    print(compare_models().to_string(index=False))
