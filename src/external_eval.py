"""Evaluate a saved sentiment model on a different labeled CSV."""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
)
from sklearn.metrics import ConfusionMatrixDisplay

from src.data_loader import ROOT
from src.predict import get_predictor


LABELS = ["negative", "neutral", "positive"]


def _map_labels(values, mode):
    if mode == "text":
        return values.astype(str).str.strip().str.lower()
    ratings = pd.to_numeric(values, errors="coerce")
    mapped = pd.Series(index=values.index, dtype="object")
    mapped.loc[ratings.between(1, 2)] = "negative"
    mapped.loc[ratings.eq(3)] = "neutral"
    mapped.loc[ratings.between(4, 5)] = "positive"
    return mapped


def main(input_path, text_col, label_col, label_mode="text", backend="baseline",
         binary=False, n=3000, out_dir="results/external"):
    data = pd.read_csv(input_path)
    missing = [column for column in [text_col, label_col] if column not in data.columns]
    if missing:
        raise ValueError(f"Missing CSV column(s): {', '.join(missing)}")
    data = data.dropna(subset=[text_col, label_col]).copy()
    data["_label"] = _map_labels(data[label_col], label_mode)
    data = data[data["_label"].isin(LABELS)]
    if binary:
        data = data[data["_label"].isin(["negative", "positive"])]
    if data.empty:
        raise ValueError("No rows have valid labels for the selected evaluation mode.")
    if len(data) > n:
        data = data.sample(n=n, random_state=42)

    predictor = get_predictor(backend)
    results = []
    texts = data[text_col].astype(str).tolist()
    for start in range(0, len(texts), 64):
        results.extend(predictor.predict(texts[start:start + 64]))
    probabilities = np.asarray([
        [result["probabilities"].get(label, 0.0) for label in LABELS]
        for result in results
    ])
    actual = data["_label"].to_numpy()
    if binary:
        binary_scores = probabilities[:, [LABELS.index("negative"), LABELS.index("positive")]]
        predicted = np.asarray(["negative", "positive"])[binary_scores.argmax(axis=1)]
        display_labels = ["negative", "positive"]
    else:
        predicted = np.asarray(LABELS)[probabilities.argmax(axis=1)]
        display_labels = LABELS

    accuracy = float(accuracy_score(actual, predicted))
    macro_f1 = float(f1_score(
        actual, predicted, labels=display_labels, average="macro", zero_division=0
    ))
    report = classification_report(
        actual, predicted, labels=display_labels, zero_division=0, output_dict=True
    )
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    print(classification_report(
        actual, predicted, labels=display_labels, zero_division=0
    ))

    output_dir = out_dir if os.path.isabs(out_dir) else os.path.join(ROOT, out_dir)
    os.makedirs(output_dir, exist_ok=True)
    suffix = "_binary" if binary else ""
    confusion = confusion_matrix(actual, predicted, labels=display_labels)
    display = ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=display_labels)
    display.plot(cmap="Blues", values_format="d")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{backend}{suffix}_confusion_matrix.png"), dpi=150)
    plt.close()

    metrics = {
        "input": input_path,
        "backend": backend,
        "binary": binary,
        "n": len(data),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "labels": display_labels,
        "classification_report": report,
        "confusion_matrix": confusion.tolist(),
    }
    with open(os.path.join(output_dir, f"{backend}{suffix}_metrics.json"), "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    print(f"Domain-shift test: model trained on airline tweets, evaluated on {input_path} without retraining.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--text-col", required=True)
    parser.add_argument("--label-col", required=True)
    parser.add_argument("--label-mode", choices=["text", "stars"], default="text")
    parser.add_argument("--backend", default="baseline")
    parser.add_argument("--binary", action="store_true")
    parser.add_argument("--n", type=int, default=3000)
    parser.add_argument("--out_dir", default="results/external")
    args = parser.parse_args()
    if args.n < 1:
        parser.error("--n must be at least 1")
    main(
        args.input, args.text_col, args.label_col, args.label_mode,
        args.backend, args.binary, args.n, args.out_dir,
    )