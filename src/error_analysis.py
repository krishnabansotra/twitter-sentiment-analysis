"""Summarize sentiment-model errors and save examples for review."""
import argparse
from collections import Counter
import os
import re

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from src.data_loader import ROOT, load_data
from src.predict import get_predictor


LABELS = ["negative", "neutral", "positive"]
NEGATION_RE = re.compile(r"\b(?:not|no|nor|never|n't)\b", re.IGNORECASE)
EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]")


def _predict(texts, backend):
    predictor = get_predictor(backend)
    rows = []
    for start in range(0, len(texts), 64):
        rows.extend(predictor.predict(texts[start:start + 64]))
    probabilities = np.asarray([
        [row["probabilities"].get(label, 0.0) for label in LABELS]
        for row in rows
    ])
    predictions = np.asarray(LABELS)[probabilities.argmax(axis=1)]
    return predictions, probabilities.max(axis=1)


def _accuracy_by_bucket(actual, predicted, buckets, names):
    for name in names:
        selected = buckets == name
        if selected.any():
            score = accuracy_score(actual[selected], predicted[selected])
            print(f"  {name}: {score:.3f} (n={int(selected.sum())})")


def main(backend="bert", n=60, out_dir="results/error_analysis"):
    _, _, test = load_data(os.path.join(ROOT, "data", "processed", "airline.csv"))
    texts = test["text"].astype(str).tolist()
    actual = test["label"].to_numpy()
    predicted, confidence = _predict(texts, backend)
    correct = actual == predicted

    print(f"Accuracy: {accuracy_score(actual, predicted):.3f}")
    print("Most common errors (true -> predicted):")
    for (true_label, predicted_label), count in Counter(
        (true, pred) for true, pred, is_correct in zip(actual, predicted, correct)
        if not is_correct
    ).most_common(6):
        print(f"  {true_label} -> {predicted_label}: {count}")

    word_counts = np.asarray([len(text.split()) for text in texts])
    length_buckets = np.select(
        [word_counts <= 6, (word_counts >= 7) & (word_counts <= 15), word_counts >= 16],
        ["<=6 words", "7-15 words", "16+ words"],
        default="16+ words",
    )
    print("Accuracy by tweet length:")
    _accuracy_by_bucket(actual, predicted, length_buckets,
                        ["<=6 words", "7-15 words", "16+ words"])

    confidence_buckets = pd.cut(
        confidence, bins=[-np.inf, 0.5, 0.7, 0.9, np.inf], right=False,
        labels=["<0.5", "0.5-0.7", "0.7-0.9", ">=0.9"],
    ).astype(str)
    print("Accuracy by confidence:")
    _accuracy_by_bucket(actual, predicted, confidence_buckets,
                        ["<0.5", "0.5-0.7", "0.7-0.9", ">=0.9"])

    errors = test.loc[~correct, ["text"]].copy()
    errors["true"] = actual[~correct]
    errors["pred"] = predicted[~correct]
    errors["confidence"] = confidence[~correct]
    errors["has_negation"] = errors["text"].astype(str).map(
        lambda text: bool(NEGATION_RE.search(text))
    )
    errors["has_question"] = errors["text"].astype(str).str.contains("?", regex=False)
    errors["is_short"] = errors["text"].astype(str).map(lambda text: len(text.split()) <= 6)
    errors["has_emoji"] = errors["text"].astype(str).map(
        lambda text: bool(EMOJI_RE.search(text))
    )
    errors["error_type"] = ""
    columns = [
        "text", "true", "pred", "confidence", "has_negation", "has_question",
        "is_short", "has_emoji", "error_type",
    ]
    errors = errors[columns]
    if len(errors) > n:
        errors = errors.sample(n=n, random_state=42)

    output_dir = out_dir if os.path.isabs(out_dir) else os.path.join(ROOT, out_dir)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{backend}_errors.csv")
    errors.to_csv(output_path, index=False)
    print(f"Saved {len(errors)} errors to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="bert")
    parser.add_argument("--n", type=int, default=60)
    parser.add_argument("--out_dir", default="results/error_analysis")
    args = parser.parse_args()
    main(args.backend, args.n, args.out_dir)