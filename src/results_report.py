"""Report test-set results and bootstrap uncertainty for trained models."""
import itertools
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from src.data_loader import ROOT, load_data
from src.predict import available_backends, get_predictor
from src.preprocessing import clean_tweet


LABELS = ["negative", "neutral", "positive"]
BOOTSTRAPS = 1000


def _bootstrap_indices(size, seed=42):
    return np.random.default_rng(seed).integers(
        0, size, size=(BOOTSTRAPS, size), dtype=np.int32
    )


def _bootstrap_metrics(actual, predicted, labels, indices):
    label_ids = {label: index for index, label in enumerate(labels)}
    actual_ids = np.asarray([label_ids[label] for label in actual], dtype=np.int32)
    predicted_ids = np.asarray([label_ids[label] for label in predicted], dtype=np.int32)
    class_count = len(labels)

    def class_f1(confusion):
        true_positive = np.diagonal(confusion, axis1=-2, axis2=-1)
        denominator = confusion.sum(axis=-1) + confusion.sum(axis=-2)
        return np.divide(
            2 * true_positive, denominator,
            out=np.zeros_like(true_positive, dtype=float), where=denominator != 0,
        )

    confusion = np.bincount(
        actual_ids * class_count + predicted_ids,
        minlength=class_count * class_count,
    ).reshape(class_count, class_count)
    per_class_f1 = class_f1(confusion)
    accuracy = float(np.mean(actual_ids == predicted_ids))
    macro_f1 = float(per_class_f1.mean())

    sample_actual = actual_ids[indices]
    sample_predicted = predicted_ids[indices]
    bootstrap_count = len(indices)
    offsets = np.arange(bootstrap_count, dtype=np.int64)[:, None] * class_count**2
    encoded = offsets + sample_actual * class_count + sample_predicted
    boot_confusion = np.bincount(
        encoded.ravel(), minlength=bootstrap_count * class_count**2
    ).reshape(bootstrap_count, class_count, class_count)
    boot_accuracy = np.mean(sample_actual == sample_predicted, axis=1)
    boot_f1 = class_f1(boot_confusion).mean(axis=1)
    accuracy_ci = np.quantile(boot_accuracy, [0.025, 0.975])
    f1_ci = np.quantile(boot_f1, [0.025, 0.975])
    return {
        "accuracy": float(accuracy),
        "accuracy_ci_low": float(accuracy_ci[0]),
        "accuracy_ci_high": float(accuracy_ci[1]),
        "macro_f1": float(macro_f1),
        "macro_f1_ci_low": float(f1_ci[0]),
        "macro_f1_ci_high": float(f1_ci[1]),
        "per_class_f1": dict(zip(labels, map(float, per_class_f1))),
        "bootstrap_f1": boot_f1,
    }


def _predict_sentiment(backend, texts):
    predictor = get_predictor(backend)
    results = []
    for start in range(0, len(texts), 64):
        results.extend(predictor.predict(texts[start:start + 64]))
    probability_rows = np.asarray([
        [result["probabilities"].get(label, 0.0) for label in LABELS]
        for result in results
    ])
    predicted = np.asarray(LABELS)[probability_rows.argmax(axis=1)]
    return predicted


def _markdown_table(headers, rows):
    if not rows:
        return "_No results available._"
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def _format_ci(low, high):
    return f"[{low:.3f}, {high:.3f}]"


def main():
    _, _, test = load_data(os.path.join(ROOT, "data", "processed", "airline.csv"))
    actual = test["label"].to_numpy()
    texts = test["text"].astype(str).tolist()
    sentiment_indices = _bootstrap_indices(len(test))
    sentiment_results = {}
    output_rows = []

    for backend in [name for name in ["baseline", "lstm", "bert"] if name in available_backends()]:
        try:
            predicted = _predict_sentiment(backend, texts)
            metrics = _bootstrap_metrics(actual, predicted, LABELS, sentiment_indices)
        except Exception as error:
            print(f"Skipping {backend}: {error}")
            continue
        sentiment_results[backend] = metrics
        row = {
            "section": "sentiment", "model": backend, "other_model": "",
            **{key: value for key, value in metrics.items() if key != "bootstrap_f1"},
        }
        for label in LABELS:
            row[f"f1_{label}"] = metrics["per_class_f1"][label]
        output_rows.append(row)

    sentiment_table = []
    for backend, metrics in sentiment_results.items():
        sentiment_table.append([
            backend,
            f"{metrics['accuracy']:.3f}",
            _format_ci(metrics["accuracy_ci_low"], metrics["accuracy_ci_high"]),
            f"{metrics['macro_f1']:.3f}",
            _format_ci(metrics["macro_f1_ci_low"], metrics["macro_f1_ci_high"]),
            *(f"{metrics['per_class_f1'][label]:.3f}" for label in LABELS),
        ])

    pairwise_results = []
    for first, second in itertools.combinations(sentiment_results, 2):
        share = float(np.mean(
            sentiment_results[first]["bootstrap_f1"] > sentiment_results[second]["bootstrap_f1"]
        ))
        pairwise_results.append([first, second, share])
        output_rows.append({
            "section": "pairwise", "model": first, "other_model": second,
            "share_first_higher": share,
        })

    issue_result = None
    model_path = os.path.join(ROOT, "models", "issue_model.pkl")
    tfidf_path = os.path.join(ROOT, "models", "issue_tfidf.pkl")
    try:
        issues = pd.read_csv(os.path.join(ROOT, "data", "processed", "issues.csv")).dropna()
        _, issue_test = train_test_split(
            issues, test_size=0.2, random_state=42, stratify=issues["label"]
        )
        model = joblib.load(model_path)
        tfidf = joblib.load(tfidf_path)
        features = tfidf.transform(issue_test["text"].apply(clean_tweet))
        issue_predicted = model.predict(features)
        issue_actual = issue_test["label"].to_numpy()
        issue_labels = sorted(set(issue_actual) | set(issue_predicted))
        issue_result = _bootstrap_metrics(
            issue_actual, issue_predicted, issue_labels,
            _bootstrap_indices(len(issue_test)),
        )
        output_rows.append({
            "section": "issue_classifier", "model": "issue",
            **{key: value for key, value in issue_result.items()
               if key not in {"bootstrap_f1", "per_class_f1"}},
        })
    except (FileNotFoundError, OSError) as error:
        print(f"Skipping issue classifier: {error}")

    pair_table = [[first, second, f"{share:.1%}"]
                  for first, second, share in pairwise_results]
    issue_table = []
    if issue_result:
        issue_table.append([
            f"{issue_result['accuracy']:.3f}",
            _format_ci(issue_result["accuracy_ci_low"], issue_result["accuracy_ci_high"]),
            f"{issue_result['macro_f1']:.3f}",
            _format_ci(issue_result["macro_f1_ci_low"], issue_result["macro_f1_ci_high"]),
        ])

    report = "\n".join([
        "## Sentiment models",
        _markdown_table(
            ["Model", "Accuracy", "Accuracy 95% CI", "Macro-F1", "Macro-F1 95% CI",
             "F1 negative", "F1 neutral", "F1 positive"], sentiment_table
        ),
        "\n## Pairwise bootstrap comparison",
        _markdown_table(
            ["First model", "Second model", "First higher"], pair_table
        ),
        "\n## Issue classifier",
        _markdown_table(
            ["Accuracy", "Accuracy 95% CI", "Macro-F1", "Macro-F1 95% CI"], issue_table
        ),
    ])

    output_dir = os.path.join(ROOT, "results")
    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame(output_rows).to_csv(
        os.path.join(output_dir, "final_results.csv"), index=False
    )
    with open(os.path.join(output_dir, "final_results.md"), "w", encoding="utf-8") as file:
        file.write(report + "\n")
    print(report)


if __name__ == "__main__":
    main()