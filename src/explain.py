"""Explain sentiment predictions with local word weights from LIME."""
import argparse
import html
import re
from functools import lru_cache

import numpy as np
from lime.lime_text import LimeTextExplainer

from src.data_loader import LABELS
from src.predict import get_predictor


@lru_cache(maxsize=1)
def _get_explainer():
    return LimeTextExplainer(class_names=LABELS, random_state=42)


def predict_fn(texts, backend="baseline"):
    """Return class probabilities in the project's fixed sentiment order."""
    predictor = get_predictor(backend)
    probabilities = []
    for start in range(0, len(texts), 64):
        results = predictor.predict(list(texts[start:start + 64]))
        probabilities.extend([
            [result["probabilities"].get(label, 0.0) for label in LABELS]
            for result in results
        ])
    return np.asarray(probabilities, dtype=float).reshape((-1, len(LABELS)))


def _highlight(text, words):
    weights = {word.lower(): weight for word, weight in words}
    pieces = []
    for match in re.finditer(r"\w+|[^\w]+", text, flags=re.UNICODE):
        token = match.group()
        weight = weights.get(token.lower())
        escaped = html.escape(token)
        if weight is None or weight == 0:
            pieces.append(escaped)
            continue
        alpha = min(0.8, max(0.1, abs(weight) * 4))
        color = "46, 204, 113" if weight > 0 else "231, 76, 60"
        pieces.append(
            f'<span style="background-color: rgba({color}, {alpha:.3f})">{escaped}</span>'
        )
    return f'<span style="white-space: pre-wrap">{"".join(pieces)}</span>'


def explain_tweet(text, backend="baseline", num_features=8, num_samples=500) -> dict:
    """Explain the predicted sentiment and highlight influential words."""
    predictor = get_predictor(backend)
    prediction = predictor.predict([text])[0]
    label_index = LABELS.index(prediction["sentiment"])
    explainer = _get_explainer()
    explainer.random_state = np.random.RandomState(42)
    explanation = explainer.explain_instance(
        text,
        lambda samples: predict_fn(samples, backend),
        labels=(label_index,),
        num_features=num_features,
        num_samples=num_samples,
    )
    words = sorted(
        explanation.as_list(label=label_index), key=lambda item: abs(item[1]), reverse=True
    )
    return {
        "label": prediction["sentiment"],
        "confidence": prediction["confidence"],
        "words": words,
        "html": _highlight(text, words),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("text")
    parser.add_argument("--backend", default="baseline")
    parser.add_argument("--num-features", type=int, default=8)
    parser.add_argument("--num-samples", type=int, default=500)
    args = parser.parse_args()
    result = explain_tweet(
        args.text, args.backend, args.num_features, args.num_samples
    )
    print(f"{result['label']} ({result['confidence']:.3f})")
    print(result["html"])