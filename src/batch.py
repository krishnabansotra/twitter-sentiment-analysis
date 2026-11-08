"""Batch sentiment prediction helpers."""
from collections.abc import Callable

from src.predict import get_predictor


def analyze_batch(
    texts: list[str],
    model_name: str,
    progress_callback: Callable[[float], None] | None = None,
) -> list[dict]:
    """Predict sentiment for each non-empty text using one predictor."""
    valid_texts = []
    for text in texts:
        if text is None:
            continue
        text = str(text)
        if text.strip():
            valid_texts.append(text)

    if not valid_texts:
        if progress_callback:
            progress_callback(1.0)
        return []

    predictor = get_predictor(model_name)
    results = []
    chunk_size = 64
    for start in range(0, len(valid_texts), chunk_size):
        chunk = valid_texts[start:start + chunk_size]
        predictions = predictor.predict(chunk)
        results.extend(
            {
                "text": text,
                "sentiment": prediction["sentiment"],
                "confidence": prediction["confidence"],
            }
            for text, prediction in zip(chunk, predictions)
        )
        if progress_callback:
            progress_callback(min(start + len(chunk), len(valid_texts)) / len(valid_texts))

    return results