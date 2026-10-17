"""Predict the complaint reason for negative tweets."""
import os
import joblib
import json
from src.data_loader import ROOT
from src.preprocessing import clean_tweet

_cache = {}


def predict_issue(texts):
    """Return [{'issue': str, 'confidence': float}, ...]. Meaningful for negative tweets only."""
    if not _cache:
        _cache["model"] = joblib.load(os.path.join(ROOT, "models", "issue_model.pkl"))
        _cache["tfidf"] = joblib.load(os.path.join(ROOT, "models", "issue_tfidf.pkl"))
    X = _cache["tfidf"].transform([clean_tweet(t) for t in texts])
    probs = _cache["model"].predict_proba(X)
    classes = list(_cache["model"].classes_)
    out = []
    for p in probs:
        i = int(p.argmax())
        out.append({"issue": classes[i], "confidence": round(float(p[i]), 4)})
    return out

RESULTS_DIR = "data/processed/"
rows = []
for fn in os.listdir(RESULTS_DIR):
    if fn.endswith(".json") and not fn.startswith("issue_"):
        with open(os.path.join(RESULTS_DIR, fn)) as f:
            rows.append(json.load(f))