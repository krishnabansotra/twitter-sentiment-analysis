"""Unified inference interface for all trained models."""
import json
import os
from functools import lru_cache
from typing import List, Dict

from src.data_loader import ROOT
from src.preprocessing import clean_tweet, normalize_for_transformer

MODELS = os.path.join(ROOT, "models")


def available_backends() -> List[str]:
    b = []
    if os.path.exists(os.path.join(MODELS, "baseline_model.pkl")): b.append("baseline")
    if os.path.exists(os.path.join(MODELS, "lstm_model.pt")): b.append("lstm")
    if os.path.isdir(os.path.join(MODELS, "bert")): b.append("bert")
    return b


class SentimentPredictor:
    def __init__(self, backend: str = "baseline"):
        self.backend = backend
        if backend == "baseline":
            import joblib
            self.model = joblib.load(os.path.join(MODELS, "baseline_model.pkl"))
            self.tfidf = joblib.load(os.path.join(MODELS, "tfidf.pkl"))
        elif backend == "lstm":
            import torch
            from src.train_lstm import BiLSTM, encode
            ckpt = torch.load(os.path.join(MODELS, "lstm_model.pt"), map_location="cpu")
            with open(os.path.join(MODELS, "lstm_vocab.json")) as f:
                self.vocab = json.load(f)
            self.labels = ckpt["labels"]
            self.model = BiLSTM(len(self.vocab), n_classes=ckpt["n_classes"])
            self.model.load_state_dict(ckpt["state"]); self.model.eval()
            self._encode, self._torch = encode, torch
        elif backend == "bert":
            from transformers import pipeline
            self.pipe = pipeline("text-classification", model=os.path.join(MODELS, "bert"),
                                 top_k=None, truncation=True, max_length=128)
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def predict(self, texts: List[str]) -> List[Dict]:
        if self.backend == "baseline":
            X = self.tfidf.transform([clean_tweet(t) for t in texts])
            probs = self.model.predict_proba(X)
            classes = list(self.model.classes_)
            return [self._fmt(classes, p) for p in probs]
        if self.backend == "lstm":
            x = self._encode([clean_tweet(t) for t in texts], self.vocab)
            with self._torch.no_grad():
                probs = self._torch.softmax(self.model(x), dim=1).numpy()
            return [self._fmt(self.labels, p) for p in probs]
        out = self.pipe([normalize_for_transformer(t) for t in texts])
        return [self._fmt([d["label"] for d in r], [d["score"] for d in r]) for r in out]

    @staticmethod
    def _fmt(classes, probs):
        probs = [float(p) for p in probs]
        i = max(range(len(probs)), key=probs.__getitem__)
        return {"sentiment": classes[i], "confidence": round(probs[i], 4),
                "probabilities": {c: round(p, 4) for c, p in zip(classes, probs)}}


@lru_cache(maxsize=4)
def get_predictor(backend: str = "baseline") -> SentimentPredictor:
    return SentimentPredictor(backend)
