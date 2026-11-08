from src import batch


def test_analyze_batch_skips_empty_text_and_reuses_predictor(monkeypatch):
    class FakePredictor:
        def __init__(self):
            self.calls = []

        def predict(self, texts):
            self.calls.append(texts)
            return [
                {"sentiment": "positive", "confidence": 0.9},
                {"sentiment": "negative", "confidence": 0.8},
            ]

    predictor = FakePredictor()
    get_predictor_calls = []

    def fake_get_predictor(model_name):
        get_predictor_calls.append(model_name)
        return predictor

    monkeypatch.setattr(batch, "get_predictor", fake_get_predictor)
    progress = []

    results = batch.analyze_batch(
        ["I love this", "  ", "I hate this"], "baseline", progress.append
    )

    assert get_predictor_calls == ["baseline"]
    assert predictor.calls == [["I love this", "I hate this"]]
    assert results == [
        {"text": "I love this", "sentiment": "positive", "confidence": 0.9},
        {"text": "I hate this", "sentiment": "negative", "confidence": 0.8},
    ]
    assert progress == [1.0]


def test_analyze_batch_with_only_empty_text_does_not_load_model(monkeypatch):
    def unexpected_model_load(_model_name):
        raise AssertionError("No model is needed for empty input")

    monkeypatch.setattr(batch, "get_predictor", unexpected_model_load)
    assert batch.analyze_batch(["", "   "], "baseline") == []