import os

import pytest

from src.data_loader import ROOT
from src.explain import explain_tweet


MODEL_PATH = os.path.join(ROOT, "models", "baseline_model.pkl")


@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="baseline model is not trained")
def test_explain_tweet_highlights_words():
    result = explain_tweet(
        "The flight was delayed and the staff were rude",
        backend="baseline",
        num_samples=200,
    )
    assert result["words"]
    assert isinstance(result["html"], str)
    assert "<span" in result["html"]