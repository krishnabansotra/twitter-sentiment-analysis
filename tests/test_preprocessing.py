from src.preprocessing import clean_tweet, normalize_for_transformer


def test_removes_urls_mentions():
    out = clean_tweet("Check this https://t.co/abc @elon #AI is great")
    assert "http" not in out and "@" not in out and "elon" not in out
    assert "ai" in out.split()


def test_keeps_negation():
    assert "not" in clean_tweet("This is not good").split()


def test_elongation():
    assert "soooooo" not in clean_tweet("soooooo happy")


def test_transformer_norm():
    assert normalize_for_transformer("hi @bob http://x.com") == "hi @user http"

def test_artifact_fix():
    assert "cancelled" in clean_tweet("my flight got Cancelled Flightled").split()
    assert "later" in clean_tweet("see you Late Flightr").split()