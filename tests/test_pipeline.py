"""Smoke test using the bundled sample CSV (no internet needed)."""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from src.data_loader import load_data
from src.preprocessing import clean_tweet


def test_baseline_trains():
    train, val, test = load_data("sample")
    tf = TfidfVectorizer()
    X = tf.fit_transform(train["text"].apply(clean_tweet))
    clf = LogisticRegression(max_iter=500).fit(X, train["label"])
    assert len(clf.predict(tf.transform(test["text"].apply(clean_tweet)))) == len(test)
