"""Level 1: TF-IDF + classical ML models."""
import argparse
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score

from src.data_loader import load_data, ROOT
from src.preprocessing import clean_tweet
from src.evaluate import evaluate


def main(source, sample_size):
    train, val, test = load_data(source, sample_size)
    print(f"Train={len(train)} Val={len(val)} Test={len(test)}")
    for d in (train, val, test):
        d["clean"] = d["text"].apply(clean_tweet)

    tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=1)
    Xtr = tfidf.fit_transform(train["clean"])
    Xva, Xte = tfidf.transform(val["clean"]), tfidf.transform(test["clean"])

    candidates = {
        "logreg": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "naive_bayes": MultinomialNB(),
        "linear_svm": LinearSVC(class_weight="balanced"),
    }
    best_name, best_f1, best_model = None, -1, None
    for name, model in candidates.items():
        model.fit(Xtr, train["label"])
        f1 = f1_score(val["label"], model.predict(Xva), average="macro")
        print(f"{name}: val macro-F1 = {f1:.4f}")
        evaluate(test["label"], model.predict(Xte), f"baseline_{name}")
        if f1 > best_f1:
            best_name, best_f1, best_model = name, f1, model

    # Only LogisticRegression/NB expose predict_proba; keep LR as the served model for confidence scores.
    served = candidates["logreg"] if best_name == "linear_svm" else best_model
    out = os.path.join(ROOT, "models")
    joblib.dump(served, os.path.join(out, "baseline_model.pkl"))
    joblib.dump(tfidf, os.path.join(out, "tfidf.pkl"))
    print(f"Saved served model ({type(served).__name__}) to models/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="tweet_eval", help="tweet_eval | sentiment140 | sample | path.csv")
    ap.add_argument("--sample_size", type=int, default=None)
    a = ap.parse_args()
    main(a.source, a.sample_size)
