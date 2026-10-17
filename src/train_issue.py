"""Issue classifier: predicts WHY a customer is unhappy (trained on negative tweets only)."""
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.data_loader import ROOT
from src.preprocessing import clean_tweet
from src.evaluate import evaluate


def main():
    df = pd.read_csv(os.path.join(ROOT, "data", "processed", "issues.csv")).dropna()
    df["clean"] = df["text"].apply(clean_tweet)
    train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
    print(f"Train={len(train)} Test={len(test)}")

    tfidf = TfidfVectorizer(max_features=30000, ngram_range=(1, 2), min_df=2)
    Xtr = tfidf.fit_transform(train["clean"])
    Xte = tfidf.transform(test["clean"])

    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    model.fit(Xtr, train["label"])
    evaluate(test["label"], model.predict(Xte), "issue_logreg")

    out = os.path.join(ROOT, "models")
    joblib.dump(model, os.path.join(out, "issue_model.pkl"))
    joblib.dump(tfidf, os.path.join(out, "issue_tfidf.pkl"))
    print("Saved issue model to models/")


if __name__ == "__main__":
    main()