"""Compare confidence cutoffs for the issue classifier."""
import os

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_loader import ROOT
from src.preprocessing import clean_tweet


def main():
    data_path = os.path.join(ROOT, "data", "processed", "issues.csv")
    model_path = os.path.join(ROOT, "models", "issue_model.pkl")
    tfidf_path = os.path.join(ROOT, "models", "issue_tfidf.pkl")

    data = pd.read_csv(data_path).dropna()
    _, test = train_test_split(
        data, test_size=0.2, random_state=42, stratify=data["label"]
    )
    model = joblib.load(model_path)
    tfidf = joblib.load(tfidf_path)

    features = tfidf.transform(test["text"].apply(clean_tweet))
    probabilities = model.predict_proba(features)
    confidences = probabilities.max(axis=1)
    predictions = model.classes_[probabilities.argmax(axis=1)]
    labels = test["label"].to_numpy()

    rows = []
    for cutoff in [0.2, 0.25, 0.3, 0.35, 0.4, 0.5]:
        selected = confidences >= cutoff
        share = selected.mean() * 100
        accuracy = (
            (predictions[selected] == labels[selected]).mean() * 100
            if selected.any() else float("nan")
        )
        rows.append({
            "cutoff": f"{cutoff:.2f}",
            "share (%)": f"{share:.1f}%",
            "accuracy (%)": f"{accuracy:.1f}%" if selected.any() else "N/A",
        })

    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()