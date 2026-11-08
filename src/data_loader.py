"""Unified dataset loading. Returns train/val/test DataFrames with columns: text, label."""

import os

import pandas as pd
from sklearn.model_selection import train_test_split

LABELS = ["negative", "neutral", "positive"]  # index = class id
LABEL2ID = {label: index for index, label in enumerate(LABELS)}
ID2LABEL = {index: label for label, index in LABEL2ID.items()}

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SENT140_PATH = os.path.join(ROOT, "data", "raw", "sentiment140.csv")
SAMPLE_PATH = os.path.join(ROOT, "data", "raw", "Datasets", "sample_tweets.csv")


def _split(df, seed=42):
    train, tmp = train_test_split(df, test_size=0.2, random_state=seed, stratify=df["label"])
    val, test = train_test_split(tmp, test_size=0.5, random_state=seed, stratify=tmp["label"])
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def load_data(source: str = "tweet_eval", sample_size: int | None = None):
    """Load TweetEval, Sentiment140, sample, or local CSV data into train/val/test splits."""
    if source == "tweet_eval":
        from datasets import load_dataset

        ds = load_dataset("tweet_eval", "sentiment")

        def convert_split(split):
            return pd.DataFrame(
                {
                    "text": ds[split]["text"],
                    "label": [ID2LABEL[label_id] for label_id in ds[split]["label"]],
                }
            )

        train, val, test = (
            convert_split("train"),
            convert_split("validation"),
            convert_split("test"),
        )
        if sample_size:
            train = train.sample(min(sample_size, len(train)), random_state=42).reset_index(
                drop=True
            )
        return train, val, test

    if source == "sentiment140":
        df = pd.read_csv(
            SENT140_PATH, encoding="latin-1", names=["target", "id", "date", "flag", "user", "text"]
        )
        df["label"] = df["target"].map({0: "negative", 4: "positive"})
        df = df[["text", "label"]].dropna()
        if sample_size:
            df = df.sample(min(sample_size, len(df)), random_state=42)
        return _split(df)

    path = SAMPLE_PATH if source == "sample" else source
    df = pd.read_csv(path)[["text", "label"]].dropna()
    if sample_size:
        df = df.sample(min(sample_size, len(df)), random_state=42)
    return _split(df)
