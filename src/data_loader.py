"""Unified dataset loading. Returns train/val/test DataFrames with columns: text, label."""
import os
import pandas as pd
from sklearn.model_selection import train_test_split

LABELS = ["negative", "neutral", "positive"]       # index = class id
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SENT140_PATH = os.path.join(ROOT, "data", "raw", "sentiment140.csv")
SAMPLE_PATH = os.path.join(ROOT, "data", "raw", "Datasets", "sample_tweets.csv")


def _split(df, seed=42):
    train, tmp = train_test_split(df, test_size=0.2, random_state=seed, stratify=df["label"])
    val, test = train_test_split(tmp, test_size=0.5, random_state=seed, stratify=tmp["label"])
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def load_data(source: str = "tweet_eval", sample_size: int | None = None):
    """
    source:
      'tweet_eval'    -> Hugging Face TweetEval sentiment (3 classes) [default, recommended]
      'sentiment140'  -> data/raw/sentiment140.csv from Kaggle (2 classes)
      'sample'        -> tiny bundled CSV (for smoke tests)
      '<path>.csv'    -> your own CSV with columns: text,label
    """
    if source == "tweet_eval":
        from datasets import load_dataset
        ds = load_dataset("tweet_eval", "sentiment")
        conv = lambda s: pd.DataFrame({"text": ds[s]["text"],
                                       "label": [ID2LABEL[i] for i in ds[s]["label"]]})
        train, val, test = conv("train"), conv("validation"), conv("test")
        if sample_size:
            train = train.sample(min(sample_size, len(train)), random_state=42).reset_index(drop=True)
        return train, val, test

    if source == "sentiment140":
        df = pd.read_csv(SENT140_PATH, encoding="latin-1",
                         names=["target", "id", "date", "flag", "user", "text"])
        df["label"] = df["target"].map({0: "negative", 4: "positive"})
        df = df[["text", "label"]].dropna()
        if sample_size:
            df = df.sample(min(sample_size, len(df)), random_state=42)
        return _split(df)

    path = SAMPLE_PATH if source == "sample" else source
    df = pd.read_csv(path)[["text", "label"]].dropna()
    return _split(df)
