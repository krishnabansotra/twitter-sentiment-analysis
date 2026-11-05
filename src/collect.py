"""Fetch posts, predict sentiment, and save the results."""
import argparse
from pathlib import Path

from src.db import init_db, save_prediction
from src.predict import get_predictor
from src.sources.csv_source import CSVSource
from src.sources.reddit_source import RedditSource


ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("csv", "reddit"), required=True)
    parser.add_argument("--file", help="CSV file to read when --source=csv")
    parser.add_argument("--text-col", default="text", help="CSV text column")
    parser.add_argument("--query", help="Reddit search query")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--model", default="baseline")
    args = parser.parse_args()

    if args.limit < 0:
        parser.error("--limit must be non-negative")
    if args.source == "csv" and not args.file:
        parser.error("--file is required when --source=csv")
    if args.source == "reddit" and not args.query:
        parser.error("--query is required when --source=reddit")
    return args


def resolve_csv_path(file_path):
    path = Path(file_path)
    if not path.is_absolute():
        path = ROOT / path
    if path.is_file():
        return path

    bundled_sample = ROOT / "data" / "raw" / "Datasets" / "sample_tweets.csv"
    requested_sample = ROOT / "data" / "sample_tweets.csv"
    if path == requested_sample and bundled_sample.is_file():
        print(f"Using bundled sample CSV: {bundled_sample}")
        return bundled_sample
    raise FileNotFoundError(f"CSV file not found: {path}")


def main():
    args = parse_args()
    if args.source == "csv":
        source = CSVSource(resolve_csv_path(args.file), text_col=args.text_col)
    else:
        source = RedditSource(args.query)

    init_db()
    posts = source.fetch(limit=args.limit)
    predictions = (
        get_predictor(args.model).predict([post.text for post in posts]) if posts else []
    )
    for post, prediction in zip(posts, predictions):
        save_prediction(
            post.text,
            prediction["sentiment"],
            prediction["confidence"],
            args.model,
            source=post.source,
        )

    print(f"Saved {len(posts)} predictions from {source.name}")


if __name__ == "__main__":
    main()
