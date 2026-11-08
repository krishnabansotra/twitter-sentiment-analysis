"""Fetch posts, predict sentiment, and save the results."""

import argparse
import csv
from pathlib import Path

from src.db import init_db, save_prediction
from src.predict import get_predictor
from src.sources.csv_source import CSVSource
from src.sources.reddit_source import RedditSource
from src.sources.youtube_source import YouTubeSource

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("csv", "reddit", "youtube"), required=True)
    parser.add_argument("--file", help="CSV file to read when --source=csv")
    parser.add_argument("--text-col", default="text", help="CSV text column")
    parser.add_argument("--query", help="Reddit search query")
    parser.add_argument("--video", help="YouTube video URL or ID")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--model", default="baseline")
    parser.add_argument("--export-csv", help="Write fetched posts to this CSV file")
    parser.add_argument(
        "--no-predict",
        action="store_true",
        help="Fetch and optionally export posts without predicting or saving them",
    )
    args = parser.parse_args()

    if args.limit < 0:
        parser.error("--limit must be non-negative")
    if args.source == "csv" and not args.file:
        parser.error("--file is required when --source=csv")
    if args.source == "reddit" and not args.query:
        parser.error("--query is required when --source=reddit")
    if args.source == "youtube" and not args.video:
        parser.error("--video is required when --source=youtube")
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
    elif args.source == "reddit":
        source = RedditSource(args.query)
    else:
        source = YouTubeSource(args.video)

    posts = source.fetch(limit=args.limit)
    if args.export_csv:
        export_path = Path(args.export_csv)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        with export_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=["comment_id", "video_id", "published_at", "text"],
            )
            writer.writeheader()
            for post in posts:
                writer.writerow(
                    {
                        "comment_id": post.meta.get("comment_id", ""),
                        "video_id": post.meta.get("video_id", ""),
                        "published_at": post.meta.get("published_at", ""),
                        "text": post.text,
                    }
                )
        print(f"Exported {len(posts)} posts to {export_path}")

    if not args.no_predict:
        init_db()
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
                posted_at=post.meta.get("published_at") or None,
            )
        print(f"Saved {len(posts)} predictions from {source.name}")


if __name__ == "__main__":
    main()
