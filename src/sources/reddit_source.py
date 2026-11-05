"""Reddit-backed post source."""
import os

from src.sources.base import Post, Source


class RedditSource(Source):
    """Search Reddit for text posts."""

    name = "reddit"

    def __init__(self, query, subreddit="all"):
        client_id = os.environ.get("REDDIT_CLIENT_ID")
        client_secret = os.environ.get("REDDIT_CLIENT_SECRET")
        missing = [
            name for name, value in (
                ("REDDIT_CLIENT_ID", client_id),
                ("REDDIT_CLIENT_SECRET", client_secret),
            )
            if not value
        ]
        if missing:
            raise ValueError(f"Missing Reddit environment variables: {', '.join(missing)}")

        try:
            import praw
        except ImportError as error:
            raise ImportError("Install praw to use the Reddit source.") from error

        self.query = query
        self.subreddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent="twitter-sentiment-analysis/1.0",
        ).subreddit(subreddit)

    def fetch(self, limit=100) -> list[Post]:
        """Return up to limit matching posts without author information."""
        if limit < 0:
            raise ValueError("limit must be non-negative")

        posts = []
        for submission in self.subreddit.search(self.query, limit=limit):
            text = f"{submission.title} {submission.selftext}".strip()[:1000]
            posts.append(Post(text=text, source=self.name))
        return posts
