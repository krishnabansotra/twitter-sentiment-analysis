"""
Fetch live tweets from X API v2 (needs a bearer token, see .env.example).
NOTE: X API access/pricing changes often, so check developer.x.com for current limits.
If you have no API access, use datasets or upload your own CSV instead.
"""
import os
import pandas as pd


def fetch_tweets(query: str, max_results: int = 50, bearer_token: str | None = None) -> pd.DataFrame:
    import tweepy
    token = bearer_token or os.getenv("X_BEARER_TOKEN")
    if not token:
        raise RuntimeError("X_BEARER_TOKEN not set. Add it to your environment or .env file.")
    client = tweepy.Client(bearer_token=token, wait_on_rate_limit=True)
    q = f"{query} -is:retweet lang:en"
    resp = client.search_recent_tweets(query=q, max_results=min(max(max_results, 10), 100),
                                       tweet_fields=["created_at", "public_metrics"])
    rows = [{"created_at": t.created_at, "text": t.text,
             "likes": t.public_metrics["like_count"], "retweets": t.public_metrics["retweet_count"]}
            for t in (resp.data or [])]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = fetch_tweets("python", 20)
    print(df.head())
    df.to_csv("data/raw/live_tweets.csv", index=False)
