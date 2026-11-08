"""Summaries, trends, and spike alerts for analyzed feedback."""
import argparse
import re

import pandas as pd


STOPWORDS = {
    "about", "above", "after", "again", "against", "all", "also", "am", "an",
    "and", "any", "are", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "could", "did", "do", "does",
    "doing", "down", "during", "each", "few", "for", "from", "further", "had",
    "has", "have", "having", "he", "her", "here", "hers", "herself", "him",
    "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself",
    "just", "me", "more", "most", "my", "myself", "no", "nor", "not", "of", "off",
    "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over",
    "own", "same", "she", "should", "so", "some", "such", "than", "that", "the",
    "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was", "we",
    "were", "what", "when", "where", "which", "while", "who", "whom", "why", "will",
    "with", "you", "your", "yours", "yourself", "yourselves",
}
URL_PATTERN = re.compile(
    r"(?:https?://|www\.)\S+|\b(?:[a-z0-9-]+\.)+[a-z]{2,}(?:/\S*)?",
    re.IGNORECASE,
)
MENTION_PATTERN = re.compile(r"(?<!\w)@\w+")
WORD_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?")


def sentiment_over_time(rows, freq="D") -> pd.DataFrame:
    """Count sentiment labels by posted date, falling back to saved date."""
    if isinstance(rows, pd.DataFrame):
        rows = rows.to_dict("records")

    records = []
    for row in rows:
        timestamp = pd.to_datetime(row.get("posted_at"), errors="coerce", utc=True)
        if pd.isna(timestamp):
            timestamp = pd.to_datetime(row.get("created_at"), errors="coerce", utc=True)
        sentiment = row.get("sentiment", row.get("label"))
        if not pd.isna(timestamp) and pd.notna(sentiment) and sentiment:
            records.append({"timestamp": timestamp, "sentiment": str(sentiment).lower()})

    standard_labels = ["negative", "neutral", "positive"]
    if not records:
        return pd.DataFrame(
            columns=standard_labels,
            index=pd.DatetimeIndex([], name="timestamp", tz="UTC"),
        )

    frame = pd.DataFrame(records)
    counts = frame.groupby(
        [pd.Grouper(key="timestamp", freq=freq), "sentiment"]
    ).size().unstack(fill_value=0)
    labels = standard_labels + sorted(set(counts.columns) - set(standard_labels))
    return counts.reindex(columns=labels, fill_value=0).astype(int)


def top_keywords(rows, n=20, label=None) -> list[dict]:
    """Return common non-stopword tokens, optionally for one sentiment label."""
    if n <= 0:
        return []
    if isinstance(rows, pd.DataFrame):
        rows = rows.to_dict("records")

    counts = {}
    for row in rows:
        sentiment = str(row.get("sentiment", row.get("label", ""))).lower()
        if label and sentiment != label.lower():
            continue
        text = str(row.get("text") or "")
        text = URL_PATTERN.sub(" ", text)
        text = MENTION_PATTERN.sub(" ", text)
        for word in WORD_PATTERN.findall(text.lower()):
            if len(word) >= 3 and word not in STOPWORDS:
                counts[word] = counts.get(word, 0) + 1

    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:n]
    return [{"keyword": word, "count": count} for word, count in ordered]


def _prep(df):
    """Copy analyzed rows, parse dates, and drop invalid dates."""
    prepared = df.copy()
    prepared["created_at"] = pd.to_datetime(prepared["created_at"], errors="coerce")
    return prepared.dropna(subset=["created_at"])


def kpis(df) -> dict:
    """Return headline feedback and urgency metrics."""
    prepared = _prep(df)
    total = len(prepared)
    pct_negative = (
        prepared["sentiment"].eq("negative").sum() * 100 / total if total else 0.0
    )
    urgency = pd.to_numeric(prepared["urgency"], errors="coerce")
    avg_urgency = urgency.mean() if urgency.notna().any() else 0.0
    return {
        "total": total,
        "pct_negative": round(float(pct_negative), 1),
        "avg_urgency": round(float(avg_urgency), 1),
        "critical": int(prepared["urgency_level"].eq("Critical").sum()),
        "high": int(prepared["urgency_level"].eq("High").sum()),
    }


def top_issues(df, n=8, include_unclear=False) -> pd.DataFrame:
    """Rank complaint issues among negative tweets."""
    columns = ["issue", "count", "share_pct"]
    negative = _prep(df)
    negative = negative[negative["sentiment"].eq("negative")]
    total = len(negative)
    issues = negative.dropna(subset=["issue"])
    if not include_unclear:
        issues = issues[~issues["issue"].isin(["Unclear", "Can't Tell"])]
    counts = issues["issue"].value_counts().head(n)
    return pd.DataFrame([
        {"issue": issue, "count": int(count),
         "share_pct": round(count * 100 / total, 1) if total else 0.0}
        for issue, count in counts.items()
    ], columns=columns)


def sentiment_trend(df, freq="D") -> pd.DataFrame:
    """Count sentiment by time period and calculate its negative share."""
    columns = ["created_at", "negative", "neutral", "positive", "total", "pct_negative"]
    prepared = _prep(df)
    if prepared.empty:
        return pd.DataFrame(columns=columns)
    counts = prepared.groupby([
        pd.Grouper(key="created_at", freq=freq), "sentiment"
    ]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=["negative", "neutral", "positive"], fill_value=0)
    counts["total"] = counts.sum(axis=1)
    counts["pct_negative"] = (counts["negative"] * 100 / counts["total"]).round(1)
    counts.index.name = "created_at"
    return counts.reset_index()[columns]


def detect_spikes(df, freq="D", z=2.0, min_count=5) -> pd.DataFrame:
    """Find unusually high negative-feedback counts by period and issue."""
    columns = ["period", "topic", "count", "typical", "ratio"]
    prepared = _prep(df)
    negative = prepared[prepared["sentiment"].eq("negative")]
    if negative.empty:
        return pd.DataFrame(columns=columns)

    groups = [("All negative", negative)]
    clear = negative.dropna(subset=["issue"])
    clear = clear[~clear["issue"].isin(["Unclear", "Can't Tell"])]
    groups.extend((str(issue), rows) for issue, rows in clear.groupby("issue", sort=False))

    spikes = []
    for topic, rows in groups:
        counts = rows.groupby(pd.Grouper(key="created_at", freq=freq)).size()
        if len(counts) < 4:
            continue
        for period, count in counts.items():
            others = counts.drop(index=period)
            mean = float(others.mean())
            std = float(others.std(ddof=0))
            if count >= min_count and count > mean + z * max(std, 1.0):
                spikes.append({
                    "period": period,
                    "topic": topic,
                    "count": int(count),
                    "typical": round(mean, 1),
                    "ratio": round(count / max(mean, 0.1), 1),
                })

    if not spikes:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(spikes, columns=columns).sort_values(
        "ratio", ascending=False
    ).reset_index(drop=True)


def issue_change(df, recent_days=3) -> pd.DataFrame:
    """Compare each clear issue's share of negative tweets over time."""
    columns = ["issue", "earlier_pct", "recent_pct", "change_pts"]
    negative = _prep(df)
    negative = negative[negative["sentiment"].eq("negative")]
    if negative.empty:
        return pd.DataFrame(columns=columns)

    recent_start = negative["created_at"].max().normalize() - pd.Timedelta(days=recent_days - 1)
    earlier = negative[negative["created_at"] < recent_start]
    recent = negative[negative["created_at"] >= recent_start]
    clear = negative.dropna(subset=["issue"])
    clear = clear[~clear["issue"].isin(["Unclear", "Can't Tell"])]
    issues = clear["issue"].unique()

    rows = []
    for issue in issues:
        earlier_pct = (
            earlier["issue"].eq(issue).sum() * 100 / len(earlier) if len(earlier) else 0.0
        )
        recent_pct = (
            recent["issue"].eq(issue).sum() * 100 / len(recent) if len(recent) else 0.0
        )
        rows.append({
            "issue": issue,
            "earlier_pct": round(float(earlier_pct), 1),
            "recent_pct": round(float(recent_pct), 1),
            "change_pts": round(float(recent_pct - earlier_pct), 1),
        })
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values(
        "change_pts", ascending=False
    ).reset_index(drop=True)


def urgent_queue(df, n=20) -> pd.DataFrame:
    """Return the highest-urgency analyzed tweets."""
    columns = [
        "text", "created_at", "sentiment", "issue", "urgency", "urgency_level",
        "urgency_reasons", "suggested_reply",
    ]
    prepared = _prep(df)
    available = [column for column in columns if column in prepared.columns]
    if "urgency" in available:
        prepared = prepared.sort_values("urgency", ascending=False)
    return prepared[available].head(n).reset_index(drop=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/analyzed.csv")
    args = parser.parse_args()
    analyzed = pd.read_csv(args.input)

    print("KPIs")
    print(kpis(analyzed))
    print("\nTop issues")
    print(top_issues(analyzed).to_string(index=False))
    print("\nDaily sentiment trend")
    print(sentiment_trend(analyzed).to_string(index=False))
    print("\nSpikes")
    print(detect_spikes(analyzed).to_string(index=False))
    print("\nIssue change")
    print(issue_change(analyzed).to_string(index=False))