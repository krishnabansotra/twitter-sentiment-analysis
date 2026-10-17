"""Summaries, trends, and spike alerts for analyzed feedback."""
import argparse

import pandas as pd


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