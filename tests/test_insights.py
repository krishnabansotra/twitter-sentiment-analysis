import pandas as pd

from src.insights import (
    detect_spikes, sentiment_over_time, top_issues, top_keywords,
)


def _sample_data():
    rows = []
    for day in pd.date_range("2026-01-01", periods=8, freq="D"):
        count = 40 if day.day == 5 else 8
        for index in range(count):
            rows.append({
                "text": f"Delayed flight feedback {index}",
                "created_at": day,
                "sentiment": "negative",
                "issue": "Late Flight",
                "urgency": 50,
                "urgency_level": "Medium",
                "urgency_reasons": "",
                "suggested_reply": "",
            })
    return pd.DataFrame(rows)


def test_detect_spikes_finds_day_five():
    spikes = detect_spikes(_sample_data())
    assert {period.day for period in spikes["period"]} == {5}
    assert (spikes["topic"] == "All negative").any()


def test_top_issues_ranks_late_flight_first():
    assert top_issues(_sample_data()).iloc[0]["issue"] == "Late Flight"


def test_sentiment_over_time_prefers_posted_at_and_falls_back_to_created_at():
    rows = [
        {
            "posted_at": "2024-01-02T10:00:00Z",
            "created_at": "2024-01-01T10:00:00Z",
            "label": "positive",
        },
        {"posted_at": None, "created_at": "2024-01-02T14:00:00Z", "label": "negative"},
        {"posted_at": "2024-01-03T08:00:00Z", "label": "positive"},
    ]

    counts = sentiment_over_time(rows)

    assert counts.loc["2024-01-02", "positive"] == 1
    assert counts.loc["2024-01-02", "negative"] == 1
    assert counts.loc["2024-01-03", "positive"] == 1


def test_top_keywords_filters_stopwords_urls_mentions_numbers_and_label():
    rows = [
        {
            "text": (
                "The great service, great flight! https://example.com "
                "example.net/help @traveler 123"
            ),
            "label": "positive",
        },
        {"text": "Terrible service delay", "label": "negative"},
    ]

    assert top_keywords(rows, n=3) == [
        {"keyword": "great", "count": 2},
        {"keyword": "service", "count": 2},
        {"keyword": "delay", "count": 1},
    ]
    assert top_keywords(rows, n=5, label="positive") == [
        {"keyword": "great", "count": 2},
        {"keyword": "flight", "count": 1},
        {"keyword": "service", "count": 1},
    ]