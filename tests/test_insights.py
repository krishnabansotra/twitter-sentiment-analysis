import pandas as pd

from src.insights import detect_spikes, top_issues


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