from src.urgency import compute_urgency


def test_angry_serious_is_critical():
    r = compute_urgency(0.95, "STRANDED for hours, I want a refund NOW!!", "Cancelled Flight")
    assert r["level"] == "Critical"
    assert any("refund" in x for x in r["reasons"])


def test_positive_is_low():
    assert compute_urgency(0.03, "Great flight, thank you!", None)["level"] == "Low"


def test_score_range():
    assert 0 <= compute_urgency(0.5, "ok", "Late Flight")["score"] <= 100