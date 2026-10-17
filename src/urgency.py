"""Rule-based urgency score (0-100) with human-readable reasons."""
import re

URGENT_KEYWORDS = [
    "refund", "lawyer", "sue", "lawsuit", "unsafe", "stranded", "stuck", "unacceptable",
    "compensation", "scam", "emergency", "hospital", "furious", "disgusting", "worst",
    "never again", "missed", "lost", "cancelled", "canceled", "rude", "complaint",
]

# How serious each complaint type is (0-1). Judgment-based, adjust if you disagree.
ISSUE_WEIGHT = {
    "Cancelled Flight": 0.9, "Lost Luggage": 0.8, "Damaged Luggage": 0.7,
    "Flight Booking Problems": 0.7, "Late Flight": 0.6, "Customer Service Issue": 0.6,
    "Flight Attendant Complaints": 0.5, "Bad Flight": 0.5, "longlines": 0.4,
    "Can't Tell": 0.3, "Unclear": 0.3,
}


def _level(score: int) -> str:
    if score >= 75: return "Critical"
    if score >= 55: return "High"
    if score >= 35: return "Medium"
    return "Low"


def compute_urgency(p_negative: float, text: str, issue: str | None = None) -> dict:
    """urgency = 100 * (0.5*negativity + 0.3*keywords/intensity + 0.2*issue severity)"""
    low = str(text).lower()
    hits = [k for k in URGENT_KEYWORDS if re.search(rf"\b{re.escape(k)}\b", low)]
    kw = min(1.0, len(hits) / 2)

    letters = [c for c in str(text) if c.isalpha()]
    shouting = len(letters) >= 10 and sum(c.isupper() for c in letters) / len(letters) > 0.5
    if shouting: kw = min(1.0, kw + 0.1)
    if str(text).count("!") >= 2: kw = min(1.0, kw + 0.1)

    iw = ISSUE_WEIGHT.get(issue, 0.0) if issue else 0.0
    score = round(100 * (0.5 * p_negative + 0.3 * kw + 0.2 * iw))

    reasons = []
    if p_negative >= 0.7: reasons.append("strongly negative")
    if hits: reasons.append("keywords: " + ", ".join(hits[:4]))
    if shouting: reasons.append("written in capitals")
    if issue and iw >= 0.7: reasons.append(f"serious issue: {issue}")
    return {"score": score, "level": _level(score), "reasons": reasons}