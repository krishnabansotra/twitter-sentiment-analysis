"""Full pipeline: sentiment -> issue (negatives only) -> urgency -> suggested reply."""
import argparse
import pandas as pd

from src.predict import get_predictor
from src.issue import predict_issue
from src.urgency import compute_urgency
from src.replies import suggest_reply

UNCLEAR_BELOW = 0.35   # issue confidence below this is shown as "Unclear"
REVIEW_BELOW = 0.5  # sentiment confidence below this needs a human check


def analyze_texts(texts, backend="baseline", chunk=128):
    texts = [str(t) for t in texts]
    predictor = get_predictor(backend)
    sent = []
    for i in range(0, len(texts), chunk):
        sent += predictor.predict(texts[i:i + chunk])

    neg_idx = [i for i, s in enumerate(sent) if s["sentiment"] == "negative"]
    issues = [None] * len(texts)
    if neg_idx:
        for i, r in zip(neg_idx, predict_issue([texts[i] for i in neg_idx])):
            issues[i] = r if r["confidence"] >= UNCLEAR_BELOW else {"issue": "Unclear", "confidence": r["confidence"]}

    rows = []
    for t, s, iss in zip(texts, sent, issues):
        issue = iss["issue"] if iss else None
        u = compute_urgency(s["probabilities"].get("negative", 0.0), t, issue)
        low_conf = s["confidence"] < REVIEW_BELOW
        reply = suggest_reply(s["sentiment"], issue)
        if low_conf and s["sentiment"] != "negative":
            reply = "[Low confidence - check this tweet before replying]"
        rows.append({
            "sentiment": s["sentiment"], "sentiment_conf": s["confidence"],
            "issue": issue, "issue_conf": iss["confidence"] if iss else None,
            "urgency": u["score"], "urgency_level": u["level"],
            "urgency_reasons": "; ".join(u["reasons"]),
            "suggested_reply": reply, "needs_review": low_conf,
        })
    return pd.DataFrame(rows)


def analyze_dataframe(df, backend="baseline"):
    res = analyze_texts(df["text"].tolist(), backend)
    return pd.concat([df.reset_index(drop=True), res], axis=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/processed/airline.csv")
    ap.add_argument("--model", default="baseline")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--out", default="data/processed/analyzed.csv")
    a = ap.parse_args()

    df = pd.read_csv(a.input)
    if a.n:
        df = df.sample(min(a.n, len(df)), random_state=1)
    out = analyze_dataframe(df, a.model)
    out.to_csv(a.out, index=False)

    print(out["sentiment"].value_counts(), "\n")
    print(out["issue"].value_counts(), "\n")
    print(out["urgency_level"].value_counts(), "\n")
    top = out.sort_values("urgency", ascending=False).head(8)
    for _, r in top.iterrows():
        print(f"[{r.urgency}] {r.urgency_level} | {r.issue} | {r.text[:90]}\n    why: {r.urgency_reasons}")