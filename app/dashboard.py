"""Streamlit dashboard.  Run: streamlit run app/dashboard.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import plotly.express as px
import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt

from src.predict import get_predictor, available_backends
from src.preprocessing import clean_tweet

st.set_page_config(page_title="Twitter/X Sentiment Analyzer", page_icon="📊", layout="wide")
st.title("📊 Twitter/X Sentiment Analyzer")

backends = available_backends()
if not backends:
    st.error("No trained model found. Run `python -m src.train_baseline` first (see README).")
    st.stop()

model_name = st.sidebar.selectbox("Model", backends)
predictor = get_predictor(model_name)
COLORS = {"positive": "#2ecc71", "neutral": "#95a5a6", "negative": "#e74c3c"}


def analyze(df: pd.DataFrame) -> pd.DataFrame:
    res = predictor.predict(df["text"].astype(str).tolist())
    df = df.copy()
    df["sentiment"] = [r["sentiment"] for r in res]
    df["confidence"] = [r["confidence"] for r in res]
    return df


def show_results(df: pd.DataFrame):
    c1, c2 = st.columns(2)
    counts = df["sentiment"].value_counts().reset_index()
    counts.columns = ["sentiment", "count"]
    c1.plotly_chart(px.pie(counts, names="sentiment", values="count", color="sentiment",
                           color_discrete_map=COLORS, title="Sentiment distribution"),
                    use_container_width=True)
    if "created_at" in df.columns:
        d = df.copy(); d["created_at"] = pd.to_datetime(d["created_at"])
        t = d.groupby([pd.Grouper(key="created_at", freq="h"), "sentiment"]).size().reset_index(name="count")
        c2.plotly_chart(px.line(t, x="created_at", y="count", color="sentiment",
                                color_discrete_map=COLORS, title="Sentiment over time"),
                        use_container_width=True)
    else:
        c2.plotly_chart(px.histogram(df, x="confidence", color="sentiment", color_discrete_map=COLORS,
                                     title="Confidence distribution"), use_container_width=True)

    st.subheader("Word clouds")
    cols = st.columns(3)
    for col, s in zip(cols, ["positive", "neutral", "negative"]):
        text = " ".join(clean_tweet(t) for t in df.loc[df["sentiment"] == s, "text"])
        if text.strip():
            wc = WordCloud(width=500, height=300, background_color="white").generate(text)
            fig, ax = plt.subplots(); ax.imshow(wc); ax.axis("off"); ax.set_title(s)
            col.pyplot(fig)
    st.dataframe(df, use_container_width=True)
    st.download_button("Download results (CSV)", df.to_csv(index=False), "results.csv")


tab1, tab2, tab3, tab4 = st.tabs(["Single tweet", "Upload CSV", "Live (X API)", "Model comparison"])

with tab1:
    text = st.text_area("Enter a tweet", "I absolutely love the new update, great job!")
    if st.button("Analyze") and text.strip():
        r = predictor.predict([text])[0]
        st.markdown(f"### Sentiment: **{r['sentiment'].upper()}** ({r['confidence']*100:.1f}% confident)")
        st.bar_chart(pd.Series(r["probabilities"]))

with tab2:
    f = st.file_uploader("CSV with a `text` column (optional `created_at`)", type="csv")
    if f:
        df = pd.read_csv(f)
        if "text" not in df.columns:
            st.error("CSV must have a 'text' column.")
        else:
            with st.spinner("Analyzing..."):
                show_results(analyze(df))

with tab3:
    st.caption("Requires an X API bearer token (env var X_BEARER_TOKEN). API access/limits change, so check developer.x.com.")
    q = st.text_input("Keyword / hashtag", "#python")
    n = st.slider("Number of tweets", 10, 100, 50)
    if st.button("Fetch & analyze"):
        try:
            from src.data_collection import fetch_tweets
            df = fetch_tweets(q, n)
            if df.empty:
                st.warning("No tweets returned.")
            else:
                show_results(analyze(df))
        except Exception as e:
            st.error(f"Could not fetch tweets: {e}")

with tab4:
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "results")
    from src.evaluate import compare_models
    cmp = compare_models()
    if cmp.empty:
        st.info("Train models first to see the comparison.")
    else:
        st.dataframe(cmp, use_container_width=True)
        st.plotly_chart(px.bar(cmp.melt(id_vars="model", value_vars=["accuracy", "f1_macro"]),
                               x="model", y="value", color="variable", barmode="group",
                               title="Model comparison"), use_container_width=True)
        for fn in sorted(os.listdir(p)):
            if fn.endswith(".png"):
                st.image(os.path.join(p, fn), caption=fn, width=380)
