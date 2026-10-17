"""Streamlit dashboard. Run: streamlit run app/dashboard.py"""
import os
import sys
from html import escape

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app import theme
from src.predict import available_backends, get_predictor


st.set_page_config(page_title="Airline Customer Feedback", layout="wide")
theme.apply_theme()

backends = available_backends()
if not backends:
    st.error("No trained model is available. Select or train a model before using the dashboard.")
    st.stop()

model_name = st.sidebar.selectbox("Sentiment model", backends)
predictor = get_predictor(model_name)
COLORS = {
    "positive": theme.POSITIVE,
    "neutral": theme.NEUTRAL,
    "negative": theme.NEGATIVE,
}

header_slot = st.empty()


def analyze(df: pd.DataFrame) -> pd.DataFrame:
    results = predictor.predict(df["text"].astype(str).tolist())
    analyzed = df.copy()
    analyzed["sentiment"] = [result["sentiment"] for result in results]
    analyzed["confidence"] = [result["confidence"] for result in results]
    return analyzed


def show_results(df: pd.DataFrame):
    if df.empty:
        st.info("No tweets to show for this selection.")
        return

    total = len(df)
    counts = df["sentiment"].value_counts().reindex(
        ["negative", "neutral", "positive"], fill_value=0
    )
    negative_pct = counts["negative"] * 100 / total
    dominant = counts.idxmax()
    tone = "negative" if dominant == "negative" else "positive" if dominant == "positive" else "accent"
    dated = df.copy()
    if "created_at" in dated.columns:
        dated["created_at"] = pd.to_datetime(dated["created_at"], errors="coerce")
        dated = dated.dropna(subset=["created_at"])
    st.session_state["feedback_date_range"] = (
        f"{dated['created_at'].min():%d %b %Y} – {dated['created_at'].max():%d %b %Y}"
        if "created_at" in dated.columns and not dated.empty else "Not provided"
    )
    daily = pd.DataFrame()
    if not dated.empty and "created_at" in dated.columns:
        daily = dated.groupby([
            pd.Grouper(key="created_at", freq="D"), "sentiment"
        ]).size().unstack(fill_value=0)
        daily = daily.reindex(columns=["negative", "neutral", "positive"], fill_value=0)
    theme.insight_banner(
        f"{negative_pct:.1f}% of {total:,} analyzed tweets are negative; "
        f"{dominant} feedback is the largest sentiment group.", tone
    )

    confidence = df["confidence"] if "confidence" in df.columns else pd.Series(dtype=float)
    average_confidence = confidence.mean() if not confidence.empty else 0
    daily_total = daily.sum(axis=1) if not daily.empty else pd.Series(dtype=float)
    negative_spark = (
        (daily["negative"] * 100 / daily_total).tolist() if not daily.empty else None
    )
    negative_delta = None
    if len(negative_spark or []) >= 2:
        negative_delta = f"{negative_spark[-1] - negative_spark[-2]:+.1f} pp"
    spark_data = [
        daily_total.tolist() if not daily.empty else None,
        negative_spark,
        daily["neutral"].tolist() if not daily.empty else None,
        daily["positive"].tolist() if not daily.empty else None,
    ]
    metric_columns = st.columns(4)
    for column, label, value, color, delta, data in zip(
        metric_columns,
        ["Tweets analyzed", "Negative feedback", "Neutral feedback", "Positive feedback"],
        [f"{total:,}", f"{negative_pct:.1f}%", f"{counts['neutral']:,}", f"{counts['positive']:,}"],
        ["accent", "negative", "neutral", "positive"],
        [None, negative_delta, None, None],
        spark_data,
    ):
        with column:
            theme.kpi_card(label, value, delta=delta, chart_data=data, tone=color)
    st.caption(f"Mean prediction confidence: {average_confidence:.1%}")

    left, right = st.columns([2, 1])
    with left:
        if daily.empty:
            st.info("No valid dates are available for a sentiment trend.")
        else:
            trend = daily.copy()
            trend["pct_negative"] = trend["negative"] * 100 / trend.sum(axis=1)
            peak_date = trend["pct_negative"].idxmax()
            peak_value = trend.loc[peak_date, "pct_negative"]
            figure = go.Figure(go.Scatter(
                x=trend.index, y=trend["pct_negative"], mode="lines+markers",
                line={"color": theme.NEGATIVE, "width": 2},
                marker={"color": theme.NEGATIVE, "size": 5},
                hovertemplate="%{x|%d %b}<br>%{y:.1f}% negative<extra></extra>",
            ))
            figure.add_annotation(
                x=peak_date, y=peak_value, text=f"Peak {peak_value:.1f}%",
                showarrow=True, arrowcolor=theme.NEGATIVE,
            )
            figure.update_layout(
                title=f"Negative feedback peaked at {peak_value:.1f}% on {peak_date:%d %b}",
                xaxis_title=None, yaxis_title="Negative tweets (%)", showlegend=False,
            )
            with st.container(border=True):
                st.plotly_chart(theme.style_fig(figure), width="stretch")

    with right:
        sentiment_counts = counts.rename_axis("sentiment").reset_index(name="count")
        share = sentiment_counts.loc[sentiment_counts["count"].idxmax(), "count"] * 100 / total
        figure = px.bar(
            sentiment_counts, x="count", y="sentiment", orientation="h",
            color="sentiment", color_discrete_map=COLORS,
            title=f"{dominant.title()} leads the sentiment mix ({share:.1f}%)",
        )
        figure.update_layout(showlegend=False, yaxis={"autorange": "reversed"},
                             xaxis_title="Tweets", yaxis_title=None)
        with st.container(border=True):
            st.plotly_chart(theme.style_fig(figure), width="stretch")

    if "urgency_level" in df.columns:
        levels = ["Critical", "High", "Medium", "Low"]
        urgency_counts = df["urgency_level"].value_counts().reindex(levels, fill_value=0)
        urgent = int(urgency_counts["Critical"] + urgency_counts["High"])
        if urgency_counts.sum():
            figure = go.Figure()
            urgency_colors = {
                "Critical": theme.NEGATIVE, "High": "#ea580c",
                "Medium": "#ca8a04", "Low": theme.NEUTRAL,
            }
            for level in levels:
                figure.add_trace(go.Bar(
                    x=[urgency_counts[level]], y=["Tweets"], name=level,
                    orientation="h", marker_color=urgency_colors[level],
                ))
            figure.update_layout(
                barmode="stack", height=105,
                title=f"{urgent:,} tweets need a high-priority reply",
                xaxis_title="Tweets", yaxis_title=None,
            )
            with st.container(border=True):
                st.plotly_chart(theme.style_fig(figure), width="stretch")
        else:
            st.info("No urgency levels are available.")

    with st.expander("Tweet-level results"):
        st.dataframe(df, width="stretch", hide_index=True)
        st.download_button("Download CSV", df.to_csv(index=False), "results.csv")


tab1, tab2, tab3, tab4 = st.tabs(
    ["Single tweet", "Upload CSV", "Live (X API)", "Model comparison"]
)

with tab1:
    _, center, _ = st.columns([1, 3, 1])
    with center:
        theme.insight_banner("Check the sentiment, urgency, and suggested next step for one tweet.")
        examples = [
            "My flight was delayed and I missed my connection.",
            "The crew were kind and made the trip great.",
            "I need help changing my booking.",
        ]
        selected_example = st.pills("Example tweets", examples, key="tweet_example")
        if selected_example is None:
            st.session_state.pop("applied_tweet_example", None)
        elif selected_example != st.session_state.get("applied_tweet_example"):
            st.session_state["single_tweet_text"] = selected_example
            st.session_state["applied_tweet_example"] = selected_example
        text = st.text_area(
            "Tweet", "I absolutely love the new update, great job!",
            height=150, key="single_tweet_text",
        )
        analyze_clicked = st.button("Analyze", type="primary", width="stretch")

        if analyze_clicked and text.strip():
            st.session_state.pop("explain_tweet_checkbox", None)
            st.session_state["feedback_source"] = "Single tweet"
            st.session_state["feedback_date_range"] = "Not applicable"
            with st.spinner("Analyzing tweet..."):
                sentiment = predictor.predict([text])[0]
                issue = None
                if sentiment["sentiment"] == "negative":
                    try:
                        from src.issue import predict_issue
                        issue = predict_issue([text])[0]["issue"]
                    except FileNotFoundError:
                        issue = "Unclear"
                from src.replies import suggest_reply
                from src.urgency import compute_urgency
                urgency = compute_urgency(
                    sentiment["probabilities"].get("negative", 0.0), text, issue
                )
                st.session_state["single_tweet_result"] = {
                    "text": text, "backend": model_name, "sentiment": sentiment,
                    "issue": issue, "urgency": urgency,
                    "reply": suggest_reply(sentiment["sentiment"], issue),
                }

        result = st.session_state.get("single_tweet_result")
        if result and result["text"] == text and result["backend"] == model_name:
            sentiment = result["sentiment"]
            urgency = result["urgency"]
            with st.container(border=True):
                sentiment_tone = sentiment["sentiment"]
                st.markdown(theme.pill(sentiment["sentiment"].upper(), sentiment_tone),
                            unsafe_allow_html=True)
                st.markdown(f"## {sentiment['confidence']:.1%} confidence")
                st.progress(sentiment["confidence"])
                result_columns = st.columns(3)
                with result_columns[0]:
                    st.caption("Issue")
                    st.write(result["issue"] or "No complaint issue")
                with result_columns[1]:
                    st.caption("Urgency")
                    st.markdown(theme.pill(
                        f"{urgency['level']} · {urgency['score']}/100",
                        urgency["level"].lower(),
                    ), unsafe_allow_html=True)
                    st.progress(urgency["score"] / 100)
                with result_columns[2]:
                    st.caption("Suggested reply")
                    st.code(result["reply"], language=None)

                if sentiment["confidence"] < 0.5:
                    st.warning("Low-confidence prediction: check this tweet before replying.")
                if urgency["reasons"]:
                    st.markdown(" ".join(theme.chip(reason) for reason in urgency["reasons"]),
                                unsafe_allow_html=True)

                if st.checkbox("Explain this prediction (slower)", key="explain_tweet_checkbox"):
                    from src.explain import explain_tweet

                    sample_count = 300 if model_name == "bert" else 500
                    with st.spinner("Generating approximate word explanation..."):
                        explanation = explain_tweet(
                            text, backend=model_name, num_samples=sample_count
                        )
                    st.markdown(explanation["html"], unsafe_allow_html=True)
                    words = explanation["words"][:8]
                    if words:
                        weights = pd.DataFrame(words, columns=["word", "weight"])
                        figure = px.bar(weights, x="weight", y="word", orientation="h")
                        figure.update_traces(marker_color=[
                            theme.POSITIVE if weight > 0 else theme.NEGATIVE
                            for weight in weights["weight"]
                        ])
                        figure.update_layout(
                            title="Words influencing this prediction",
                            xaxis_title="Weight toward predicted sentiment", yaxis_title=None,
                            showlegend=False,
                        )
                        st.plotly_chart(theme.style_fig(figure), width="stretch")
                    st.caption("Highlighting is approximate (LIME) and may vary slightly between runs.")

with tab2:
    theme.insight_banner("Review a labeled overview of sentiment from an uploaded tweet file.")
    with st.container(border=True):
        uploaded = st.file_uploader(
            "CSV with a text column and optional created_at column", type="csv"
        )
    if uploaded:
        data = pd.read_csv(uploaded)
        if "text" not in data.columns:
            st.error("CSV must have a 'text' column.")
        else:
            with st.spinner("Analyzing uploaded tweets..."):
                st.session_state["feedback_source"] = "Uploaded CSV"
                show_results(analyze(data))

with tab3:
    theme.insight_banner("Fetch a small live sample and review its sentiment. Requires an X API bearer token.")
    with st.container(border=True):
        query = st.text_input("Keyword or hashtag", "#python")
        count = st.slider("Number of tweets", 10, 100, 50)
        fetch_clicked = st.button("Fetch and analyze", type="primary")
    if fetch_clicked:
        try:
            from src.data_collection import fetch_tweets

            live_data = fetch_tweets(query, count)
            if live_data.empty:
                st.info("No tweets were returned for this query.")
            else:
                with st.spinner("Analyzing live tweets..."):
                    st.session_state["feedback_source"] = "Live X API"
                    show_results(analyze(live_data))
        except Exception as error:
            st.error(f"Could not fetch tweets: {error}")
    st.caption("Live access depends on X API credentials and availability.")

with tab4:
    from src.evaluate import compare_models

    comparison = compare_models()
    if comparison.empty:
        theme.insight_banner("No model results are available yet.", "neutral")
        st.info("Train and evaluate models to populate this comparison.")
    else:
        comparison = comparison.sort_values("f1_macro", ascending=False).reset_index(drop=True)
        best = comparison.iloc[0]
        if len(comparison) > 1:
            baseline = comparison.loc[comparison["model"] == "baseline_logreg"]
            reference = baseline.iloc[0] if not baseline.empty else comparison.iloc[-1]
            message = (
                f"{best['model']} leads on macro-F1 ({best['f1_macro']:.1%}) versus "
                f"{reference['model']} ({reference['f1_macro']:.1%})."
            )
        else:
            message = f"{best['model']} is the top available model with macro-F1 {best['f1_macro']:.1%}."
        theme.insight_banner(message)

        metric_columns = st.columns(min(3, len(comparison)))
        for column, (_, row) in zip(metric_columns, comparison.head(3).iterrows()):
            with column:
                theme.kpi_card(
                    f"{row['model']} accuracy", f"{row['accuracy']:.1%}",
                    tone="accent" if row["model"] == best["model"] else "neutral",
                )

        with st.container(border=True):
            scores = comparison["f1_macro"].tolist()
            peak = float(comparison["f1_macro"].max())
            figure = go.Figure(go.Bar(
                x=comparison["f1_macro"], y=comparison["model"], orientation="h",
                marker_color=[theme.ACCENT if value == peak else "#64748b" for value in scores],
                text=[f"{value:.1%}" for value in scores], textposition="outside",
            ))
            figure.update_layout(
                title=f"{best['model']} has the strongest macro-F1 ({peak:.1%})",
                xaxis_title="Macro-F1", yaxis_title=None, showlegend=False,
                yaxis={"autorange": "reversed"},
            )
            st.plotly_chart(theme.style_fig(figure), width="stretch")

        display_columns = [
            column for column in ["model", "accuracy", "precision", "recall", "f1_macro"]
            if column in comparison.columns
        ]
        with st.expander("Detailed model metrics"):
            st.dataframe(comparison[display_columns], hide_index=True, width="stretch")

        results_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "results"
        )
        if os.path.isdir(results_dir):
            figures = sorted(name for name in os.listdir(results_dir) if name.endswith(".png"))
            if figures:
                with st.expander("Confusion matrices"):
                    selected_figure = st.selectbox("Model result", figures)
                    st.image(os.path.join(results_dir, selected_figure), caption=selected_figure)

with header_slot.container():
    header_left, header_right = st.columns([2, 1])
    with header_left:
        st.title("Airline Customer Feedback")
    with header_right:
        source = escape(st.session_state.get("feedback_source", "CSV upload or live feed"))
        date_range = escape(st.session_state.get("feedback_date_range", "Awaiting dated data"))
        st.markdown(
            f'<div class="feedback-header-meta">Source: {source}<br>'
            f'Date range: {date_range}<br>Model: {escape(model_name)}</div>',
            unsafe_allow_html=True,
        )

st.markdown(
    '<div class="feedback-footer">Predictions are estimates; review sensitive or low-confidence '
    'cases before responding. LIME highlights are approximate.</div>',
    unsafe_allow_html=True,
)
