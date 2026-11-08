"""Streamlit dashboard. Run: streamlit run app/dashboard.py"""

import os
import sys
from html import escape
from math import ceil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app import theme
from src.batch import analyze_batch
from src.db import (
    clear_predictions,
    delete_predictions,
    get_predictions,
    init_db,
    save_prediction,
)
from src.insights import sentiment_over_time, top_keywords
from src.predict import available_backends, get_predictor
from src.sources.youtube_source import YouTubeSource

init_db()

st.set_page_config(page_title="Twitter Sentiment Analysis", layout="wide")
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


def show_result_metrics(df, total, counts, negative_pct, daily):
    confidence = df["confidence"] if "confidence" in df.columns else pd.Series(dtype=float)
    average_confidence = confidence.mean() if not confidence.empty else 0
    daily_total = daily.sum(axis=1) if not daily.empty else pd.Series(dtype=float)
    negative_spark = (daily["negative"] * 100 / daily_total).tolist() if not daily.empty else None
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
        ["Posts analyzed", "Negative feedback", "Neutral feedback", "Positive feedback"],
        [f"{total:,}", f"{negative_pct:.1f}%", f"{counts['neutral']:,}", f"{counts['positive']:,}"],
        ["accent", "negative", "neutral", "positive"],
        [None, negative_delta, None, None],
        spark_data,
    ):
        with column:
            theme.kpi_card(label, value, delta=delta, chart_data=data, tone=color)
    st.caption(f"Mean prediction confidence: {average_confidence:.1%}")


def show_sentiment_trend(daily):
    if daily.empty:
        st.info("No valid dates are available for a sentiment trend.")
        return

    trend = daily.copy()
    trend["pct_negative"] = trend["negative"] * 100 / trend.sum(axis=1)
    peak_date = trend["pct_negative"].idxmax()
    peak_value = trend.loc[peak_date, "pct_negative"]
    figure = go.Figure(
        go.Scatter(
            x=trend.index,
            y=trend["pct_negative"],
            mode="lines+markers",
            line={"color": theme.NEGATIVE, "width": 2},
            marker={"color": theme.NEGATIVE, "size": 5},
            hovertemplate="%{x|%d %b}<br>%{y:.1f}% negative<extra></extra>",
        )
    )
    figure.add_annotation(
        x=peak_date,
        y=peak_value,
        text=f"Peak {peak_value:.1f}%",
        showarrow=True,
        arrowcolor=theme.NEGATIVE,
    )
    figure.update_layout(
        title=f"Negative feedback peaked at {peak_value:.1f}% on {peak_date:%d %b}",
        xaxis_title=None,
        yaxis_title="Negative posts (%)",
        showlegend=False,
    )
    with st.container(border=True):
        st.plotly_chart(theme.style_fig(figure), width="stretch")


def show_sentiment_mix(counts, total, dominant):
    sentiment_counts = counts.rename_axis("sentiment").reset_index(name="count")
    share = sentiment_counts.loc[sentiment_counts["count"].idxmax(), "count"] * 100 / total
    figure = px.bar(
        sentiment_counts,
        x="count",
        y="sentiment",
        orientation="h",
        color="sentiment",
        color_discrete_map=COLORS,
        title=f"{dominant.title()} leads the sentiment mix ({share:.1f}%)",
    )
    figure.update_layout(
        showlegend=False, yaxis={"autorange": "reversed"}, xaxis_title="Posts", yaxis_title=None
    )
    with st.container(border=True):
        st.plotly_chart(theme.style_fig(figure), width="stretch")


def show_urgency_summary(df):
    if "urgency_level" not in df.columns:
        return

    with st.expander("Optional airline-specific service guidance"):
        levels = ["Critical", "High", "Medium", "Low"]
        urgency_counts = df["urgency_level"].value_counts().reindex(levels, fill_value=0)
        urgent = int(urgency_counts["Critical"] + urgency_counts["High"])
        if not urgency_counts.sum():
            st.info("No urgency levels are available.")
            return

        urgency_colors = {
            "Critical": theme.NEGATIVE,
            "High": "#ea580c",
            "Medium": "#ca8a04",
            "Low": theme.NEUTRAL,
        }
        figure = go.Figure()
        for level in levels:
            figure.add_trace(
                go.Bar(
                    x=[urgency_counts[level]],
                    y=["Posts"],
                    name=level,
                    orientation="h",
                    marker_color=urgency_colors[level],
                )
            )
        figure.update_layout(
            barmode="stack",
            height=105,
            title=f"{urgent:,} posts need a high-priority reply",
            xaxis_title="Posts",
            yaxis_title=None,
        )
        st.plotly_chart(theme.style_fig(figure), width="stretch")


def show_results(df: pd.DataFrame):
    if df.empty:
        st.info("No posts to show for this selection.")
        return

    total = len(df)
    counts = (
        df["sentiment"].value_counts().reindex(["negative", "neutral", "positive"], fill_value=0)
    )
    negative_pct = counts["negative"] * 100 / total
    dominant = counts.idxmax()
    tone = (
        "negative" if dominant == "negative" else "positive" if dominant == "positive" else "accent"
    )
    dated = df.copy()
    if "created_at" in dated.columns:
        dated["created_at"] = pd.to_datetime(dated["created_at"], errors="coerce")
        dated = dated.dropna(subset=["created_at"])
    st.session_state["feedback_date_range"] = (
        f"{dated['created_at'].min():%d %b %Y} – {dated['created_at'].max():%d %b %Y}"
        if "created_at" in dated.columns and not dated.empty
        else "Not provided"
    )
    daily = pd.DataFrame()
    if not dated.empty and "created_at" in dated.columns:
        daily = (
            dated.groupby([pd.Grouper(key="created_at", freq="D"), "sentiment"])
            .size()
            .unstack(fill_value=0)
        )
        daily = daily.reindex(columns=["negative", "neutral", "positive"], fill_value=0)
    theme.insight_banner(
        f"{negative_pct:.1f}% of {total:,} analyzed posts are negative; "
        f"{dominant} feedback is the largest sentiment group.",
        tone,
    )
    show_result_metrics(df, total, counts, negative_pct, daily)

    left, right = st.columns([2, 1])
    with left:
        show_sentiment_trend(daily)

    with right:
        show_sentiment_mix(counts, total, dominant)

    show_urgency_summary(df)

    with st.expander("Post-level results"):
        st.dataframe(df, width="stretch", hide_index=True)
        st.download_button("Download CSV", df.to_csv(index=False), "results.csv")


@st.cache_data(show_spinner=False, ttl=3600)
def get_youtube_video_details(video_id):
    return YouTubeSource(video_id).fetch_video_details()


@st.cache_data(show_spinner=False, ttl=3600)
def analyze_youtube_video(video_id, comment_limit, order, model_name, _progress_callback=None):
    source = YouTubeSource(video_id)
    posts = source.fetch(limit=comment_limit, order=order)
    posts = [post for post in posts if post.text and post.text.strip()]
    predictions = analyze_batch(
        [post.text for post in posts], model_name, progress_callback=_progress_callback
    )
    return [
        {
            "text": post.text,
            "sentiment": prediction["sentiment"],
            "confidence": prediction["confidence"],
            "published_at": post.meta.get("published_at", ""),
            "posted_at": post.meta.get("published_at") or None,
        }
        for post, prediction in zip(posts, predictions)
    ]


tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Single text",
        "Batch analysis",
        "YouTube analysis",
        "Model comparison",
        "Insights",
    ]
)

with tab1:
    _, center, _ = st.columns([1, 3, 1])
    with center:
        theme.insight_banner("Check sentiment for a short text, with optional service guidance.")
        examples = [
            "This phone has a sharp camera and great battery life.",
            "The new movie kept me smiling from start to finish.",
            "Our team played brilliant cricket today.",
        ]
        selected_example = st.pills("Example texts", examples, key="text_example")
        if selected_example is None:
            st.session_state.pop("applied_text_example", None)
        elif selected_example != st.session_state.get("applied_text_example"):
            st.session_state["single_tweet_text"] = selected_example
            st.session_state["applied_text_example"] = selected_example
        text = st.text_area(
            "Text",
            "The new phone update works beautifully!",
            height=150,
            key="single_tweet_text",
        )
        analyze_clicked = st.button("Analyze", type="primary", width="stretch")

        if analyze_clicked and text.strip():
            st.session_state.pop("explain_tweet_checkbox", None)
            st.session_state.pop("single_text_guidance", None)
            st.session_state.pop("single_text_guidance_checkbox", None)
            st.session_state["feedback_source"] = "Single text"
            st.session_state["feedback_date_range"] = "Not applicable"
            with st.spinner("Analyzing tweet..."):
                sentiment = predictor.predict([text])[0]
                save_prediction(
                    text,
                    sentiment["sentiment"],
                    sentiment["confidence"],
                    model_name,
                    source="dashboard",
                )
                st.session_state["single_tweet_result"] = {
                    "text": text,
                    "backend": model_name,
                    "sentiment": sentiment,
                }

        result = st.session_state.get("single_tweet_result")
        if result and result["text"] == text and result["backend"] == model_name:
            sentiment = result["sentiment"]
            with st.container(border=True):
                sentiment_tone = sentiment["sentiment"]
                st.markdown(
                    theme.pill(sentiment["sentiment"].upper(), sentiment_tone),
                    unsafe_allow_html=True,
                )
                st.markdown(f"## {sentiment['confidence']:.1%} confidence")
                st.progress(sentiment["confidence"])

                if sentiment["confidence"] < 0.5:
                    st.warning("Low-confidence prediction: check this tweet before replying.")

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
                        figure.update_traces(
                            marker_color=[
                                theme.POSITIVE if weight > 0 else theme.NEGATIVE
                                for weight in weights["weight"]
                            ]
                        )
                        figure.update_layout(
                            title="Words influencing this prediction",
                            xaxis_title="Weight toward predicted sentiment",
                            yaxis_title=None,
                            showlegend=False,
                        )
                        st.plotly_chart(theme.style_fig(figure), width="stretch")
                    st.caption(
                        "Highlighting is approximate (LIME) and may vary slightly between runs."
                    )

            with st.expander("Optional airline-specific service guidance"):
                st.caption(
                    "Issue labels, urgency scores, and reply suggestions are tuned for airline feedback."
                )
                guidance_enabled = st.checkbox(
                    "Generate issue, urgency, and reply guidance",
                    key="single_text_guidance_checkbox",
                )
                guidance_key = (text, model_name)
                guidance = st.session_state.get("single_text_guidance")
                if guidance_enabled and (not guidance or guidance["key"] != guidance_key):
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
                    guidance = {
                        "key": guidance_key,
                        "issue": issue,
                        "urgency": urgency,
                        "reply": suggest_reply(sentiment["sentiment"], issue),
                    }
                    st.session_state["single_text_guidance"] = guidance

                if guidance_enabled and guidance and guidance["key"] == guidance_key:
                    guidance_columns = st.columns(3)
                    with guidance_columns[0]:
                        st.caption("Issue")
                        st.write(guidance["issue"] or "No issue classification")
                    with guidance_columns[1]:
                        st.caption("Urgency")
                        urgency = guidance["urgency"]
                        st.markdown(
                            theme.pill(
                                f"{urgency['level']} · {urgency['score']}/100",
                                urgency["level"].lower(),
                            ),
                            unsafe_allow_html=True,
                        )
                        st.progress(urgency["score"] / 100)
                    with guidance_columns[2]:
                        st.caption("Suggested reply")
                        st.code(guidance["reply"], language=None)

with tab2:
    theme.insight_banner("Analyze a CSV column and review sentiment across its non-empty rows.")
    uploaded_batch = st.file_uploader("Upload CSV", type="csv", key="batch_upload")
    if uploaded_batch:
        try:
            batch_data = pd.read_csv(uploaded_batch)
        except Exception as error:
            st.error(f"Could not read the uploaded CSV: {error}")
            batch_data = pd.DataFrame()

        if not len(batch_data.columns):
            st.info("Upload a CSV file with at least one column.")
        else:
            columns = batch_data.columns.tolist()
            default_column = columns.index("text") if "text" in columns else 0
            text_column = st.selectbox("Text column", columns, index=default_column)
            batch_model = st.selectbox(
                "Sentiment model", backends, index=backends.index(model_name), key="batch_model"
            )
            maximum_rows = st.slider("Maximum rows", 1, 5000, 500)
            save_to_history = st.checkbox("Save results to history database", value=True)

            if st.button("Analyze", type="primary", key="analyze_batch"):
                from src.batch import analyze_batch

                selected_rows = batch_data.head(maximum_rows)
                selected_texts = selected_rows[text_column]
                valid_mask = selected_texts.notna() & selected_texts.astype(str).str.strip().ne("")
                with st.spinner("Analyzing CSV rows..."):
                    progress = st.progress(0.0, text="Preparing batch analysis...")
                    batch_results = analyze_batch(
                        selected_texts.tolist(),
                        batch_model,
                        progress_callback=lambda value: progress.progress(
                            value, text=f"Analyzing rows: {value:.0%}"
                        ),
                    )

                result_rows = selected_rows.loc[valid_mask].copy()
                result_texts = selected_texts.loc[valid_mask].astype(str).tolist()
                if text_column != "text":
                    if "text" in result_rows.columns:
                        result_rows = result_rows.rename(columns={"text": "original_text"})
                    result_rows.insert(0, "text", result_texts)
                else:
                    result_rows["text"] = result_texts
                result_rows["sentiment"] = [item["sentiment"] for item in batch_results]
                result_rows["confidence"] = [item["confidence"] for item in batch_results]

                st.session_state["batch_analysis_results"] = result_rows
                st.session_state["batch_analysis_key"] = (
                    uploaded_batch.name,
                    text_column,
                    batch_model,
                    maximum_rows,
                )
                st.session_state["feedback_source"] = "Batch analysis"
                if save_to_history:
                    for result in batch_results:
                        save_prediction(
                            result["text"],
                            result["sentiment"],
                            result["confidence"],
                            batch_model,
                            source="upload",
                        )

            batch_key = (uploaded_batch.name, text_column, batch_model, maximum_rows)
            if st.session_state.get("batch_analysis_key") == batch_key:
                results_frame = st.session_state["batch_analysis_results"]
                if results_frame.empty:
                    st.info("No non-empty text rows were found within the selected limit.")
                    st.dataframe(results_frame, width="stretch", hide_index=True)
                    st.download_button(
                        "Download results as CSV",
                        results_frame.to_csv(index=False),
                        "batch_results.csv",
                        mime="text/csv",
                    )
                else:
                    show_results(results_frame)

with tab3:
    theme.insight_banner("Analyze public comments on a YouTube video.")
    video_url = st.text_input("YouTube URL or video ID")
    youtube_count = st.slider(
        "Number of comments", 50, 1000, 200, step=50, key="youtube_comment_count"
    )
    youtube_order_label = st.selectbox("Order", ["Newest", "Relevance"])
    youtube_model = st.selectbox(
        "Sentiment model", backends, index=backends.index(model_name), key="youtube_model"
    )
    save_youtube_results = st.checkbox("Save results to history", value=True)
    youtube_order = {"Newest": "newest", "Relevance": "relevance"}[youtube_order_label]

    if video_url.strip():
        try:
            video_id = YouTubeSource.extract_video_id(video_url.strip())
        except ValueError:
            st.error("Enter a valid YouTube URL or video ID.")
        else:
            try:
                video_details = get_youtube_video_details(video_id)
            except ValueError as error:
                st.info(f"{error} Set YOUTUBE_API_KEY in your environment or .env file.")
            except RuntimeError as error:
                st.error(str(error))
            else:
                st.subheader(video_details["title"])
                st.caption(f"Total comments: {video_details['comment_count']:,}")
                estimated_comment_units = ceil(youtube_count / 100)
                st.caption(
                    f"Estimated quota on a cache miss: about {estimated_comment_units} units "
                    "for comments; video details use 1 unit and are cached."
                )

                analysis_key = (video_id, youtube_count, youtube_order, youtube_model)
                if st.button("Analyze", type="primary", key="analyze_youtube"):
                    progress = st.progress(0.0, text="Fetching and analyzing comments...")

                    def update_youtube_progress(value):
                        progress.progress(value, text=f"Analyzing comments: {value:.0%}")

                    try:
                        results = analyze_youtube_video(
                            video_id,
                            youtube_count,
                            youtube_order,
                            youtube_model,
                            _progress_callback=update_youtube_progress,
                        )
                    except ValueError as error:
                        progress.empty()
                        st.info(f"{error} Set YOUTUBE_API_KEY in your environment or .env file.")
                    except RuntimeError as error:
                        progress.empty()
                        st.error(str(error))
                    else:
                        progress.progress(1.0, text="Analysis complete")
                        st.session_state["youtube_analysis"] = {
                            "key": analysis_key,
                            "results": results,
                        }
                        st.session_state["feedback_source"] = "YouTube analysis"
                        saved_key = st.session_state.get("youtube_saved_key")
                        if save_youtube_results and saved_key != analysis_key:
                            for result in results:
                                save_prediction(
                                    result["text"],
                                    result["sentiment"],
                                    result["confidence"],
                                    youtube_model,
                                    source="youtube",
                                    posted_at=result["posted_at"],
                                )
                            st.session_state["youtube_saved_key"] = analysis_key

                analysis = st.session_state.get("youtube_analysis")
                if analysis and analysis["key"] == analysis_key:
                    results = analysis["results"]
                    results_frame = pd.DataFrame(
                        results,
                        columns=["text", "sentiment", "confidence", "published_at", "posted_at"],
                    )
                    total_comments = len(results_frame)
                    sentiment_counts = results_frame["sentiment"].value_counts().reindex(
                        ["positive", "neutral", "negative"], fill_value=0
                    )
                    percentages = {
                        sentiment: sentiment_counts[sentiment] * 100 / total_comments
                        if total_comments
                        else 0
                        for sentiment in ["positive", "neutral", "negative"]
                    }
                    average_confidence = (
                        results_frame["confidence"].mean() if total_comments else 0
                    )
                    metric_columns = st.columns(5)
                    metric_columns[0].metric("Comments analyzed", f"{total_comments:,}")
                    for column, sentiment in zip(
                        metric_columns[1:], ["positive", "neutral", "negative"]
                    ):
                        column.metric(f"{sentiment.title()}", f"{percentages[sentiment]:.1f}%")
                    metric_columns[4].metric("Average confidence", f"{average_confidence:.1%}")

                    if total_comments:
                        st.bar_chart(sentiment_counts)
                        trend = sentiment_over_time(results, freq="D")
                        if trend.empty:
                            st.info("No published dates are available for a sentiment trend.")
                        else:
                            trend_frame = (
                                trend.rename_axis("date")
                                .reset_index()
                                .melt(id_vars="date", var_name="sentiment", value_name="count")
                            )
                            trend_figure = px.line(
                                trend_frame,
                                x="date",
                                y="count",
                                color="sentiment",
                                markers=True,
                                color_discrete_map=COLORS,
                                title="Comment sentiment over time",
                            )
                            st.plotly_chart(theme.style_fig(trend_figure), width="stretch")

                        keywords = top_keywords(results, n=20)
                        if keywords:
                            keyword_figure = px.bar(
                                pd.DataFrame(keywords),
                                x="count",
                                y="keyword",
                                orientation="h",
                                title="Top comment keywords",
                                color_discrete_sequence=[theme.ACCENT],
                            )
                            keyword_figure.update_layout(yaxis={"autorange": "reversed"})
                            st.plotly_chart(theme.style_fig(keyword_figure), width="stretch")
                        else:
                            st.info("No keywords are available for these comments.")

                        positive_comments, negative_comments = st.columns(2)
                        for column, sentiment, heading in [
                            (positive_comments, "positive", "Most positive comments"),
                            (negative_comments, "negative", "Most negative comments"),
                        ]:
                            with column:
                                st.markdown(f"**{heading}**")
                                top_comments = results_frame[
                                    results_frame["sentiment"] == sentiment
                                ].nlargest(3, "confidence")
                                if top_comments.empty:
                                    st.caption("No comments in this group.")
                                for _, comment in top_comments.iterrows():
                                    st.caption(f"{comment['confidence']:.1%} confidence")
                                    st.write(comment["text"])
                    else:
                        st.info("No comments were returned for this video and order.")

                    display_columns = ["text", "sentiment", "confidence", "published_at"]
                    results_display = results_frame[display_columns].rename(
                        columns={"published_at": "published date"}
                    )
                    st.dataframe(results_display, width="stretch", hide_index=True)
                    st.download_button(
                        "Download CSV",
                        results_display.to_csv(index=False),
                        "youtube_comments.csv",
                        mime="text/csv",
                    )

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
            message = (
                f"{best['model']} is the top available model with macro-F1 {best['f1_macro']:.1%}."
            )
        theme.insight_banner(message)

        metric_columns = st.columns(min(3, len(comparison)))
        for column, (_, row) in zip(metric_columns, comparison.head(3).iterrows()):
            with column:
                theme.kpi_card(
                    f"{row['model']} accuracy",
                    f"{row['accuracy']:.1%}",
                    tone="accent" if row["model"] == best["model"] else "neutral",
                )

        with st.container(border=True):
            scores = comparison["f1_macro"].tolist()
            peak = float(comparison["f1_macro"].max())
            figure = go.Figure(
                go.Bar(
                    x=comparison["f1_macro"],
                    y=comparison["model"],
                    orientation="h",
                    marker_color=[theme.ACCENT if value == peak else "#64748b" for value in scores],
                    text=[f"{value:.1%}" for value in scores],
                    textposition="outside",
                )
            )
            figure.update_layout(
                title=f"{best['model']} has the strongest macro-F1 ({peak:.1%})",
                xaxis_title="Macro-F1",
                yaxis_title=None,
                showlegend=False,
                yaxis={"autorange": "reversed"},
            )
            st.plotly_chart(theme.style_fig(figure), width="stretch")

        display_columns = [
            column
            for column in ["model", "accuracy", "precision", "recall", "f1_macro"]
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

with tab5:
    theme.insight_banner("Explore saved sentiment patterns and frequently used words.")
    insight_data = pd.DataFrame(get_predictions(limit=None))
    if insight_data.empty:
        st.info("No saved predictions yet. Analyze or collect posts to build your insights.")
    else:
        if "source" not in insight_data:
            insight_data["source"] = "manual"
        if "model" not in insight_data:
            insight_data["model"] = ""
        if "label" not in insight_data:
            insight_data["label"] = ""

        filter_columns = st.columns(3)
        with filter_columns[0]:
            selected_source = st.selectbox("Source", ["all", "csv", "upload", "reddit", "youtube"])
        with filter_columns[1]:
            model_options = sorted(
                {str(value) for value in insight_data["model"].dropna() if str(value)}
            )
            selected_model = st.selectbox("Model", ["all", *model_options])
        with filter_columns[2]:
            selected_sentiment = st.selectbox(
                "Sentiment", ["all", "positive", "neutral", "negative"]
            )

        filtered_data = insight_data.copy()
        if selected_source != "all":
            filtered_data = filtered_data[filtered_data["source"] == selected_source]
        if selected_model != "all":
            filtered_data = filtered_data[filtered_data["model"] == selected_model]
        keyword_rows = filtered_data.to_dict("records")
        if selected_sentiment != "all":
            filtered_data = filtered_data[filtered_data["label"] == selected_sentiment]
        rows_for_insights = filtered_data.to_dict("records")

        total_rows = len(filtered_data)
        sentiment_counts = filtered_data["label"].value_counts()
        metric_columns = st.columns(4)
        metric_columns[0].metric("Total rows", f"{total_rows:,}")
        for column, sentiment in zip(metric_columns[1:], ["positive", "neutral", "negative"]):
            percentage = sentiment_counts.get(sentiment, 0) * 100 / total_rows if total_rows else 0
            column.metric(sentiment.title(), f"{percentage:.1f}%")

        if not rows_for_insights:
            st.info("No predictions match these filters. Try a broader selection.")
        else:
            date_grouping = st.selectbox("Time grouping", ["Day", "Week", "Hour"])
            frequency = {"Day": "D", "Week": "W", "Hour": "h"}[date_grouping]
            trend = sentiment_over_time(rows_for_insights, freq=frequency)
            if trend.empty:
                st.info("There are no valid post or prediction dates for this selection.")
            else:
                trend_frame = (
                    trend.rename_axis("date")
                    .reset_index()
                    .melt(id_vars="date", var_name="sentiment", value_name="count")
                )
                trend_figure = px.line(
                    trend_frame,
                    x="date",
                    y="count",
                    color="sentiment",
                    markers=True,
                    color_discrete_map=COLORS,
                    title="Sentiment over time",
                )
                st.plotly_chart(theme.style_fig(trend_figure), width="stretch")

            keyword_filter = st.selectbox(
                "Keyword sentiment", ["all", "positive", "neutral", "negative"]
            )
            keywords = top_keywords(
                keyword_rows,
                n=20,
                label=None if keyword_filter == "all" else keyword_filter,
            )
            if keywords:
                keyword_frame = pd.DataFrame(keywords)
                keyword_figure = px.bar(
                    keyword_frame,
                    x="count",
                    y="keyword",
                    orientation="h",
                    title="Top keywords",
                    color_discrete_sequence=[theme.ACCENT],
                )
                keyword_figure.update_layout(yaxis={"autorange": "reversed"})
                st.plotly_chart(theme.style_fig(keyword_figure), width="stretch")
            else:
                st.info("No keywords are available for this selection.")

with header_slot.container():
    header_left, header_right = st.columns([2, 1])
    with header_left:
        st.title("Twitter Sentiment Analysis")
    with header_right:
        source = escape(st.session_state.get("feedback_source", "CSV upload or YouTube analysis"))
        date_range = escape(st.session_state.get("feedback_date_range", "Awaiting dated data"))
        st.markdown(
            f'<div class="feedback-header-meta">Source: {source}<br>'
            f"Date range: {date_range}<br>Model: {escape(model_name)}</div>",
            unsafe_allow_html=True,
        )

st.markdown(
    '<div class="feedback-footer">Predictions are estimates; review sensitive or low-confidence '
    "cases before responding. LIME highlights are approximate.</div>",
    unsafe_allow_html=True,
)

st.subheader("Prediction history")
delete_message = st.session_state.pop("history_delete_message", None)
if delete_message:
    st.success(delete_message)

rows = get_predictions(limit=None)
if rows:
    history = pd.DataFrame(rows)
    selection = history.copy()
    selection.insert(0, "Select", False)
    editor_version = st.session_state.get("history_editor_version", 0)
    edited_history = st.data_editor(
        selection,
        column_config={
            "Select": st.column_config.CheckboxColumn("Select"),
        },
        disabled=[column for column in selection.columns if column != "Select"],
        width="stretch",
        hide_index=True,
        key=f"history_editor_{editor_version}",
    )

    history_actions = st.columns(3)
    with history_actions[0]:
        if st.button("Delete selected", key="delete_selected_predictions"):
            selected_ids = edited_history.loc[edited_history["Select"].fillna(False), "id"].tolist()
            deleted = delete_predictions(selected_ids)
            st.session_state["history_delete_message"] = f"Deleted {deleted} prediction(s)."
            st.session_state["history_editor_version"] = editor_version + 1
            st.rerun()

    with history_actions[1]:
        confirm_clear = st.checkbox("I understand this cannot be undone")
        if st.button("Clear all history", disabled=not confirm_clear):
            deleted = clear_predictions()
            st.session_state["history_delete_message"] = f"Deleted {deleted} prediction(s)."
            st.session_state["history_editor_version"] = editor_version + 1
            st.rerun()

    with history_actions[2]:
        sources = sorted({str(row["source"]) for row in rows if row.get("source")})
        selected_source = st.selectbox("Clear only this source", ["Choose a source", *sources])
        if st.button("Clear source history", disabled=selected_source == "Choose a source"):
            deleted = clear_predictions(source=selected_source)
            st.session_state["history_delete_message"] = (
                f"Deleted {deleted} prediction(s) from {selected_source}."
            )
            st.session_state["history_editor_version"] = editor_version + 1
            st.rerun()

    st.bar_chart(history["label"].value_counts())
else:
    st.info("No predictions saved yet.")
