"""Shared theme and small presentation helpers for the dashboard."""
from html import escape

import plotly.graph_objects as go
import streamlit as st


ACCENT = "#2563eb"
NEGATIVE = "#dc2626"
POSITIVE = "#16a34a"
NEUTRAL = "#64748b"
MUTED = "#94a3b8"

_TONES = {
    "accent": (ACCENT, "rgba(37, 99, 235, 0.12)"),
    "negative": (NEGATIVE, "rgba(220, 38, 38, 0.12)"),
    "positive": (POSITIVE, "rgba(22, 163, 74, 0.12)"),
    "neutral": (NEUTRAL, "rgba(100, 116, 139, 0.12)"),
    "critical": (NEGATIVE, "rgba(220, 38, 38, 0.12)"),
    "high": ("#c2410c", "rgba(234, 88, 12, 0.12)"),
    "medium": ("#a16207", "rgba(202, 138, 4, 0.12)"),
    "low": (NEUTRAL, "rgba(100, 116, 139, 0.12)"),
}


def apply_theme():
    """Load typography and shared spacing, cards, and control styles."""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    :root { --feedback-accent: #2563eb; }
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; letter-spacing: 0; }
    [data-testid="stAppViewContainer"] { background: transparent; }
    [data-testid="stMainBlockContainer"] { max-width: 1440px; padding-top: 1.5rem; }
    [data-testid="stHeader"] { background: transparent; }
    #MainMenu, footer { visibility: hidden; }
    h1 { font-size: 28px !important; font-weight: 700 !important; letter-spacing: 0 !important; }
    h2, h3 { font-size: 18px !important; font-weight: 600 !important; letter-spacing: 0 !important; }
    [data-testid="stCaptionContainer"] { color: rgba(128, 128, 128, 0.92); font-size: 12px; }
    [data-testid="stVerticalBlockBorderWrapper"] {
      border-radius: 12px; box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
      background: rgba(128, 128, 128, 0.035);
    }
    [data-testid="stMetric"] { border-radius: 12px; padding: 16px 18px; }
    [data-testid="stMetricLabel"] p { color: rgba(128, 128, 128, 0.9); font-size: 12px; }
    [data-testid="stMetricValue"] { font-size: 32px; font-weight: 700; }
    [data-testid="stTabs"] [role="tablist"] { gap: 0.35rem; }
    [data-testid="stTabs"] button[role="tab"] { border-radius: 8px 8px 0 0; }
    [data-testid="stPlotlyChart"] { border-radius: 12px; overflow: hidden; }
    .feedback-pill, .feedback-chip {
      display: inline-flex; align-items: center; width: fit-content; border-radius: 999px;
      padding: 4px 9px; font-size: 12px; font-weight: 600;
    }
    .feedback-chip { font-weight: 500; }
    .feedback-banner {
      border-radius: 12px; border: 1px solid rgba(128, 128, 128, 0.2);
      border-left: 4px solid var(--banner-color); background: rgba(128, 128, 128, 0.06);
      padding: 16px 18px; margin: 0.4rem 0 1rem; color: inherit;
    }
    .feedback-banner p { margin: 0; }
    .feedback-header-meta { color: rgba(128, 128, 128, 0.9); font-size: 12px; text-align: right; }
    .feedback-footer { color: rgba(128, 128, 128, 0.85); font-size: 12px; }
    </style>
    """, unsafe_allow_html=True)


def kpi_card(label, value, delta=None, chart_data=None, tone="accent", delta_color=None):
    """Render a compact bordered metric card."""
    color, _ = _TONES.get(tone, _TONES["accent"])
    if delta_color is None:
        delta_color = "inverse" if tone == "negative" else "normal"
    st.metric(
        label, value, delta=delta, delta_color=delta_color, border=True,
        chart_data=chart_data, chart_type="line",
    )
    st.markdown(
        f'<div style="height:3px;background:{color};border-radius:3px;margin-top:-8px"></div>',
        unsafe_allow_html=True,
    )


def pill(label, tone="neutral"):
    """Return a compact colored status pill."""
    color, background = _TONES.get(tone.lower(), _TONES["neutral"])
    return (
        f'<span class="feedback-pill" style="color:{color};background:{background};'
        f'border:1px solid {color}33">{escape(str(label))}</span>'
    )


def chip(label):
    """Return a muted tag for small pieces of context."""
    color, background = _TONES["neutral"]
    return (
        f'<span class="feedback-chip" style="color:{color};background:{background};'
        f'border:1px solid {color}33">{escape(str(label))}</span>'
    )


def insight_banner(text, tone="accent"):
    """Render a concise insight in a left-accented banner."""
    color, _ = _TONES.get(tone, _TONES["accent"])
    st.markdown(
        f'<div class="feedback-banner" style="--banner-color:{color}">'
        f'<p>{escape(str(text))}</p></div>',
        unsafe_allow_html=True,
    )


def style_fig(figure):
    """Apply quiet axes and transparent surfaces to a Plotly figure."""
    text_color = st.get_option("theme.textColor") or "#475569"
    figure.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, sans-serif", "size": 12, "color": text_color},
        title_font={"size": 15, "color": text_color},
        margin={"l": 12, "r": 12, "t": 48, "b": 12},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02,
                "xanchor": "right", "x": 1},
    )
    figure.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.16)", zeroline=False)
    figure.update_yaxes(showgrid=False, zeroline=False)
    return figure


def issue_bar(issues):
    """Build a ranked issue-share bar chart with one highlighted leader."""
    if issues.empty:
        figure = go.Figure()
        figure.update_layout(title="No complaint issues in this selection")
        return style_fig(figure)

    colors = [ACCENT] + ["#64748b"] * (len(issues) - 1)
    leader = issues.iloc[0]
    figure = go.Figure(go.Bar(
        x=issues["share_pct"], y=issues["issue"], orientation="h",
        marker_color=colors, text=[f"{share:.1f}%" for share in issues["share_pct"]],
        textposition="outside", hovertemplate="%{y}<br>%{x:.1f}% of negative tweets<extra></extra>",
    ))
    figure.update_layout(
        title=f"{leader['issue']} is the leading complaint ({leader['share_pct']:.1f}%)",
        showlegend=False, yaxis={"autorange": "reversed"},
        xaxis_title="Share of negative tweets (%)", yaxis_title=None,
    )
    return style_fig(figure)