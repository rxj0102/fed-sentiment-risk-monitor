"""Fed Sentiment Risk Monitor — single-page Streamlit dashboard.

Run with::

    streamlit run app.py

On first load the app scrapes & caches historical FOMC statements, builds the
market-reaction table, scores every statement with the hawk/dove lexicon, fits a
TF-IDF similarity model, and renders the dashboard with the latest statement
pre-loaded. All heavy steps are cached with ``@st.cache_data`` (TTL 1 hour).
"""

from __future__ import annotations

import datetime as dt
import logging

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import config
from data import fetch_fed, fetch_market
from nlp import lexicon, similarity

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("app")

st.set_page_config(
    page_title="Fed Sentiment Risk Monitor",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------------- #
# Cached data pipeline
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=config.CACHE_TTL_SECONDS, show_spinner="Loading FOMC statements…")
def load_statements(refresh: bool = True) -> pd.DataFrame:
    """Scrape + cache all historical statements (falls back to local cache)."""
    try:
        return fetch_fed.get_statements(refresh=refresh)
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Statement load failed, using cache only: %s", exc)
        return fetch_fed.get_statements(refresh=False)


@st.cache_data(ttl=config.CACHE_TTL_SECONDS, show_spinner="Scoring statements…")
def load_scores(statements: pd.DataFrame) -> pd.DataFrame:
    return lexicon.score_all(statements)


@st.cache_data(ttl=config.CACHE_TTL_SECONDS, show_spinner="Building market reactions…")
def load_reactions(dates: list) -> pd.DataFrame:
    try:
        return fetch_market.load_or_build_reactions(dates)
    except Exception as exc:  # pragma: no cover
        logger.error("Market reaction build failed: %s", exc)
        return fetch_market._empty_reactions()


@st.cache_resource(show_spinner="Fitting similarity model…")
def load_similarity_model(
    _statements: pd.DataFrame, cache_key: str
) -> similarity.StatementSimilarity:
    """Fit the TF-IDF model. ``cache_key`` (hashable) controls cache invalidation;
    ``_statements`` is excluded from hashing via its leading underscore."""
    return similarity.StatementSimilarity(_statements)


@st.cache_data(ttl=config.CACHE_TTL_SECONDS, show_spinner="Fetching live prices…")
def load_live_snapshot() -> pd.DataFrame:
    try:
        return fetch_market.get_live_snapshot()
    except Exception as exc:  # pragma: no cover
        logger.error("Live snapshot failed: %s", exc)
        return pd.DataFrame(columns=["asset", "price", "change", "pct_change"])


# --------------------------------------------------------------------------- #
# Chart builders
# --------------------------------------------------------------------------- #
def gauge_chart(score: float) -> go.Figure:
    """Plotly gauge from -100 (dovish) to +100 (hawkish)."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": "", "font": {"size": 48, "color": "#C9D1D9"}},
            gauge={
                "axis": {
                    "range": [-100, 100],
                    "tickwidth": 1,
                    "tickcolor": "#C9D1D9",
                    "tickvals": [-100, -50, 0, 50, 100],
                },
                "bar": {"color": "rgba(255,255,255,0.25)"},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [-100, -60], "color": "#1D4ED8"},
                    {"range": [-60, -20], "color": "#4D96FF"},
                    {"range": [-20, 20], "color": "#6B7280"},
                    {"range": [20, 60], "color": "#F87171"},
                    {"range": [60, 100], "color": "#E63946"},
                ],
                "threshold": {
                    "line": {"color": "#FFFFFF", "width": 4},
                    "thickness": 0.8,
                    "value": score,
                },
            },
        )
    )
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#C9D1D9", "family": "monospace"},
    )
    return fig


def timeline_chart(
    scores: pd.DataFrame,
    reactions: pd.DataFrame,
    current_date: pd.Timestamp,
    show_vix: bool,
    date_range: tuple[dt.date, dt.date],
) -> go.Figure:
    """Sentiment score over time with optional VIX overlay."""
    df = scores.copy()
    df["date"] = pd.to_datetime(df["date"])
    lo, hi = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
    df = df[(df["date"] >= lo) & (df["date"] <= hi)]

    df["chair"] = df["date"].dt.strftime("%Y-%m-%d").map(config.chair_for_date)

    # VIX level per statement date from the reactions frame (level change base).
    vix_levels = {}
    if not reactions.empty and "VIX" in reactions["asset"].unique():
        vix_rows = reactions[reactions["asset"] == "VIX"]
        # vix_change_1d is a change; approximate level via cumulative not available,
        # so we instead pull live-ish proxy: use the absolute return_1d sign only.
        # For a meaningful overlay we read VIX close from a fresh light download.
        vix_levels = _vix_levels_for_dates(df["date"].tolist())

    secondary = show_vix and bool(vix_levels)
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    customdata = np.stack(
        [
            df["chair"].values,
            [vix_levels.get(pd.Timestamp(d).normalize(), np.nan) for d in df["date"]],
        ],
        axis=-1,
    )

    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["sentiment_score"],
            mode="lines+markers",
            name="Sentiment",
            line=dict(color="#E63946", width=2),
            marker=dict(size=5, color="#E63946"),
            customdata=customdata,
            hovertemplate=(
                "<b>%{x|%Y-%m-%d}</b><br>"
                "Score: %{y:+.1f}<br>"
                "Chair: %{customdata[0]}<br>"
                "VIX: %{customdata[1]:.1f}<extra></extra>"
            ),
        ),
        secondary_y=False,
    )

    # Mark the current statement with a star.
    cur = df[df["date"].dt.normalize() == pd.Timestamp(current_date).normalize()]
    if not cur.empty:
        fig.add_trace(
            go.Scatter(
                x=cur["date"],
                y=cur["sentiment_score"],
                mode="markers",
                name="Current",
                marker=dict(size=18, color="#FFD60A", symbol="star",
                            line=dict(color="#FFFFFF", width=1)),
                hovertemplate="<b>Current statement</b><br>%{x|%Y-%m-%d}<br>"
                "Score: %{y:+.1f}<extra></extra>",
            ),
            secondary_y=False,
        )

    if secondary:
        vix_y = [vix_levels.get(pd.Timestamp(d).normalize(), np.nan) for d in df["date"]]
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=vix_y,
                mode="lines",
                name="VIX",
                line=dict(color="#4D96FF", width=1.5, dash="dot"),
                hovertemplate="VIX: %{y:.1f}<extra></extra>",
            ),
            secondary_y=True,
        )
        fig.update_yaxes(title_text="VIX", secondary_y=True, color="#4D96FF")

    fig.add_hline(y=0, line=dict(color="#6B7280", width=1, dash="dash"))
    fig.update_yaxes(title_text="Sentiment (-100 dovish → +100 hawkish)",
                     secondary_y=False, range=[-100, 100])
    fig.update_layout(
        height=420,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#C9D1D9", "family": "monospace"},
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor="#21262D")
    fig.update_yaxes(gridcolor="#21262D")
    return fig


@st.cache_data(ttl=config.CACHE_TTL_SECONDS, show_spinner=False)
def _vix_levels_for_dates(dates: list) -> dict:
    """Best-effort VIX close on each statement date for the timeline overlay."""
    try:
        import yfinance as yf
    except Exception:
        return {}
    if not dates:
        return {}
    start = (min(pd.to_datetime(dates)) - pd.Timedelta(days=5)).date()
    end = (max(pd.to_datetime(dates)) + pd.Timedelta(days=2)).date()
    try:
        data = yf.download(
            "^VIX", start=start.isoformat(), end=end.isoformat(),
            progress=False, auto_adjust=True,
        )
    except Exception:
        return {}
    if data.empty:
        return {}
    close = data["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.dropna()
    out: dict = {}
    for d in pd.to_datetime(dates):
        on_or_before = close[close.index <= d]
        if not on_or_before.empty:
            out[pd.Timestamp(d).normalize()] = float(on_or_before.iloc[-1])
    return out


# --------------------------------------------------------------------------- #
# Rendering sections
# --------------------------------------------------------------------------- #
def render_header(statement_date: pd.Timestamp) -> None:
    st.markdown(
        "<h1 style='color:#E63946;margin-bottom:0;'>Fed Sentiment Risk Monitor</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='color:#8B949E;font-size:1.05rem;margin-top:0.2rem;'>"
        "Live FOMC statement hawkishness scoring + cross-asset risk reactions</p>",
        unsafe_allow_html=True,
    )
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.caption(
        f"Last updated: {now}  •  Statement date: "
        f"{pd.Timestamp(statement_date).strftime('%Y-%m-%d')}"
    )


def render_sentiment(result: lexicon.SentimentResult, statement_text: str) -> None:
    st.subheader("Sentiment Score")
    st.plotly_chart(gauge_chart(result.sentiment_score), use_container_width=True)

    color = (
        "#E63946" if result.sentiment_score > 0
        else "#4D96FF" if result.sentiment_score < 0
        else "#6B7280"
    )
    st.markdown(
        f"<h2 style='text-align:center;color:{color};'>{result.label}</h2>",
        unsafe_allow_html=True,
    )
    st.caption(
        f"Words: {result.word_count}  •  Hawkish hits: {result.hawkish_count}  "
        f"•  Dovish hits: {result.dovish_count}"
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Top hawkish phrases**")
        if result.top_hawkish:
            for term, n in result.top_hawkish:
                st.markdown(
                    f"<span style='color:#E63946;'>▲ {term}</span> "
                    f"<span style='color:#8B949E;'>×{n}</span>",
                    unsafe_allow_html=True,
                )
        else:
            st.caption("None found")
    with c2:
        st.markdown("**Top dovish phrases**")
        if result.top_dovish:
            for term, n in result.top_dovish:
                st.markdown(
                    f"<span style='color:#4D96FF;'>▼ {term}</span> "
                    f"<span style='color:#8B949E;'>×{n}</span>",
                    unsafe_allow_html=True,
                )
        else:
            st.caption("None found")


def render_statement_text(statement_text: str) -> None:
    st.subheader("Statement Text")
    st.caption("Hawkish terms in red, dovish terms in blue.")
    html = lexicon.highlight_terms(statement_text)
    st.markdown(
        f"<div style='height:340px;overflow-y:auto;padding:1rem;"
        f"background-color:#161B22;border:1px solid #21262D;border-radius:8px;"
        f"line-height:1.6;'>{html}</div>",
        unsafe_allow_html=True,
    )


def render_timeline(
    scores: pd.DataFrame,
    reactions: pd.DataFrame,
    current_date: pd.Timestamp,
    date_range: tuple[dt.date, dt.date],
) -> None:
    st.subheader("Historical Sentiment Timeline")
    show_vix = st.checkbox("Overlay VIX (secondary axis)", value=False)
    fig = timeline_chart(scores, reactions, current_date, show_vix, date_range)
    st.plotly_chart(fig, use_container_width=True)


def render_similar(similar_df: pd.DataFrame) -> None:
    st.subheader("Most similar past statements (by language)")
    if similar_df.empty:
        st.info("Not enough historical data to compute similarity.")
        return

    display = similar_df.copy()
    display["date"] = pd.to_datetime(display["date"]).dt.strftime("%Y-%m-%d")
    display["similarity %"] = (display["similarity"] * 100).round(1)
    display = display.rename(
        columns={
            "sentiment_score": "sentiment",
            "spy_5d": "SPY 5d %",
            "vix_change_1d": "VIX 1d Δ",
            "oas_change_1d": "HY OAS 1d Δ (bps)",
        }
    )
    cols = ["date", "similarity %", "sentiment", "SPY 5d %", "VIX 1d Δ", "HY OAS 1d Δ (bps)"]
    st.dataframe(
        display[cols].style.format(
            {
                "sentiment": "{:+.1f}",
                "SPY 5d %": "{:+.2f}",
                "VIX 1d Δ": "{:+.2f}",
                "HY OAS 1d Δ (bps)": "{:+.1f}",
            },
            na_rep="—",
        ),
        use_container_width=True,
        hide_index=True,
    )

    avg = similarity.average_reaction(similar_df)
    st.markdown("**Average market reaction after similar statements:**")
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("SPY 5d return", _fmt(avg["spy_5d"], "%", signed=True))
    a2.metric("VIX 1d change", _fmt(avg["vix_change_1d"], " pts", signed=True))
    a3.metric("TLT 5d return", _fmt(avg["tlt_5d"], "%", signed=True))
    a4.metric("HY OAS 1d change", _fmt(avg["oas_change_1d"], " bps", signed=True))


def render_market_snapshot(snapshot: pd.DataFrame) -> None:
    st.subheader("Current Market Snapshot")
    if snapshot.empty:
        st.info("Live market data unavailable.")
        return
    snap = snapshot.set_index("asset")
    cols = st.columns(5)
    order = ["VIX", "SPY", "TLT", "HYG", "TNX"]
    labels = {
        "VIX": "VIX", "SPY": "SPY", "TLT": "TLT", "HYG": "HYG",
        "TNX": "10Y Yield",
    }
    for col, asset in zip(cols, order):
        if asset not in snap.index:
            col.metric(labels[asset], "—")
            continue
        row = snap.loc[asset]
        price = row["price"]
        if asset == "TNX":
            # ^TNX is quoted as yield*10 (e.g. 42.0 -> 4.20%).
            price_disp = f"{price / 10:.2f}%"
            delta_disp = f"{row['change'] / 10:+.2f} pts" if pd.notna(row["change"]) else None
        else:
            price_disp = f"{price:,.2f}"
            delta_disp = (
                f"{row['pct_change']:+.2f}%" if pd.notna(row["pct_change"]) else None
            )
        col.metric(labels[asset], price_disp, delta_disp)


def _fmt(value: float, suffix: str = "", signed: bool = False) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    fmt = "{:+.2f}" if signed else "{:.2f}"
    return fmt.format(value) + suffix


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def render_sidebar(
    scores: pd.DataFrame,
    sim_model: similarity.StatementSimilarity,
    reactions: pd.DataFrame,
) -> dict:
    st.sidebar.header("Controls")

    if st.sidebar.button("🔄 Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.subheader("Classification thresholds")
    hawk = st.sidebar.slider(
        "Hawkish cutoff", 0.0, 60.0, float(config.DEFAULT_HAWKISH_THRESHOLD), 1.0
    )
    dove = st.sidebar.slider(
        "Dovish cutoff", -60.0, 0.0, float(config.DEFAULT_DOVISH_THRESHOLD), 1.0
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Timeline date range")
    if not scores.empty:
        min_d = pd.to_datetime(scores["date"]).min().date()
        max_d = pd.to_datetime(scores["date"]).max().date()
    else:
        min_d = dt.date(config.HISTORY_START_YEAR, 1, 1)
        max_d = dt.date.today()
    date_range = st.sidebar.slider(
        "Range",
        min_value=min_d,
        max_value=max_d,
        value=(min_d, max_d),
        format="YYYY-MM",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Analyze a custom statement")
    custom_text = st.sidebar.text_area(
        "Paste any Fed statement or text",
        height=200,
        placeholder="Paste text here to score it and find similar historical "
        "statements…",
    )

    custom_result = None
    custom_similar = None
    if custom_text.strip():
        custom_result = lexicon.score_statement(custom_text, hawk, dove)
        st.sidebar.metric("Custom score", f"{custom_result.sentiment_score:+.1f}")
        st.sidebar.caption(custom_result.label)
        custom_similar = similarity.similar_with_reactions(
            custom_text, sim_model, scores, reactions
        )
        if not custom_similar.empty:
            st.sidebar.caption("Most similar historical statement:")
            top = custom_similar.iloc[0]
            st.sidebar.write(
                f"{pd.Timestamp(top['date']).strftime('%Y-%m-%d')} "
                f"({top['similarity'] * 100:.0f}% match)"
            )

    return {
        "hawkish_threshold": hawk,
        "dovish_threshold": dove,
        "date_range": date_range,
        "custom_text": custom_text,
        "custom_result": custom_result,
        "custom_similar": custom_similar,
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    statements = load_statements(refresh=True)

    if statements.empty:
        st.error(
            "No FOMC statements available. The Fed website may be unreachable and "
            "no cached statements were found. Try again later."
        )
        st.stop()

    scores = load_scores(statements)
    reactions = load_reactions(statements["date"].tolist())
    sim_cache_key = (
        f"{len(statements)}-"
        f"{pd.to_datetime(statements['date']).max().strftime('%Y%m%d')}"
    )
    sim_model = load_similarity_model(statements, sim_cache_key)
    snapshot = load_live_snapshot()

    latest = fetch_fed.get_latest_statement(statements)
    latest_date, latest_text, _latest_url = latest

    controls = render_sidebar(scores, sim_model, reactions)

    # Re-score the latest statement with the (possibly adjusted) thresholds.
    latest_result = lexicon.score_statement(
        latest_text, controls["hawkish_threshold"], controls["dovish_threshold"]
    )

    render_header(latest_date)
    st.markdown("---")
    render_sentiment(latest_result, latest_text)
    st.markdown("---")
    render_statement_text(latest_text)
    st.markdown("---")
    render_timeline(scores, reactions, latest_date, controls["date_range"])
    st.markdown("---")

    similar_df = similarity.similar_with_reactions(
        latest_text, sim_model, scores, reactions, exclude_date=latest_date
    )
    render_similar(similar_df)
    st.markdown("---")
    render_market_snapshot(snapshot)

    # If a custom statement was entered, surface its results in the main pane too.
    if controls["custom_result"] is not None:
        st.markdown("---")
        st.subheader("Custom statement analysis")
        render_sentiment(controls["custom_result"], controls["custom_text"])
        if controls["custom_similar"] is not None:
            render_similar(controls["custom_similar"])

    st.markdown("---")
    st.caption(
        "Data: Federal Reserve (statements), yfinance (prices), FRED (HY OAS). "
        "For research/educational use only — not investment advice."
    )


if __name__ == "__main__":
    main()
