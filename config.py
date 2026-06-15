"""Central configuration for the Fed Sentiment Risk Monitor.

All paths are resolved relative to the project root via ``pathlib`` so the
application behaves identically on a local machine and on Streamlit Cloud.
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT: Path = Path(__file__).resolve().parent

DATA_DIR: Path = PROJECT_ROOT / "data"
STATEMENTS_DIR: Path = DATA_DIR / "statements"
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
FIGURES_DIR: Path = OUTPUTS_DIR / "figures"

MARKET_REACTIONS_CSV: Path = OUTPUTS_DIR / "market_reactions.csv"
SCORES_CSV: Path = OUTPUTS_DIR / "sentiment_scores.csv"

# Make sure the writable directories exist on import.
for _d in (STATEMENTS_DIR, OUTPUTS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Federal Reserve sources
# --------------------------------------------------------------------------- #
FED_CALENDAR_URL: str = (
    "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
)
# Statement permalink template. ``date`` is formatted as YYYYMMDD.
FED_STATEMENT_URL_TEMPLATE: str = (
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary{date}a.htm"
)

# Earliest year we attempt to scrape.
HISTORY_START_YEAR: int = 2010

# Polite scraping controls.
REQUEST_DELAY_SECONDS: float = 1.0
REQUEST_TIMEOUT_SECONDS: int = 20
MAX_RETRIES: int = 4
USER_AGENT: str = (
    "fed-sentiment-risk-monitor/1.0 (+https://github.com/rxj0102/"
    "fed-sentiment-risk-monitor)"
)

# --------------------------------------------------------------------------- #
# Market data
# --------------------------------------------------------------------------- #
# yfinance tickers keyed by a friendly label.
MARKET_TICKERS: dict[str, str] = {
    "VIX": "^VIX",
    "SPY": "SPY",
    "TLT": "TLT",
    "HYG": "HYG",
    "TNX": "^TNX",  # CBOE 10-Year Treasury Note Yield index (yield * 10)
}

# Assets used for forward-return reaction analysis.
REACTION_ASSETS: list[str] = ["VIX", "SPY", "TLT", "HYG", "TNX"]

# FRED series.
FRED_HY_OAS_SERIES: str = "BAMLH0A0HYM2"  # ICE BofA US High Yield OAS
# MOVE-index proxy: 3-month T-bill secondary market rate volatility input.
FRED_TBILL_SERIES: str = "DTB3"

# Lookback window around each statement (trading-day approximation in calendar
# days; downstream code uses positional indexing on the returned price frame).
DAYS_BEFORE: int = 2
DAYS_AFTER: int = 5

# Forward-return horizons (trading days).
RETURN_HORIZONS: tuple[int, ...] = (1, 2, 5)

# --------------------------------------------------------------------------- #
# Sentiment classification thresholds (defaults; overridable in the UI)
# --------------------------------------------------------------------------- #
# Score above HAWKISH_THRESHOLD -> hawkish, below DOVISH_THRESHOLD -> dovish.
DEFAULT_HAWKISH_THRESHOLD: float = 10.0
DEFAULT_DOVISH_THRESHOLD: float = -10.0

# Number of similar statements to surface.
TOP_N_SIMILAR: int = 5
# Number of representative phrases to surface per direction.
TOP_N_PHRASES: int = 5

# --------------------------------------------------------------------------- #
# Caching
# --------------------------------------------------------------------------- #
CACHE_TTL_SECONDS: int = 3600  # refresh once per hour

# --------------------------------------------------------------------------- #
# Credentials
# --------------------------------------------------------------------------- #
# FRED API key is optional. The app degrades gracefully (skips OAS) without it.
FRED_API_KEY: str | None = os.environ.get("FRED_API_KEY")

# --------------------------------------------------------------------------- #
# Fed chairs by tenure (used for hover annotations on the timeline)
# --------------------------------------------------------------------------- #
FED_CHAIRS: list[tuple[str, str, str]] = [
    # (start ISO date, end ISO date, name)
    ("2006-02-01", "2014-01-31", "Ben Bernanke"),
    ("2014-02-01", "2018-02-03", "Janet Yellen"),
    ("2018-02-05", "2099-12-31", "Jerome Powell"),
]


def chair_for_date(date_iso: str) -> str:
    """Return the Fed chair serving on ``date_iso`` (YYYY-MM-DD)."""
    for start, end, name in FED_CHAIRS:
        if start <= date_iso <= end:
            return name
    return "Unknown"
