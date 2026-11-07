"""Fetch cross-asset market data and compute reactions to FOMC statements.

Two public entry points:

* :func:`build_market_reactions` – for every statement date, compute forward
  returns / changes for each tracked asset and persist the result to
  ``outputs/market_reactions.csv``.
* :func:`get_live_snapshot` – current price + 1-day change for each ticker,
  used by the dashboard's live market panel.
"""

from __future__ import annotations

import datetime as dt
import logging

import numpy as np
import pandas as pd

import config

try:  # yfinance is required for live data but we degrade gracefully in tests
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None  # type: ignore

try:
    from fredapi import Fred
except Exception:  # pragma: no cover
    Fred = None  # type: ignore

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )


# --------------------------------------------------------------------------- #
# Price download helpers
# --------------------------------------------------------------------------- #
def _download_history(start: dt.date, end: dt.date) -> pd.DataFrame:
    """Download adjusted close prices for all tracked tickers between dates.

    Returns a DataFrame indexed by date with one column per *friendly* asset
    label (e.g. ``SPY``), not the raw ticker symbol.
    """
    if yf is None:
        logger.error("yfinance unavailable; cannot download prices.")
        return pd.DataFrame()

    symbol_to_label = {sym: label for label, sym in config.MARKET_TICKERS.items()}
    symbols = list(symbol_to_label)

    try:
        raw = yf.download(
            symbols,
            start=start.isoformat(),
            end=(end + dt.timedelta(days=1)).isoformat(),
            progress=False,
            auto_adjust=True,
            group_by="column",
        )
    except Exception as exc:  # pragma: no cover - network
        logger.error("yfinance download failed: %s", exc)
        return pd.DataFrame()

    if raw.empty:
        return pd.DataFrame()

    # Normalise to a simple close-price frame keyed by friendly label.
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw
    else:  # single ticker case
        close = raw[["Close"]]
        close.columns = symbols

    close = close.rename(columns=symbol_to_label)
    # Keep only known labels, drop fully empty columns.
    keep = [c for c in close.columns if c in config.MARKET_TICKERS]
    return close[keep].dropna(how="all")


def _event_position(series: pd.Series, event_date: pd.Timestamp) -> int | None:
    """Position of the first bar on/after ``event_date`` (the announcement day).

    FOMC statements are released at 2pm ET, before the close, so the market
    reaction starts on the announcement day itself.  The base price is
    therefore the close of the *previous* bar (``position - 1``).
    Returns ``None`` if there is no bar on/after the date or no prior bar.
    """
    idx = series.index
    pos = int(idx.searchsorted(event_date, side="left"))
    if pos >= len(series) or pos == 0:
        return None
    return pos


def _forward_returns(
    series: pd.Series, event_date: pd.Timestamp, horizons: tuple[int, ...]
) -> dict[int, float]:
    """Percent return from the close before the announcement day.

    Horizon ``h`` covers ``h`` trading days starting with the announcement
    day, so ``h=1`` is announcement-day close vs prior close.
    """
    series = series.dropna().sort_index()
    if series.empty:
        return {h: np.nan for h in horizons}
    pos = _event_position(series, event_date)
    if pos is None:
        return {h: np.nan for h in horizons}
    base_price = series.iloc[pos - 1]

    out: dict[int, float] = {}
    for h in horizons:
        target_pos = pos + h - 1
        if target_pos < len(series) and base_price:
            out[h] = float((series.iloc[target_pos] / base_price - 1.0) * 100.0)
        else:
            out[h] = np.nan
    return out


def _level_change_1d(
    series: pd.Series, event_date: pd.Timestamp
) -> float:
    """Announcement-day change in level (VIX points, TNX yield)."""
    series = series.dropna().sort_index()
    pos = _event_position(series, event_date) if not series.empty else None
    if pos is None:
        return np.nan
    return float(series.iloc[pos] - series.iloc[pos - 1])


# --------------------------------------------------------------------------- #
# FRED helpers
# --------------------------------------------------------------------------- #
def _fred_client() -> "Fred | None":
    if Fred is None or not config.FRED_API_KEY:
        return None
    try:
        return Fred(api_key=config.FRED_API_KEY)
    except Exception as exc:  # pragma: no cover
        logger.warning("Could not initialise FRED client: %s", exc)
        return None


def _fred_series(client, series_id: str, start: dt.date, end: dt.date) -> pd.Series:
    if client is None:
        return pd.Series(dtype=float)
    try:
        s = client.get_series(
            series_id,
            observation_start=start.isoformat(),
            observation_end=end.isoformat(),
        )
        s.index = pd.to_datetime(s.index)
        return s.dropna()
    except Exception as exc:  # pragma: no cover - network
        logger.warning("FRED fetch failed for %s: %s", series_id, exc)
        return pd.Series(dtype=float)


def _oas_change_1d(oas: pd.Series, event_date: pd.Timestamp) -> float:
    """Announcement-day change in HY OAS, in basis points (FRED reports percent)."""
    oas = oas.dropna().sort_index()
    pos = _event_position(oas, event_date) if not oas.empty else None
    if pos is None:
        return np.nan
    return float((oas.iloc[pos] - oas.iloc[pos - 1]) * 100.0)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def build_market_reactions(
    statement_dates: list[pd.Timestamp] | pd.Series,
    save: bool = True,
) -> pd.DataFrame:
    """Build the long-form market-reaction table.

    Returns one row per (date, asset) with columns::

        date, asset, return_1d, return_2d, return_5d,
        vix_change_1d, oas_change_1d

    ``vix_change_1d`` and ``oas_change_1d`` are repeated across the asset rows
    of a given date for convenient joins downstream.
    """
    dates = pd.to_datetime(pd.Series(list(statement_dates))).sort_values()
    dates = dates[~dates.duplicated()].reset_index(drop=True)
    if dates.empty:
        logger.warning("No statement dates supplied to build_market_reactions.")
        return _empty_reactions()

    overall_start = (dates.min() - pd.Timedelta(days=config.DAYS_BEFORE + 5)).date()
    overall_end = (
        min(dates.max() + pd.Timedelta(days=config.DAYS_AFTER + 5), pd.Timestamp.today())
    ).date()

    logger.info(
        "Downloading prices %s -> %s for %d statement dates",
        overall_start,
        overall_end,
        len(dates),
    )
    prices = _download_history(overall_start, overall_end)

    fred = _fred_client()
    oas = _fred_series(fred, config.FRED_HY_OAS_SERIES, overall_start, overall_end)

    rows: list[dict] = []
    for event in dates:
        vix_change = (
            _level_change_1d(prices["VIX"], event)
            if "VIX" in prices.columns
            else np.nan
        )
        oas_change = _oas_change_1d(oas, event) if not oas.empty else np.nan

        for asset in config.REACTION_ASSETS:
            if asset not in prices.columns:
                fwd = {h: np.nan for h in config.RETURN_HORIZONS}
            elif asset in ("VIX", "TNX"):
                # For level series, report level change as "return" surrogate.
                fwd = _forward_returns(prices[asset], event, config.RETURN_HORIZONS)
            else:
                fwd = _forward_returns(prices[asset], event, config.RETURN_HORIZONS)

            rows.append(
                {
                    "date": event,
                    "asset": asset,
                    "return_1d": fwd.get(1, np.nan),
                    "return_2d": fwd.get(2, np.nan),
                    "return_5d": fwd.get(5, np.nan),
                    "vix_change_1d": vix_change,
                    "oas_change_1d": oas_change,
                }
            )

    df = pd.DataFrame(rows)
    if save and not df.empty:
        df.to_csv(config.MARKET_REACTIONS_CSV, index=False)
        logger.info("Saved market reactions -> %s", config.MARKET_REACTIONS_CSV)
    return df


def _empty_reactions() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "date",
            "asset",
            "return_1d",
            "return_2d",
            "return_5d",
            "vix_change_1d",
            "oas_change_1d",
        ]
    )


def load_or_build_reactions(
    statement_dates: list[pd.Timestamp] | pd.Series,
) -> pd.DataFrame:
    """Load cached reactions if present and covering all dates, else rebuild."""
    if config.MARKET_REACTIONS_CSV.exists():
        try:
            cached = pd.read_csv(config.MARKET_REACTIONS_CSV, parse_dates=["date"])
            wanted = set(pd.to_datetime(pd.Series(list(statement_dates))).dt.normalize())
            have = set(cached["date"].dt.normalize())
            if wanted and wanted.issubset(have):
                logger.info("Using cached market reactions (%d rows)", len(cached))
                return cached
        except Exception as exc:  # pragma: no cover
            logger.warning("Could not read cached reactions: %s", exc)
    return build_market_reactions(statement_dates)


def get_live_snapshot() -> pd.DataFrame:
    """Return current price + 1-day change for each tracked ticker.

    Columns: ``asset``, ``price``, ``change``, ``pct_change``.
    """
    if yf is None:
        return pd.DataFrame(columns=["asset", "price", "change", "pct_change"])

    end = dt.date.today() + dt.timedelta(days=1)
    start = end - dt.timedelta(days=10)
    prices = _download_history(start, end)
    if prices.empty:
        return pd.DataFrame(columns=["asset", "price", "change", "pct_change"])

    rows: list[dict] = []
    for label in config.MARKET_TICKERS:
        if label not in prices.columns:
            continue
        series = prices[label].dropna()
        if len(series) < 1:
            continue
        last = float(series.iloc[-1])
        prev = float(series.iloc[-2]) if len(series) >= 2 else np.nan
        change = last - prev if not np.isnan(prev) else np.nan
        pct = (change / prev * 100.0) if prev else np.nan
        rows.append(
            {
                "asset": label,
                "price": last,
                "change": change,
                "pct_change": pct,
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":  # manual smoke test
    import data.fetch_fed as fetch_fed

    stmts = fetch_fed.get_statements()
    reactions = build_market_reactions(stmts["date"].tolist())
    print(reactions.head(20).to_string(index=False))
    print("\nLive snapshot:")
    print(get_live_snapshot().to_string(index=False))
