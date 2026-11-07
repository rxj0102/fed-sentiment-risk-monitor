"""Scrape and cache Federal Reserve FOMC statements.

The public entry points are:

* :func:`get_statements` – build (and cache) a DataFrame of every FOMC
  statement from :data:`config.HISTORY_START_YEAR` to the present.
* :func:`get_latest_statement` – convenience wrapper returning the most
  recent statement as ``(date, text, url)``.

Network access is best-effort: when the Fed website is unreachable the
functions transparently fall back to whatever has been cached under
``data/statements/``.
"""

from __future__ import annotations

import datetime as dt
import logging
import re
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

import config

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

# Matches the YYYYMMDD token inside a monetary statement permalink.
_STATEMENT_DATE_RE = re.compile(r"monetary(\d{8})a?\.htm", re.IGNORECASE)


# --------------------------------------------------------------------------- #
# Low-level HTTP helper
# --------------------------------------------------------------------------- #
def _http_get(url: str) -> str | None:
    """GET ``url`` with retries + exponential backoff. Returns text or None."""
    headers = {"User-Agent": config.USER_AGENT}
    backoff = 2.0
    for attempt in range(1, config.MAX_RETRIES + 1):
        try:
            resp = requests.get(
                url, headers=headers, timeout=config.REQUEST_TIMEOUT_SECONDS
            )
            if resp.status_code == 200:
                return resp.text
            if resp.status_code == 404:
                logger.warning("404 Not Found: %s", url)
                return None
            logger.warning(
                "HTTP %s for %s (attempt %d/%d)",
                resp.status_code,
                url,
                attempt,
                config.MAX_RETRIES,
            )
        except requests.RequestException as exc:  # pragma: no cover - network
            logger.warning(
                "Request error for %s (attempt %d/%d): %s",
                url,
                attempt,
                config.MAX_RETRIES,
                exc,
            )
        if attempt < config.MAX_RETRIES:
            time.sleep(backoff)
            backoff *= 2
    logger.error("Giving up on %s after %d attempts", url, config.MAX_RETRIES)
    return None


# --------------------------------------------------------------------------- #
# Calendar scraping
# --------------------------------------------------------------------------- #
def scrape_statement_dates() -> list[dt.date]:
    """Return candidate FOMC statement dates discovered on the Fed website.

    We combine two strategies for robustness:

    1. Parse the public FOMC calendar page for direct statement links.
    2. Fall back to crawling the per-year press-release archive pages, which
       reliably list every ``monetaryYYYYMMDDa.htm`` permalink.
    """
    dates: set[dt.date] = set()

    # Strategy 1: the calendar landing page.
    html = _http_get(config.FED_CALENDAR_URL)
    if html:
        dates.update(_extract_dates_from_html(html))

    # Strategy 2: per-year archives (covers older years not on the landing page).
    current_year = dt.date.today().year
    for year in range(config.HISTORY_START_YEAR, current_year + 1):
        archive_url = (
            "https://www.federalreserve.gov/newsevents/pressreleases/"
            f"{year}-monetary.htm"
        )
        page = _http_get(archive_url)
        if page:
            dates.update(_extract_dates_from_html(page))
        time.sleep(config.REQUEST_DELAY_SECONDS)

    filtered = sorted(
        d for d in dates if d.year >= config.HISTORY_START_YEAR and d <= dt.date.today()
    )
    logger.info("Discovered %d candidate statement dates", len(filtered))
    return filtered


def _extract_dates_from_html(html: str) -> set[dt.date]:
    """Pull every ``monetaryYYYYMMDD`` date referenced in ``html``."""
    found: set[dt.date] = set()
    soup = BeautifulSoup(html, "lxml")
    for anchor in soup.find_all("a", href=True):
        match = _STATEMENT_DATE_RE.search(anchor["href"])
        if match:
            try:
                found.add(dt.datetime.strptime(match.group(1), "%Y%m%d").date())
            except ValueError:
                continue
    # Also scan raw text in case links are rendered via JS placeholders.
    for match in _STATEMENT_DATE_RE.finditer(html):
        try:
            found.add(dt.datetime.strptime(match.group(1), "%Y%m%d").date())
        except ValueError:
            continue
    return found


# --------------------------------------------------------------------------- #
# Statement fetching / caching
# --------------------------------------------------------------------------- #
def _cache_path(date: dt.date) -> Path:
    return config.STATEMENTS_DIR / f"{date.strftime('%Y%m%d')}.txt"


def _extract_statement_text(html: str) -> str:
    """Extract the readable body of a Fed press-release page."""
    soup = BeautifulSoup(html, "lxml")

    # The article body lives in a div with class 'col-xs-12.col-sm-8.col-md-8'
    # on modern pages, but markup has shifted over the years, so we try a few
    # selectors and fall back to the whole document.
    candidates = [
        soup.find("div", id="article"),
        soup.find("div", class_="col-xs-12 col-sm-8 col-md-8"),
        soup.find("div", class_="col-md-8"),
    ]
    container = next((c for c in candidates if c is not None), soup)

    paragraphs = [p.get_text(" ", strip=True) for p in container.find_all("p")]
    text = "\n\n".join(p for p in paragraphs if p)

    if not text:  # last resort
        text = container.get_text(" ", strip=True)

    # Trim common boilerplate footers.
    for marker in ("Last Update:", "Implementation Note", "Voting for the"):
        idx = text.find(marker)
        if idx > 200:  # keep the marker if it appears too early (false positive)
            text = text[:idx].strip()
    return text.strip()


def fetch_statement(date: dt.date, use_cache: bool = True) -> str | None:
    """Fetch a single statement's text, using the on-disk cache when possible."""
    cache_file = _cache_path(date)
    if use_cache and cache_file.exists():
        text = cache_file.read_text(encoding="utf-8").strip()
        if text:
            logger.debug("Cache hit for %s", date)
            return text

    url = config.FED_STATEMENT_URL_TEMPLATE.format(date=date.strftime("%Y%m%d"))
    logger.info("Fetching statement %s -> %s", date, url)
    html = _http_get(url)
    if html is None:
        return None

    text = _extract_statement_text(html)
    if not text:
        logger.warning("Empty statement text parsed for %s", date)
        return None

    cache_file.write_text(text, encoding="utf-8")
    logger.info("Cached statement %s (%d chars)", date, len(text))
    return text


def _statement_url(date: dt.date) -> str:
    return config.FED_STATEMENT_URL_TEMPLATE.format(date=date.strftime("%Y%m%d"))


def _load_cached_dates() -> list[dt.date]:
    """Return all statement dates currently available on disk."""
    dates: list[dt.date] = []
    for path in sorted(config.STATEMENTS_DIR.glob("*.txt")):
        try:
            dates.append(dt.datetime.strptime(path.stem, "%Y%m%d").date())
        except ValueError:
            continue
    return dates


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def get_statements(refresh: bool = True) -> pd.DataFrame:
    """Build a DataFrame of all FOMC statements.

    Columns: ``date`` (datetime64), ``statement_text`` (str),
    ``statement_url`` (str). Sorted ascending by date.

    When ``refresh`` is False, or when the network is unavailable, the function
    relies entirely on the local cache.
    """
    dates: list[dt.date] = []
    if refresh:
        try:
            dates = scrape_statement_dates()
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Calendar scrape failed: %s", exc)

    # Always union with whatever is already cached.
    cached = set(_load_cached_dates())
    all_dates = sorted(set(dates) | cached)

    if not all_dates:
        logger.error("No statement dates available (network + cache empty).")
        return pd.DataFrame(columns=["date", "statement_text", "statement_url"])

    rows: list[dict] = []
    for date in all_dates:
        text = fetch_statement(date, use_cache=True)
        if text:
            rows.append(
                {
                    "date": pd.Timestamp(date),
                    "statement_text": text,
                    "statement_url": _statement_url(date),
                }
            )
        # Be polite only when we actually hit the network (cache misses).
        if not _cache_path(date).exists():
            time.sleep(config.REQUEST_DELAY_SECONDS)

    df = (
        pd.DataFrame(rows)
        .sort_values("date")
        .reset_index(drop=True)
    )
    logger.info("Assembled %d statements", len(df))
    return df


def get_latest_statement(
    statements: pd.DataFrame | None = None,
) -> tuple[pd.Timestamp, str, str] | None:
    """Return ``(date, text, url)`` for the most recent statement, or None."""
    if statements is None:
        statements = get_statements()
    if statements.empty:
        return None
    latest = statements.sort_values("date").iloc[-1]
    return latest["date"], latest["statement_text"], latest["statement_url"]


if __name__ == "__main__":  # manual smoke test
    df = get_statements()
    print(df.tail()[["date", "statement_url"]].to_string(index=False))
    latest = get_latest_statement(df)
    if latest:
        d, text, _ = latest
        print(f"\nLatest statement {d.date()} ({len(text)} chars):\n")
        print(text[:500], "...")
