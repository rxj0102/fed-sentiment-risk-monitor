"""Hawkish / dovish lexicon and a rule-based FOMC statement scorer.

The scorer counts occurrences of curated multi-word and single-word terms in a
statement, then produces a normalised score on ``[-100, +100]`` where positive
is hawkish (tighter policy bias) and negative is dovish (easier policy bias).
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field

import pandas as pd

import config

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

# --------------------------------------------------------------------------- #
# Lexicons (50+ terms each). Multi-word phrases are matched as phrases.
# --------------------------------------------------------------------------- #
HAWKISH_TERMS: list[str] = [
    "inflation", "price stability", "tighten", "tightening", "restrictive",
    "elevated", "persistent", "above target", "rate increase", "rate hike",
    "hike", "raise the target", "raising the target", "balance sheet reduction",
    "quantitative tightening", "upside risk", "upside risks", "overheating",
    "strong labor market", "robust", "vigilant", "determined", "aggressive",
    "firm", "hawkish", "firmer", "higher rates", "higher for longer",
    "inflationary pressures", "price pressures", "elevated inflation",
    "sticky inflation", "wage pressures", "tight labor market",
    "broad-based inflation", "rising prices", "well above", "remains elevated",
    "additional firming", "policy firming", "reduce inflation",
    "bring inflation down", "two percent over the longer run",
    "longer-run goal of 2", "running high", "accelerating", "strong demand",
    "solid pace", "expanded at a strong", "remain attentive",
    "inflation risks", "committed to returning inflation", "sufficiently restrictive",
    "ongoing increases", "supply constraints", "input prices",
]

DOVISH_TERMS: list[str] = [
    "accommodative", "accommodation", "support", "supportive", "easing", "ease",
    "cut", "rate cut", "lower rates", "patient", "patience", "gradual",
    "transitory", "below target", "slack", "uncertainty", "uncertain",
    "downside risk", "downside risks", "fragile", "moderate", "moderated",
    "moderating", "softening", "soften", "pause", "hold", "flexible",
    "stimulative", "dovish", "subdued", "weak", "weakening", "weaker",
    "cooling", "cooled", "disinflation", "declined", "declining", "slowing",
    "slowdown", "stimulus", "monetary stimulus", "asset purchases",
    "quantitative easing", "balance sheet expansion", "well below",
    "muted", "low inflation", "soft labor market", "rising unemployment",
    "elevated unemployment", "labor market weakness", "contracted", "contraction",
    "recession", "headwinds", "spare capacity", "remain patient",
    "monitor", "monitoring", "data dependent", "highly accommodative",
    "extended period", "low for longer", "considerable time",
]

# Pre-compile word-boundary patterns for every term (case-insensitive).
def _compile(terms: list[str]) -> list[tuple[str, re.Pattern]]:
    compiled = []
    for term in terms:
        # \b around alphanumerics; allow internal spaces/hyphens to match flexibly.
        escaped = re.escape(term).replace(r"\ ", r"\s+").replace(r"\-", r"[-\s]")
        compiled.append((term, re.compile(rf"\b{escaped}\b", re.IGNORECASE)))
    return compiled


_HAWKISH_PATTERNS = _compile(HAWKISH_TERMS)
_DOVISH_PATTERNS = _compile(DOVISH_TERMS)

_WORD_RE = re.compile(r"\b\w+\b")


@dataclass
class SentimentResult:
    """Structured output of :func:`score_statement`."""

    sentiment_score: float
    label: str
    word_count: int
    hawkish_count: int
    dovish_count: int
    top_hawkish: list[tuple[str, int]] = field(default_factory=list)
    top_dovish: list[tuple[str, int]] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "sentiment_score": self.sentiment_score,
            "label": self.label,
            "word_count": self.word_count,
            "hawkish_count": self.hawkish_count,
            "dovish_count": self.dovish_count,
            "top_hawkish": self.top_hawkish,
            "top_dovish": self.top_dovish,
        }


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def _count_hits(text: str, patterns: list[tuple[str, re.Pattern]]) -> Counter:
    hits: Counter = Counter()
    for term, pattern in patterns:
        n = len(pattern.findall(text))
        if n:
            hits[term] = n
    return hits


def classify(
    score: float,
    hawkish_threshold: float | None = None,
    dovish_threshold: float | None = None,
) -> str:
    """Map a numeric score to a human-readable hawk/dove label."""
    hawk = (
        config.DEFAULT_HAWKISH_THRESHOLD
        if hawkish_threshold is None
        else hawkish_threshold
    )
    dove = (
        config.DEFAULT_DOVISH_THRESHOLD
        if dovish_threshold is None
        else dovish_threshold
    )

    if score >= hawk * 3:
        return f"Strongly Hawkish: {score:+.0f}"
    if score >= hawk:
        return f"Moderately Hawkish: {score:+.0f}"
    if score <= dove * 3:
        return f"Strongly Dovish: {score:+.0f}"
    if score <= dove:
        return f"Moderately Dovish: {score:+.0f}"
    return f"Neutral: {score:+.0f}"


def score_statement(
    text: str,
    hawkish_threshold: float | None = None,
    dovish_threshold: float | None = None,
) -> SentimentResult:
    """Score a single statement.

    Net score = (hawkish_hits - dovish_hits) / total_words * 100, then scaled
    and clamped to ``[-100, +100]``.
    """
    if not text or not text.strip():
        return SentimentResult(0.0, classify(0.0), 0, 0, 0, [], [])

    word_count = len(_WORD_RE.findall(text))
    hawkish_hits = _count_hits(text, _HAWKISH_PATTERNS)
    dovish_hits = _count_hits(text, _DOVISH_PATTERNS)

    hawkish_total = sum(hawkish_hits.values())
    dovish_total = sum(dovish_hits.values())

    if word_count == 0:
        raw = 0.0
    else:
        # Per-1000-words net tilt; multiply to spread the realistic range.
        raw = (hawkish_total - dovish_total) / word_count * 1000.0 * 2.5

    score = float(max(-100.0, min(100.0, raw)))

    top_hawkish = hawkish_hits.most_common(config.TOP_N_PHRASES)
    top_dovish = dovish_hits.most_common(config.TOP_N_PHRASES)

    return SentimentResult(
        sentiment_score=round(score, 1),
        label=classify(score, hawkish_threshold, dovish_threshold),
        word_count=word_count,
        hawkish_count=hawkish_total,
        dovish_count=dovish_total,
        top_hawkish=top_hawkish,
        top_dovish=top_dovish,
    )


def score_all(
    statements: pd.DataFrame,
    hawkish_threshold: float | None = None,
    dovish_threshold: float | None = None,
    save: bool = True,
) -> pd.DataFrame:
    """Score every statement in ``statements`` (date, statement_text, ...).

    Returns a DataFrame::

        date, sentiment_score, label, hawkish_count, dovish_count,
        word_count, top_hawkish, top_dovish
    """
    if statements.empty:
        return pd.DataFrame(
            columns=[
                "date", "sentiment_score", "label", "hawkish_count",
                "dovish_count", "word_count", "top_hawkish", "top_dovish",
            ]
        )

    rows: list[dict] = []
    for _, row in statements.iterrows():
        res = score_statement(
            row["statement_text"], hawkish_threshold, dovish_threshold
        )
        rows.append(
            {
                "date": row["date"],
                "sentiment_score": res.sentiment_score,
                "label": res.label,
                "hawkish_count": res.hawkish_count,
                "dovish_count": res.dovish_count,
                "word_count": res.word_count,
                "top_hawkish": ", ".join(t for t, _ in res.top_hawkish),
                "top_dovish": ", ".join(t for t, _ in res.top_dovish),
            }
        )

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    if save and not df.empty:
        df.to_csv(config.SCORES_CSV, index=False)
        logger.info("Saved sentiment scores -> %s", config.SCORES_CSV)
    return df


# --------------------------------------------------------------------------- #
# Highlighting helper (used by the Streamlit statement viewer)
# --------------------------------------------------------------------------- #
def highlight_terms(
    text: str,
    hawkish_color: str = "#E63946",
    dovish_color: str = "#4D96FF",
) -> str:
    """Return ``text`` as HTML with hawkish/dovish terms wrapped in colored spans.

    Hawkish terms are highlighted in red, dovish terms in blue. Matching is done
    on the longest terms first to avoid partial overlaps.
    """
    if not text:
        return ""

    import html as _html

    # Build a combined, length-sorted list so longer phrases win.
    spans: list[tuple[int, int, str]] = []
    for color, patterns in (
        (hawkish_color, _HAWKISH_PATTERNS),
        (dovish_color, _DOVISH_PATTERNS),
    ):
        for _term, pattern in patterns:
            for m in pattern.finditer(text):
                spans.append((m.start(), m.end(), color))

    if not spans:
        return _html.escape(text).replace("\n", "<br>")

    # Resolve overlaps: keep the longest span starting at each position.
    spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))
    chosen: list[tuple[int, int, str]] = []
    last_end = -1
    for start, end, color in spans:
        if start >= last_end:
            chosen.append((start, end, color))
            last_end = end

    out: list[str] = []
    cursor = 0
    for start, end, color in chosen:
        out.append(_html.escape(text[cursor:start]))
        snippet = _html.escape(text[start:end])
        out.append(
            f'<span style="background-color:{color};color:#0D1117;'
            f'padding:0 2px;border-radius:3px;font-weight:600;">{snippet}</span>'
        )
        cursor = end
    out.append(_html.escape(text[cursor:]))
    return "".join(out).replace("\n", "<br>")


if __name__ == "__main__":  # manual smoke test
    sample = (
        "The Committee judges that inflation remains elevated and persistent. "
        "Risks are weighted to the upside. The Committee remains determined and "
        "vigilant, and is prepared for additional firming. The labor market is "
        "strong and robust."
    )
    result = score_statement(sample)
    print(result.as_dict())
