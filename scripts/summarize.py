"""Build sentiment scores and market reactions, then summarise them.

Usage: python scripts/summarize.py
Writes outputs/sentiment_scores.csv, outputs/market_reactions.csv and
outputs/summary.md (the tables quoted in the README).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from data import fetch_fed, fetch_market  # noqa: E402
from nlp import lexicon  # noqa: E402

stmts = fetch_fed.get_statements(refresh=False)
scores = lexicon.score_all(stmts)
react = fetch_market.build_market_reactions(stmts["date"].tolist())

wide = react.pivot_table(index="date", columns="asset", values="return_1d")
first = react.groupby("date")[["vix_change_1d", "oas_change_1d"]].first()
scores = lexicon.add_tone_change(scores)
df = scores.set_index("date").join(wide).join(first)
df["category"] = df["label"].str.split(":").str[0]
from data import opendata  # noqa: E402
_y10 = opendata.fred("DGS10")["DGS10"].dropna() * 100.0  # basis points
df["y10_bp_1d"] = [fetch_market._level_change_1d(_y10, d) for d in df.index]

lines = []
lines.append(f"Statements: {len(scores)} ({scores['date'].min().date()} to {scores['date'].max().date()})")
lines.append("")
lines.append("Label counts: " + ", ".join(f"{k} {v}" for k, v in df["category"].value_counts().items()))
lines.append("")
lines.append("| Label | N | Mean score | SPY 1d % | TLT 1d % | VIX 1d (pts) | 10Y 1d (bp) |")
lines.append("|---|---|---|---|---|---|---|")
order = ["Strongly Dovish", "Moderately Dovish", "Neutral", "Moderately Hawkish", "Strongly Hawkish"]
for lab in [o for o in order if o in set(df["category"])]:
    g = df[df["category"] == lab]
    lines.append(
        f"| {lab} | {len(g)} | {g['sentiment_score'].mean():.1f} | {g['SPY'].mean():+.3f} | "
        f"{g['TLT'].mean():+.3f} | {g['vix_change_1d'].mean():+.2f} | {g['y10_bp_1d'].mean():+.1f} |"
    )
lines.append("")
lines.append("Means of announcement day reactions (close before to close on the statement day). 10Y is the FRED DGS10 change in basis points.")
lines.append("")
lines.append("| Asset | Spearman rho (score vs 1d reaction) | p-value | N |")
lines.append("|---|---|---|---|")
for col, name in [("SPY", "SPY 1d return"), ("TLT", "TLT 1d return"), ("vix_change_1d", "VIX 1d change"), ("y10_bp_1d", "10Y yield 1d change (bp)")]:
    sub = df[["sentiment_score", col]].dropna()
    rho, p = stats.spearmanr(sub["sentiment_score"], sub[col])
    lines.append(f"| {name} | {rho:+.3f} | {p:.3f} | {len(sub)} |")
lines.append("")
lines.append("Tone change (score minus previous statement's score) vs announcement day reaction:")
lines.append("")
lines.append("| Reaction | Spearman rho | p-value | N |")
lines.append("|---|---|---|---|")
for col, name in [("SPY", "US market 1d return"), ("TLT", "20Y Treasury 1d return"), ("vix_change_1d", "VIX 1d change"), ("y10_bp_1d", "10Y yield 1d change (bp)")]:
    sub = df[["tone_change", col]].dropna()
    rho, p = stats.spearmanr(sub["tone_change"], sub[col])
    lines.append(f"| {name} | {rho:+.3f} | {p:.3f} | {len(sub)} |")
q = df.dropna(subset=["tone_change"]).copy()
q["tercile"] = pd.qcut(q["tone_change"], 3, labels=["More dovish", "Little change", "More hawkish"])
lines.append("")
lines.append("| Tone change tercile | N | Mean change | US market 1d % | 20Y Treasury 1d % | VIX 1d (pts) | 10Y 1d (bp) |")
lines.append("|---|---|---|---|---|---|---|")
for lab, g in q.groupby("tercile", observed=True):
    lines.append(f"| {lab} | {len(g)} | {g['tone_change'].mean():+.1f} | {g['SPY'].mean():+.3f} | {g['TLT'].mean():+.3f} | {g['vix_change_1d'].mean():+.2f} | {g['y10_bp_1d'].mean():+.1f} |")
oas = df["oas_change_1d"].dropna()
lines.append("")
lines.append(f"HY OAS 1d change available for {len(oas)} statements (FRED keyless endpoint covers the last three years only).")
out = "\n".join(lines)
(config.OUTPUTS_DIR / "summary.md").write_text(out + "\n")
print(out)
