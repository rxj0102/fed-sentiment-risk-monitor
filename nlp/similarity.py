"""TF-IDF based similarity between FOMC statements.

Builds a TF-IDF model over all historical statements and, given a new statement,
returns the most similar prior statements together with their sentiment scores
and market reactions.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import config

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )


class StatementSimilarity:
    """Fit a TF-IDF model and query it for similar historical statements."""

    def __init__(self, statements: pd.DataFrame):
        """``statements`` must have columns ``date`` and ``statement_text``."""
        self.statements = statements.sort_values("date").reset_index(drop=True)
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            sublinear_tf=True,
        )
        if self.statements.empty:
            self.matrix = None
            logger.warning("No statements provided to StatementSimilarity.")
        else:
            corpus = self.statements["statement_text"].fillna("").tolist()
            self.matrix = self.vectorizer.fit_transform(corpus)
            logger.info(
                "Fitted TF-IDF on %d statements (vocab=%d)",
                self.matrix.shape[0],
                self.matrix.shape[1],
            )

    def most_similar(
        self,
        text: str,
        top_n: int = config.TOP_N_SIMILAR,
        exclude_date: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Return the ``top_n`` most similar historical statements to ``text``.

        Output columns: ``date``, ``similarity`` (0-1), ``rank``.
        """
        empty = pd.DataFrame(columns=["date", "similarity", "rank"])
        if self.matrix is None or not text.strip():
            return empty

        query_vec = self.vectorizer.transform([text])
        sims = cosine_similarity(query_vec, self.matrix).ravel()

        order = np.argsort(sims)[::-1]
        results: list[dict] = []
        for idx in order:
            row_date = self.statements.iloc[idx]["date"]
            if exclude_date is not None and pd.Timestamp(row_date).normalize() == pd.Timestamp(
                exclude_date
            ).normalize():
                continue
            results.append(
                {"date": row_date, "similarity": float(sims[idx])}
            )
            if len(results) >= top_n:
                break

        df = pd.DataFrame(results)
        if not df.empty:
            df["rank"] = range(1, len(df) + 1)
        return df


def similar_with_reactions(
    text: str,
    sim_model: StatementSimilarity,
    scores: pd.DataFrame,
    reactions: pd.DataFrame,
    top_n: int = config.TOP_N_SIMILAR,
    exclude_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Join similar statements with sentiment scores and market reactions.

    Returns columns::

        date, similarity, sentiment_score, spy_5d, vix_change_1d,
        tlt_5d, oas_change_1d
    """
    sim = sim_model.most_similar(text, top_n=top_n, exclude_date=exclude_date)
    if sim.empty:
        return pd.DataFrame(
            columns=[
                "date", "similarity", "sentiment_score", "spy_5d",
                "vix_change_1d", "tlt_5d", "oas_change_1d",
            ]
        )

    sim = sim.copy()
    sim["_key"] = pd.to_datetime(sim["date"]).dt.normalize()

    # Sentiment scores keyed by normalised date.
    score_map = {}
    if not scores.empty:
        s = scores.copy()
        s["_key"] = pd.to_datetime(s["date"]).dt.normalize()
        score_map = dict(zip(s["_key"], s["sentiment_score"]))

    def _reaction(date_key, asset, col):
        if reactions.empty:
            return np.nan
        mask = (
            pd.to_datetime(reactions["date"]).dt.normalize() == date_key
        ) & (reactions["asset"] == asset)
        sub = reactions.loc[mask, col]
        return float(sub.iloc[0]) if not sub.empty else np.nan

    rows: list[dict] = []
    for _, r in sim.iterrows():
        key = r["_key"]
        rows.append(
            {
                "date": r["date"],
                "similarity": r["similarity"],
                "sentiment_score": score_map.get(key, np.nan),
                "spy_5d": _reaction(key, "SPY", "return_5d"),
                "vix_change_1d": _reaction(key, "VIX", "vix_change_1d"),
                "tlt_5d": _reaction(key, "TLT", "return_5d"),
                "oas_change_1d": _reaction(key, "SPY", "oas_change_1d"),
            }
        )
    return pd.DataFrame(rows)


def average_reaction(similar_df: pd.DataFrame) -> dict[str, float]:
    """Average market reaction across the similar-statement table."""
    if similar_df.empty:
        return {"spy_5d": np.nan, "vix_change_1d": np.nan, "tlt_5d": np.nan, "oas_change_1d": np.nan}
    return {
        "spy_5d": float(similar_df["spy_5d"].mean(skipna=True)),
        "vix_change_1d": float(similar_df["vix_change_1d"].mean(skipna=True)),
        "tlt_5d": float(similar_df["tlt_5d"].mean(skipna=True)),
        "oas_change_1d": float(similar_df["oas_change_1d"].mean(skipna=True)),
    }


if __name__ == "__main__":  # manual smoke test
    import data.fetch_fed as fetch_fed
    import nlp.lexicon as lexicon

    stmts = fetch_fed.get_statements()
    model = StatementSimilarity(stmts)
    scores = lexicon.score_all(stmts)
    latest = fetch_fed.get_latest_statement(stmts)
    if latest:
        _, text, _ = latest
        out = model.most_similar(text, exclude_date=latest[0])
        print(out.to_string(index=False))
