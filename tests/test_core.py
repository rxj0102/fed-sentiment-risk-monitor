import numpy as np
import pandas as pd
import pytest

from data import fetch_market as fm
from nlp import lexicon


def _series():
    idx = pd.bdate_range("2024-01-01", periods=10)
    return pd.Series([100, 100, 100, 110, 110, 121, 121, 121, 121, 121.0], index=idx)


def test_reaction_includes_announcement_day():
    s = _series()
    event = s.index[3]  # jump happens on the announcement day itself
    out = fm._forward_returns(s, event, (1, 2, 5))
    assert out[1] == pytest.approx(10.0)
    assert out[2] == pytest.approx(10.0)
    assert out[5] == pytest.approx(21.0)


def test_level_change_is_announcement_day():
    s = _series()
    assert fm._level_change_1d(s, s.index[3]) == pytest.approx(10.0)


def test_oas_change_bps():
    s = pd.Series([3.0, 3.0, 3.5], index=pd.bdate_range("2024-01-01", periods=3))
    assert fm._oas_change_1d(s, s.index[2]) == pytest.approx(50.0)


def test_event_without_prior_bar_is_nan():
    s = _series()
    out = fm._forward_returns(s, s.index[0], (1,))
    assert np.isnan(out[1])


def test_empty_text_is_neutral():
    assert lexicon.score_statement("").sentiment_score == 0.0


def test_hawkish_beats_dovish_and_is_clamped():
    h = lexicon.HAWKISH_TERMS[0]
    d = lexicon.DOVISH_TERMS[0]
    assert lexicon.score_statement(f"{h} {h} policy").sentiment_score > 0
    assert lexicon.score_statement(f"{d} {d} policy").sentiment_score < 0
    assert -100 <= lexicon.score_statement(" ".join([h] * 50)).sentiment_score <= 100
