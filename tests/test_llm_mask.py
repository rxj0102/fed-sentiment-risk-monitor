"""The look-ahead study must not leak dates through the masked text."""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from llm_lookahead import mask  # noqa: E402


def test_mask_removes_years_dates_and_links():
    text = (
        "October 11, 2019\n\nFor release at 11:00 a.m. EDT\n\nConsistent with its January 2019 "
        "Statement, see www.federalreserve.gov/files/fedlistens-report-20200612.pdf . On March 15 the "
        "Committee met; in 2020 rates fell."
    )
    out = mask(text)
    assert not re.search(r"(19|20)\d\d", out)
    assert "October 11" not in out and "March 15" not in out
    assert "www." not in out
    assert "January [year]" in out
