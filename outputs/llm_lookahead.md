# Look-ahead bias of a language model on FOMC statements

Written by `scripts/llm_lookahead.py analyze` from `outputs/llm_probs.csv`. Model: Qwen/Qwen2.5-1.5B-Instruct (revision 989aa7980e4c), released 2024-09-19, run on CPU. Statements after the release date cannot be in its training data. Each probe reads the model's probabilities over the allowed answers. AUC is the chance that a day the 10 year yield rose gets a higher P(rise) than a day it fell (0.5 is no knowledge). Brackets are 95% bootstrap intervals.

| | Before release | After release |
|---|---|---|
| Statements | 115 (2011 to 2024-09) | 18 (2024-11 to 2026-10) |
| Probability on the true year, dates masked (uniform guess 0.06) | 0.22 | 0.00 |
| True year ranked first of 16 | 26% | 0% |
| Mean rank of the true year | 4.5 | 15.3 |
| Yield direction, AUC of P(rise), dates masked | 0.51 (0.40 to 0.62) | 0.52 (0.24 to 0.81) |
| Yield direction, AUC of P(rise), date shown | 0.50 (0.39 to 0.62) | 0.51 (0.21 to 0.80) |
| Hawkishness vs 10Y change, Spearman, dates masked | -0.18 (-0.38 to 0.01) | +0.05 (-0.48 to 0.59) |
| Hawkishness vs 10Y change, Spearman, date shown | -0.14 (-0.34 to 0.05) | +0.03 (-0.50 to 0.54) |

Mean probability on the true year by statement year, dates masked:

| Year | P(true year) |
|---|---|
| 2011 | 0.02 |
| 2012 | 0.03 |
| 2013 | 0.19 |
| 2014 | 0.18 |
| 2015 | 0.10 |
| 2016 | 0.06 |
| 2017 | 0.10 |
| 2018 | 0.04 |
| 2019 | 0.06 |
| 2020 | 0.53 |
| 2021 | 0.38 |
| 2022 | 0.64 |
| 2023 | 0.46 |
| 2024 | 0.00 |
| 2025 | 0.00 |
| 2026 | 0.00 |

## Reading

The model knows when statements were written. With every date removed it puts 0.22 on the true year for statements before its release, 3.4 times a uniform guess, and names it first 26% of the time. Its knowledge is sharpest for 2020 to 2023 and stops after 2023: statements from later years, including those released before the model itself, get almost no probability on their true year. So the practical cutoff is the end of the training data, not the release date.

It does not remember what markets did. P(rise) for the 10 year yield on statement day is no better than chance before release (AUC 0.51 with dates masked, 0.50 with the date shown) or after, and showing the date does not help. Its hawkishness score is at most weakly related to the yield change, and no more so with the date shown.

For this model, then, look-ahead bias would come from knowing the economic history around a statement, not from recalling the day's market move. That is the risk Glasserman and Lin (2023) and Sarkar and Vafa (2024) describe, and it is why any backtest of a language model signal should start after the model's training data ends.

## What this does not show

One small model (1.5 billion parameters). Larger models memorize far more, so the null on recall here says little about them. Only 18 statements come after the release, so every after-release interval is wide. Committee member names and rate levels stay in the masked text, and they are part of what lets the model date a statement.
