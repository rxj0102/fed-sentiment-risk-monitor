# Fed Sentiment Risk Monitor

[![CI](https://github.com/rxj0102/fed-sentiment-risk-monitor/actions/workflows/ci.yml/badge.svg)](https://github.com/rxj0102/fed-sentiment-risk-monitor/actions/workflows/ci.yml)

> Live FOMC statement hawkishness scoring + cross-asset risk reactions, in a single Streamlit page.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://YOUR-APP-URL.streamlit.app)

<!-- demo.gif — add after first deploy -->
![demo](demo.gif)

---

## How it works

- **Scores the latest Fed statement** for hawkishness/dovishness using a curated
  lexicon, producing a single number from −100 (most dovish) to +100 (most hawkish).
- **Finds the most similar past statements** by language (TF-IDF + cosine similarity)
  and shows how markets actually reacted after those historically similar meetings.
- **Pulls live cross-asset data** (VIX, SPY, TLT, HYG, 10Y yield, HY credit spreads)
  so you can compare today's tone against the market backdrop.

## Methodology

| Component | Approach |
|-----------|----------|
| **Sentiment** | Rule-based lexicon of 50+ hawkish and 50+ dovish terms. Net score = (hawkish − dovish) / total words, scaled and clamped to [−100, +100]. |
| **Similarity** | All statements (2010–present) are TF-IDF vectorized (uni- + bi-grams). A new statement is compared via cosine similarity; the top 5 matches are surfaced. |
| **Market reactions** | For each statement date, 1/2/5-day forward returns are computed for each asset from yfinance, plus 1-day VIX point change and 1-day HY OAS change from FRED. |

The "expected reaction" panel simply averages the realized market moves that
followed the most linguistically similar past statements — a nearest-neighbor,
not a forecast.

## Corrections

- **Reaction window.** FOMC statements are released at 2pm ET, before the close. Reactions were previously measured from the announcement-day close, which excluded the announcement-day move itself. They are now measured from the prior close: the 1-day figure is announcement-day close vs prior close, and the 2-day and 5-day figures cover 2 and 5 trading days starting with the announcement day. VIX and OAS 1-day changes follow the same convention. Delete `outputs/market_reactions.csv` to rebuild the cache.

## Data sources

- **Federal Reserve** — FOMC statement text, scraped from
  [federalreserve.gov](https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm)
  and cached locally.
- **yfinance** — daily prices for VIX, SPY, TLT, HYG, and the 10Y yield (^TNX).
- **FRED** — ICE BofA US High Yield OAS (`BAMLH0A0HYM2`). Requires a free
  [FRED API key](https://fred.stlouisfed.org/docs/api/api_key.html) (optional —
  the app degrades gracefully without it).

## Local setup

```bash
git clone https://github.com/rxj0102/fed-sentiment-risk-monitor.git
cd fed-sentiment-risk-monitor
pip install -r requirements.txt

# Optional: enable HY credit-spread data
export FRED_API_KEY="your_key_here"   # or put it in a .env file

streamlit run app.py
```

The first run scrapes and caches historical statements (a minute or two);
subsequent runs are near-instant thanks to on-disk caching and `@st.cache_data`.

### Optional `.env`

```dotenv
FRED_API_KEY=your_fred_api_key
```

## Deploy to Streamlit Cloud

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io).
3. Connect the repo and set the main file to **`app.py`**.
4. (Optional) Add `FRED_API_KEY` under **App → Settings → Secrets**.
5. Deploy — a live URL is generated instantly. Replace the badge URL at the top
   of this README with it.

## Project structure

```
fed-sentiment-risk-monitor/
├── app.py                  # single Streamlit entry point
├── config.py               # paths, tickers, URLs, thresholds
├── data/
│   ├── fetch_fed.py        # FOMC statement scraper + cache
│   ├── fetch_market.py     # yfinance + FRED reactions
│   └── statements/         # cached statements (.txt, gitignored)
├── nlp/
│   ├── lexicon.py          # hawk/dove word lists + scorer
│   └── similarity.py       # TF-IDF cosine similarity
├── outputs/figures/
├── .streamlit/config.toml  # dark terminal theme
├── requirements.txt
└── README.md
```

## Reproducibility

### Data availability

All inputs come from free, public sources. No paid data or API keys are required unless noted.

- Yahoo Finance via the yfinance package (free, no key)
- FRED, Federal Reserve Bank of St. Louis (free)
- Federal Reserve Board publications (free)
- Files shipped in the `data/` folder of this repository

### Computational requirements

- Python 3.12.3. Every dependency version is pinned in `uv.lock`.
- Checked on Linux x86_64 with 2 CPU cores and 8 GB RAM. No GPU is needed.

```bash
pip install uv     # once
uv sync            # installs the exact pinned environment
uv run pytest -q   # runs the test suite
```

### Runtime

The test suite (6 tests) finishes in about 10 seconds on the machine above, including any data downloads it triggers.

### Citation

Citation metadata is in `CITATION.cff`. GitHub turns it into a Cite this repository button on the repo page.


## Disclaimer

For research and educational purposes only. Nothing here is investment advice.

## License

[MIT](LICENSE) © 2026 rxj0102
