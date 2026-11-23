"""Look-ahead bias of a language model on FOMC statements.

A model that has read about past FOMC meetings during training can appear to
"understand" a statement because it remembers what followed. Glasserman and Lin
(2023) and Sarkar and Vafa (2024) show this in news sentiment. The clean test is to
compare the model on statements it could have seen (before its release date) with
statements it cannot have seen (after).

Model: Qwen/Qwen2.5-1.5B-Instruct, pinned revision, released 2024-09-19, run
locally on CPU. Statements released after 2024-09-19 cannot be in its training data.
A model this small gives nearly the same top answer to every statement (a first run
with greedy decoding answered 2023 and rise almost every time), so every probe reads
the model's probabilities over the allowed answers instead of its top answer.

Three probes, each asked of every statement:

1. date   : the statement with every year and its release date masked; which year?
2. score  : hawkishness on a 1 to 7 scale, asked with the date masked and with the
            date shown; Spearman correlation with the 10 year yield change that day.
3. recall : did the 10 year Treasury yield rise or fall on the day of the statement?
            asked with the date masked and with the date shown. A model with no
            memory of outcomes should be right about half the time.

    python scripts/llm_lookahead.py run       # about 40 minutes on 2 CPUs; writes outputs/llm_probs.csv
    python scripts/llm_lookahead.py analyze   # writes outputs/llm_lookahead.md and .json

The probabilities file is committed, so the analysis reruns without the model.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import fetch_fed, fetch_market, opendata  # noqa: E402

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
RELEASE = pd.Timestamp("2024-09-19")
PROBS = ROOT / "outputs" / "llm_probs.csv"
YEARS = list(range(2011, 2027))
OUT_MD = ROOT / "outputs" / "llm_lookahead.md"
OUT_JSON = ROOT / "outputs" / "llm_lookahead.json"
MAX_CHARS = 5000

MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"


def mask(text: str) -> str:
    """Remove the release date line, links, long digit strings, every year and month-day date."""
    lines = text.split("\n")
    if lines and re.match(rf"^\s*({MONTHS}) \d{{1,2}}, \d{{4}}\s*$", lines[0]):
        lines = lines[1:]
    t = "\n".join(lines)
    t = re.sub(rf"\b({MONTHS}) \d{{1,2}}\b(, \d{{4}})?", "[date]", t)
    t = re.sub(r"\S*(www\.|https?://)\S*", "[link]", t)
    t = re.sub(r"\S*\d{6,}\S*", "[number]", t)
    t = re.sub(r"(19|20)\d{2}", "[year]", t)
    return t.strip()[:MAX_CHARS]


def dated(text: str, date: pd.Timestamp) -> str:
    return f"Released on {date:%B %d, %Y}.\n\n" + mask(text)


PROMPTS = {
    "date": "Here is a Federal Reserve FOMC statement. Years and dates have been removed.\n\n{text}\n\n"
    "In which year was this statement released? Answer with a four digit year only.",
    "score": "Here is a Federal Reserve FOMC statement.\n\n{text}\n\nRate how hawkish it is on a scale "
    "from 1 (very dovish) to 7 (very hawkish). Answer with one digit only.",
    "recall": "Here is a Federal Reserve FOMC statement.\n\n{text}\n\nOn the day this statement was "
    "released, did the 10-year US Treasury yield rise or fall from the previous close? Answer with "
    "one word: rise or fall.",
}


def run() -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, dtype=torch.bfloat16)
    model.eval()
    digit = {str(d): tok.encode(str(d))[0] for d in range(10)}
    rise, fall = tok.encode("rise")[0], tok.encode("fall")[0]

    def prompt_ids(probe: str, text: str) -> list[int]:
        msgs = [{"role": "user", "content": PROMPTS[probe].format(text=text)}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
        ids = ids["input_ids"] if hasattr(ids, "keys") else ids
        return ids[0].tolist()

    @torch.no_grad()
    def next_logprobs(ids: list[int], past=None):
        out = model(torch.tensor([ids]), past_key_values=past, use_cache=True)
        return torch.log_softmax(out.logits[0, -1].float(), -1), out.past_key_values

    stmts = fetch_fed.get_statements(refresh=False).sort_values("date").reset_index(drop=True)
    rows = pd.read_csv(PROBS).to_dict("records") if PROBS.exists() else []
    seen = {(r["date"], r["view"]) for r in rows}
    for _, s in stmts.iterrows():
        d = pd.Timestamp(s["date"])
        for view in ("masked", "dated"):
            if (str(d.date()), view) in seen:
                continue
            text = mask(s["statement_text"]) if view == "masked" else dated(s["statement_text"], d)
            row = {"date": str(d.date()), "view": view}
            # score: probabilities of the digits 1 to 7
            lp, _ = next_logprobs(prompt_ids("score", text))
            p = torch.softmax(torch.stack([lp[digit[str(k)]] for k in range(1, 8)]), 0)
            row["score_expected"] = float((p * torch.arange(1, 8)).sum())
            # recall: P(rise) against P(fall)
            lp, _ = next_logprobs(prompt_ids("recall", text))
            row["recall_logit_rise"] = float(lp[rise] - lp[fall])
            # date: probability of each year 2011 to 2026, digit by digit (masked view only)
            if view == "masked":
                base = prompt_ids("date", text)
                lp0, past = next_logprobs(base)
                lp1, past = next_logprobs([digit["2"]], past)
                lp2, past = next_logprobs([digit["0"]], past)
                head = float(lp0[digit["2"]] + lp1[digit["0"]])
                logs = {}
                for third in ("1", "2"):
                    import copy

                    lp3, _ = next_logprobs([digit[third]], copy.deepcopy(past))
                    for y in YEARS:
                        ys = str(y)
                        if ys[2] == third:
                            logs[y] = head + float(lp2[digit[third]] + lp3[digit[ys[3]]])
                v = np.array([logs[y] for y in YEARS])
                pr = np.exp(v - v.max())
                pr /= pr.sum()
                for y, q in zip(YEARS, pr):
                    row[f"p_{y}"] = float(q)
            rows.append(row)
            print(row["date"], view, round(row["score_expected"], 2), round(row["recall_logit_rise"], 2), flush=True)
        pd.DataFrame(rows).to_csv(PROBS, index=False)


def auc(score: np.ndarray, label: np.ndarray) -> float:
    """Probability that a random rise day gets a higher score than a random fall day."""
    pos, neg = score[label == 1], score[label == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    r = stats.rankdata(np.concatenate([pos, neg]))
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def boot_ci(f, *arrays, n=2000, seed=0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    m = len(arrays[0])
    vals = []
    for _ in range(n):
        i = rng.integers(0, m, m)
        v = f(*(a[i] for a in arrays))
        if np.isfinite(v):
            vals.append(v)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (np.nan, np.nan)


def analyze() -> None:
    pr = pd.read_csv(PROBS, parse_dates=["date"])
    y10 = opendata.fred("DGS10")["DGS10"].dropna() * 100.0
    react = pd.Series({d: fetch_market._level_change_1d(y10, d) for d in pr["date"].unique()})
    pr["y10_bp"] = pr["date"].map(react)
    pr["after"] = pr["date"] > RELEASE
    res: dict = {"model": MODEL, "revision": REVISION, "release": str(RELEASE.date())}
    m = pr[pr["view"] == "masked"].copy()
    P = m[[f"p_{y}" for y in YEARS]].to_numpy()
    yr = m["date"].dt.year.to_numpy()
    m["p_true"] = P[np.arange(len(m)), yr - YEARS[0]]
    m["rank_true"] = (P > m["p_true"].to_numpy()[:, None]).sum(1) + 1
    m["exp_year"] = P @ np.array(YEARS)
    for name, flag in (("before", False), ("after", True)):
        r: dict = {}
        g = m[m["after"] == flag]
        r["n"], r["first"], r["last"] = int(len(g)), str(g["date"].min().date()), str(g["date"].max().date())
        r["p_true_mean"] = float(g["p_true"].mean())
        r["top1"] = float((g["rank_true"] == 1).mean())
        r["rank_mean"] = float(g["rank_true"].mean())
        r["exp_year_mae"] = float((g["exp_year"] - g["date"].dt.year).abs().mean())
        for view in ("masked", "dated"):
            v = pr[(pr["view"] == view) & (pr["after"] == flag)].dropna(subset=["y10_bp"])
            v = v[v["y10_bp"] != 0]
            lab = (v["y10_bp"] > 0).astype(int).to_numpy()
            sc = v["recall_logit_rise"].to_numpy()
            r[f"recall_{view}_auc"] = auc(sc, lab)
            r[f"recall_{view}_ci"] = boot_ci(auc, sc, lab)
            r[f"recall_{view}_n"] = int(len(v))
            rho, p = stats.spearmanr(v["score_expected"], v["y10_bp"])
            r[f"score_{view}_rho"], r[f"score_{view}_p"] = float(rho), float(p)
            r[f"score_{view}_ci"] = boot_ci(
                lambda a, b: stats.spearmanr(a, b)[0], v["score_expected"].to_numpy(), v["y10_bp"].to_numpy()
            )
        res[name] = r
    # uniform guessing over 16 years would put 1/16 on the true year
    res["uniform_p_true"] = 1 / len(YEARS)
    res["p_true_by_year"] = {int(k): round(float(v), 4) for k, v in m.groupby(m["date"].dt.year)["p_true"].mean().items()}
    OUT_JSON.write_text(json.dumps(res, indent=1, default=float) + "\n")
    OUT_MD.write_text(render(res))
    print(OUT_MD.read_text())


def render(r: dict) -> str:
    b, a = r["before"], r["after"]
    ci = lambda c: f"{c[0]:.2f} to {c[1]:.2f}"
    rows = [
        ("Statements", f"{b['n']} ({b['first'][:4]} to {b['last'][:7]})", f"{a['n']} ({a['first'][:7]} to {a['last'][:7]})"),
        ("Probability on the true year, dates masked (uniform guess 0.06)", f"{b['p_true_mean']:.2f}", f"{a['p_true_mean']:.2f}"),
        ("True year ranked first of 16", f"{100 * b['top1']:.0f}%", f"{100 * a['top1']:.0f}%"),
        ("Mean rank of the true year", f"{b['rank_mean']:.1f}", f"{a['rank_mean']:.1f}"),
        ("Yield direction, AUC of P(rise), dates masked", f"{b['recall_masked_auc']:.2f} ({ci(b['recall_masked_ci'])})", f"{a['recall_masked_auc']:.2f} ({ci(a['recall_masked_ci'])})"),
        ("Yield direction, AUC of P(rise), date shown", f"{b['recall_dated_auc']:.2f} ({ci(b['recall_dated_ci'])})", f"{a['recall_dated_auc']:.2f} ({ci(a['recall_dated_ci'])})"),
        ("Hawkishness vs 10Y change, Spearman, dates masked", f"{b['score_masked_rho']:+.2f} ({ci(b['score_masked_ci'])})", f"{a['score_masked_rho']:+.2f} ({ci(a['score_masked_ci'])})"),
        ("Hawkishness vs 10Y change, Spearman, date shown", f"{b['score_dated_rho']:+.2f} ({ci(b['score_dated_ci'])})", f"{a['score_dated_rho']:+.2f} ({ci(a['score_dated_ci'])})"),
    ]
    lines = [
        "# Look-ahead bias of a language model on FOMC statements",
        "",
        f"Written by `scripts/llm_lookahead.py analyze` from `outputs/llm_probs.csv`. Model: "
        f"{r['model']} (revision {r['revision'][:12]}), released {r['release']}, run on CPU. "
        "Statements after the release date cannot be in its training data. Each probe reads the "
        "model's probabilities over the allowed answers. AUC is the chance that a day the 10 year "
        "yield rose gets a higher P(rise) than a day it fell (0.5 is no knowledge). Brackets are "
        "95% bootstrap intervals.",
        "",
        "| | Before release | After release |",
        "|---|---|---|",
    ] + [f"| {x} | {y} | {z} |" for x, y, z in rows]
    lines += ["", "Mean probability on the true year by statement year, dates masked:", "", "| Year | P(true year) |", "|---|---|"]
    lines += [f"| {y} | {p:.2f} |" for y, p in r["p_true_by_year"].items()]
    py = r["p_true_by_year"]
    last_known = max(y for y, p in py.items() if p > r["uniform_p_true"])
    sharp = [y for y, p in py.items() if p >= 0.3]
    lines += [
        "",
        "## Reading",
        "",
        f"The model knows when statements were written. With every date removed it puts "
        f"{b['p_true_mean']:.2f} on the true year for statements before its release, "
        f"{b['p_true_mean'] / r['uniform_p_true']:.1f} times a uniform guess, and names it first "
        f"{100 * b['top1']:.0f}% of the time. Its knowledge is sharpest for {min(sharp)} to "
        f"{max(sharp)} and stops after {last_known}: statements from later years, including those "
        "released before the model itself, get almost no probability on their true year. So the "
        "practical cutoff is the end of the training data, not the release date.",
        "",
        "It does not remember what markets did. P(rise) for the 10 year yield on statement day is "
        f"no better than chance before release (AUC {b['recall_masked_auc']:.2f} with dates masked, "
        f"{b['recall_dated_auc']:.2f} with the date shown) or after, and showing the date does not "
        "help. Its hawkishness score is at most weakly related to the yield change, and no more so "
        "with the date shown.",
        "",
        "For this model, then, look-ahead bias would come from knowing the economic history around a "
        "statement, not from recalling the day's market move. That is the risk Glasserman and Lin "
        "(2023) and Sarkar and Vafa (2024) describe, and it is why any backtest of a language model "
        "signal should start after the model's training data ends.",
        "",
        "## What this does not show",
        "",
        f"One small model (1.5 billion parameters). Larger models memorize far more, so the null on "
        f"recall here says little about them. Only {a['n']} statements come after the release, so "
        "every after-release interval is wide. Committee member names and rate levels stay in the "
        "masked text, and they are part of what lets the model date a statement.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    {"run": run, "analyze": analyze}[sys.argv[1]]()
