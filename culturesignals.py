#!/usr/bin/env python3
#culture_signals.py
#Goal:Quantifyemotion+cognitionsignalsintextandtrackthemovertime.
#Usecases:historicalcorpora,fiction,news,archives,culturaldatasets.

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


#-----------------------------#
#Textprocessing
#-----------------------------#

_TOKEN_RE = re.compile(r"[a-zA-Z]+(?:'[a-z]+)?")

def tokenize(text: str) -> List[str]:
    #Simple,fasttokenizerforEnglish-liketext.
    #Keepsapostrophesinsidewords(e.g.,don't).
    return _TOKEN_RE.findall(str(text).lower())


#-----------------------------#
#Lexiconhandling
#-----------------------------#

@dataclass(frozen=True)
class Lexicon:
    #word->listofcategories
    word_to_cats: Dict[str, List[str]]

    @staticmethod
    def built_in_minimal() -> "Lexicon":
        #Smallbuilt-inlexiconforarepo-friendlydemo.
        #Forrealresearch,pass--lexiconwithalargercuratedresource.
        pos = {
            "love","happy","joy","delight","pleased","smile","hope","calm","relief","peace",
            "wonderful","excellent","good","great","beautiful","fortunate","success","brave",
        }
        neg = {
            "hate","sad","anger","angry","fear","terrified","anxiety","worry","pain","grief",
            "awful","terrible","bad","horrible","failure","cruel","panic","despair",
        }
        cog = {
            "think","know","reason","because","therefore","infer","decide","plan","goal","idea",
            "understand","explain","analyze","consider","evaluate","compare","evidence",
        }
        cert = {
            "certain","sure","definitely","clearly","undoubtedly","always","never","must",
        }

        w2c: Dict[str, List[str]] = {}
        for w in pos:
            w2c.setdefault(w, []).append("emotion_positive")
        for w in neg:
            w2c.setdefault(w, []).append("emotion_negative")
        for w in cog:
            w2c.setdefault(w, []).append("cognition")
        for w in cert:
            w2c.setdefault(w, []).append("certainty")
        return Lexicon(word_to_cats=w2c)

    @staticmethod
    def from_csv(path: Path) -> "Lexicon":
        #CSVformat:word,category
        df = pd.read_csv(path)
        if "word" not in df.columns or "category" not in df.columns:
            raise ValueError("LexiconCSVmusthavecolumns:word,category")
        w2c: Dict[str, List[str]] = {}
        for _, row in df.iterrows():
            w = str(row["word"]).strip().lower()
            c = str(row["category"]).strip()
            if not w or not c:
                continue
            w2c.setdefault(w, []).append(c)
        return Lexicon(word_to_cats=w2c)


#-----------------------------#
#Scoring
#-----------------------------#

def score_document(tokens: List[str], lex: Lexicon) -> Dict[str, float]:
    #Returnsper-1000-tokenratesforeachcategory+basiccounts.
    total = len(tokens)
    counts: Dict[str, int] = {}
    for t in tokens:
        cats = lex.word_to_cats.get(t)
        if not cats:
            continue
        for c in cats:
            counts[c] = counts.get(c, 0) + 1

    out: Dict[str, float] = {"token_count": float(total)}
    if total == 0:
        for c in sorted(set(cat for cats in lex.word_to_cats.values() for cat in cats)):
            out[f"{c}_per_1k"] = 0.0
        return out

    for c, k in counts.items():
        out[f"{c}_per_1k"] = 1000.0 * (k / total)
    return out


def score_corpus(
    df: pd.DataFrame,
    text_col: str,
    time_col: str,
    lex: Lexicon,
    id_col: Optional[str] = None,
) -> pd.DataFrame:
    #Computesper-doclexiconscores.
    if text_col not in df.columns:
        raise ValueError(f"Missingtextcolumn:{text_col}")
    if time_col not in df.columns:
        raise ValueError(f"Missingtimecolumn:{time_col}")

    rows = []
    for i, row in df.iterrows():
        text = row[text_col]
        t = row[time_col]
        doc_id = row[id_col] if (id_col and id_col in df.columns) else i
        toks = tokenize(text)
        scores = score_document(toks, lex)
        scores["doc_id"] = doc_id
        scores[time_col] = t
        rows.append(scores)

    out = pd.DataFrame(rows)
    return out


#-----------------------------#
#Aggregation+trends
#-----------------------------#

def bootstrap_ci(values: np.ndarray, n_boot: int = 1000, alpha: float = 0.05, rng_seed: int = 0) -> Tuple[float, float]:
    #BasicbootstrapCIforthemean.
    if len(values) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(rng_seed)
    means = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        means.append(float(np.mean(sample)))
    means = np.array(means)
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return (lo, hi)


def aggregate_by_time(
    scored_df: pd.DataFrame,
    time_col: str,
    n_boot: int = 1000,
    rng_seed: int = 0,
) -> pd.DataFrame:
    #Aggregatesper-docscoresintoper-timepointmeans+CI.
    if time_col not in scored_df.columns:
        raise ValueError(f"Missingtimecolumn:{time_col}")

    score_cols = [c for c in scored_df.columns if c.endswith("_per_1k")]
    if not score_cols:
        raise ValueError("Noscoredcolumnsfound(endingwith_per_1k).")

    out_rows = []
    for t, grp in scored_df.groupby(time_col):
        row = {time_col: t, "n_docs": int(len(grp))}
        for c in score_cols:
            vals = grp[c].to_numpy(dtype=float)
            row[f"{c}_mean"] = float(np.mean(vals)) if len(vals) else np.nan
            row[f"{c}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            lo, hi = bootstrap_ci(vals, n_boot=n_boot, rng_seed=rng_seed)
            row[f"{c}_ci_low"] = lo
            row[f"{c}_ci_high"] = hi
        out_rows.append(row)

    out = pd.DataFrame(out_rows).sort_values(time_col).reset_index(drop=True)
    return out


def fit_linear_trend(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    #Descriptivelineartrend:slope,intercept,R^2.
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2:
        return {"slope": np.nan, "intercept": np.nan, "r2": np.nan}

    slope, intercept = np.polyfit(x, y, 1)
    yhat = slope * x + intercept
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = np.nan if ss_tot == 0 else 1.0 - (ss_res / ss_tot)
    return {"slope": float(slope), "intercept": float(intercept), "r2": float(r2)}


def compute_trends(agg_df: pd.DataFrame, time_col: str) -> Dict[str, Dict[str, float]]:
    #Trendpercategoryusingtheaggregatedmeans.
    trend = {}
    mean_cols = [c for c in agg_df.columns if c.endswith("_per_1k_mean")]
    x = pd.to_numeric(agg_df[time_col], errors="coerce").to_numpy(dtype=float)
    for c in mean_cols:
        y = agg_df[c].to_numpy(dtype=float)
        trend[c] = fit_linear_trend(x, y)
    return trend


#-----------------------------#
#CLI
#-----------------------------#

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Quantifyemotion+cognitionsignalsintextandtrackthemovertime."
    )
    p.add_argument("--input", required=True, help="Path to CSV with at least [time_col,text_col].")
    p.add_argument("--text-col", default="text", help="Column containing raw text.")
    p.add_argument("--time-col", default="year", help="Time column (e.g.,year).")
    p.add_argument("--id-col", default=None, help="Optional document id column.")
    p.add_argument("--lexicon", default=None, help="Optional lexicon CSV with columns [word,category].")
    p.add_argument("--out-doc", default="scored_by_doc.csv", help="Output per-document scored CSV.")
    p.add_argument("--out-time", default="scored_by_time.csv", help="Output aggregated-by-time CSV.")
    p.add_argument("--out-trends", default="trend_summary.json", help="Output trend summary JSON.")
    p.add_argument("--bootstrap", type=int, default=500, help="Bootstrap samples for CI (default 500).")
    p.add_argument("--seed", type=int, default=0, help="Random seed.")
    return p.parse_args()


def main():
    df = pd.read_csv("demo_corpus.csv")

    lex = Lexicon.built_in_minimal()

    scored = score_corpus(
        df=df,
        text_col="text",
        time_col="year",
        lex=lex,
        id_col=None,
    )
    scored.to_csv("scored_by_doc.csv", index=False)

    agg = aggregate_by_time(
        scored_df=scored,
        time_col="year",
        n_boot=200,
        rng_seed=0,
    )
    agg.to_csv("scored_by_time.csv", index=False)

    trends = compute_trends(agg_df=agg, time_col="year")
    with open("trend_summary.json", "w") as f:
        json.dump(trends, f, indent=2)

    print("Test run complete.")


if __name__ == "__main__":
    main()
