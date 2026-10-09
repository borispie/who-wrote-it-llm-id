"""RQ4 (extra credit), part 1: what characteristics distinguish the LLM families?

1. Compute simple, human-readable style features for every response (length, lexical diversity, formatting,
   punctuation, how the answer opens and closes).
2. Report per-family averages (results/rq4_feature_means.csv) and the most common opening words.
3. Train logistic regression on ONLY these features, grouped into length / formatting (structural) /
   linguistic, to see which kind of signal carries the most information.

Run:  python src/rq4_features.py
Output: results/rq4_features.json, results/rq4_feature_means.csv, results/rq4_features_per_response.csv
"""
import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from llmid import FAMILIES, load_data, metrics, save_json

WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿]")
CLOSING = re.compile(r"let me know|feel free|hope (this|that) helps|happy to help|any (other|more|further) questions", re.I)
SUMMARY = re.compile(r"\b(in summary|in conclusion|overall,|to summarize|in short)\b", re.I)
REFUSAL = re.compile(r"\b(i'm sorry|i am sorry|i cannot|i can't|as an ai|i'm not able|i am not able)\b", re.I)


def mattr(words, window=50):
    """Moving-average type-token ratio: lexical diversity that does not shrink just because a text is longer."""
    if len(words) < window:
        return len(set(words)) / max(len(words), 1)
    return float(np.mean([len(set(words[i : i + window])) / window for i in range(0, len(words) - window + 1, 5)]))


def features(text):
    words = WORD.findall(text)
    lw = [w.lower() for w in words]
    nw = max(len(words), 1)
    lines = [l for l in text.split("\n") if l.strip()]
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text) if WORD.search(s)]
    per100 = lambda c: 100.0 * c / nw
    return {
        # length
        "n_words": len(words),
        "n_lines": len(lines),
        "n_paragraphs": len([p for p in re.split(r"\n\s*\n", text) if p.strip()]),
        # formatting / structure
        "headers": sum(bool(re.match(r"\s*#{1,6}\s", l)) for l in lines),
        "bold": len(re.findall(r"\*\*[^*\n]+\*\*", text)),
        "bullets": sum(bool(re.match(r"\s*[-*•+]\s", l)) for l in lines),
        "numbered": sum(bool(re.match(r"\s*\d+[.)]\s", l)) for l in lines),
        "code_blocks": text.count("```") // 2,
        "table_rows": sum(l.count("|") >= 2 for l in lines),
        "emojis": len(EMOJI.findall(text)),
        # linguistic
        "mattr50": mattr(lw),
        "words_per_sentence": len(words) / max(len(sentences), 1),
        "avg_word_len": float(np.mean([len(w) for w in words])) if words else 0.0,
        "exclam_per100": per100(text.count("!")),
        "question_per100": per100(text.count("?")),
        "colon_per100": per100(text.count(":")),
        "dash_per100": per100(text.count("—") + text.count(" - ")),
        "I_per100": per100(sum(w == "I" for w in words)),
        "you_per100": per100(sum(w == "you" for w in lw)),
        "closing_offer": int(bool(CLOSING.search(text[-300:]))),
        "summary_phrase": int(bool(SUMMARY.search(text))),
        "refusal_phrase": int(bool(REFUSAL.search(text))),
    }


GROUPS = {
    "length only": ["n_words", "n_lines", "n_paragraphs"],
    "formatting only": ["headers", "bold", "bullets", "numbered", "code_blocks", "table_rows", "emojis"],
    "linguistic only": ["mattr50", "words_per_sentence", "avg_word_len", "exclam_per100", "question_per100",
                        "colon_per100", "dash_per100", "I_per100", "you_per100", "closing_offer", "summary_phrase",
                        "refusal_phrase"],
}
GROUPS["all features"] = sum(GROUPS.values(), [])


def first_word(text):
    m = WORD.search(text)
    return m.group(0).lower() if m else "<none>"


def main():
    df = load_data()
    feats = pd.DataFrame([features(t) for t in df["llm_output"]])
    feats.insert(0, "llm_family", df["llm_family"].values)
    feats.insert(1, "split", df["split"].values)
    feats.to_csv("results/rq4_features_per_response.csv", index=False)

    means = feats.groupby("llm_family")[GROUPS["all features"]].mean().reindex(FAMILIES)
    means.round(3).to_csv("results/rq4_feature_means.csv")
    print(means.round(2).T.to_string())
    # share of responses that use each formatting device at least once (easier to read than averages,
    # which a few extreme answers can inflate, e.g. one answer made of 100 emojis)
    share_cols = ["headers", "bold", "bullets", "numbered", "code_blocks", "table_rows", "emojis"]
    shares = (feats[share_cols] > 0).groupby(feats["llm_family"]).mean().reindex(FAMILIES)
    shares.round(4).to_csv("results/rq4_feature_shares.csv")
    print("\nshare of responses using each device:\n", shares.round(2).T.to_string())

    # simple "fingerprint" markers: share of each family's answers that show the marker
    g = df.groupby("llm_family")["llm_output"]
    markers = pd.DataFrame({
        "curly apostrophe (’)": g.apply(lambda x: x.str.contains("\u2019").mean()),
        "bullets written as '* '": g.apply(lambda x: x.str.contains(r"(?m)^\s*\* ").mean()),
        "bullets written as '- '": g.apply(lambda x: x.str.contains(r"(?m)^\s*- ").mean()),
        "starts with Sure/Certainly/Absolutely/Of course": g.apply(
            lambda x: x.str.match(r"(Sure|Certainly|Absolutely|Of course)").mean()),
        "starts with Here's/Here is/Here are": g.apply(lambda x: x.str.match(r"Here('s| is| are)").mean()),
        "starts with What a/What an": g.apply(lambda x: x.str.match(r"What an? ").mean()),
    }).reindex(FAMILIES)
    markers.round(4).to_csv("results/rq4_markers.csv")
    print("\nmarkers:\n", markers.round(2).T.to_string())

    # how responses open: most common first words per family
    df["opener"] = df["llm_output"].map(first_word)
    openers = {fam: Counter(df.loc[df["llm_family"] == fam, "opener"]).most_common(5) for fam in FAMILIES}
    for fam, o in openers.items():
        print(f"{fam:9s} opens with:", ", ".join(f"{w} ({c})" for w, c in o))

    # feature-only classifiers
    y = df["label"].values
    tr, va, te = (df["split"].values == s for s in ("train", "val", "test"))
    clf_res = {}
    for name, cols in GROUPS.items():
        X = feats[cols].values.astype(float)
        best = None
        for C in (0.1, 1.0, 10.0):
            m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000)).fit(X[tr], y[tr])
            f1 = metrics(y[va], m.predict(X[va]))["macro_f1"]
            if best is None or f1 > best[0]:
                best = (f1, C, m)
        r = metrics(y[te], best[2].predict(X[te]))
        clf_res[name] = {"accuracy": r["accuracy"], "macro_f1": r["macro_f1"], "per_class_f1": r["per_class_f1"],
                         "C": best[1], "n_features": len(cols)}
        print(f"features: {name:16s} test acc {r['accuracy']:.3f}  macro-F1 {r['macro_f1']:.3f}")

    save_json({"feature_means": means.round(4).to_dict(orient="index"), "openers": openers,
               "feature_classifiers": clf_res, "groups": GROUPS}, "results/rq4_features.json")


if __name__ == "__main__":
    main()
