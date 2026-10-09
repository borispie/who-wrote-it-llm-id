"""Non-neural baselines for RQ1/RQ2: chance, and TF-IDF + logistic regression.

The TF-IDF model sees exactly the same (truncated) token stream as the CNN/LSTM, so the comparison is fair.
Its weights are also used in RQ4 to list the n-grams that point most strongly to each LLM family.

Run:  python src/baselines.py
Output: results/baselines.json
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from llmid import FAMILIES, TextConfig, load_data, make_tokens, metrics, save_json


def fit_tfidf_lr(tr_docs, y_tr, va_docs, y_va, Cs=(0.3, 1.0, 3.0, 10.0)):
    """Pick C on the validation set (macro-F1), return the fitted vectorizer + model."""
    vec = TfidfVectorizer(analyzer=lambda d: d, min_df=2, sublinear_tf=True)
    Xtr, Xva = vec.fit_transform(tr_docs), None
    Xva = vec.transform(va_docs)
    best = None
    for C in Cs:
        clf = LogisticRegression(C=C, max_iter=3000)
        clf.fit(Xtr, y_tr)
        f1 = metrics(y_va, clf.predict(Xva))["macro_f1"]
        if best is None or f1 > best[0]:
            best = (f1, C, clf)
    return vec, best[2], best[1], best[0]


def with_bigrams(tokens):
    """Unigrams + bigrams as features (tokens already include formatting marks)."""
    return tokens + [a + " " + b for a, b in zip(tokens, tokens[1:])]


def docs(df, mode, cfg):
    return [with_bigrams(make_tokens(i, o, mode, cfg)) for i, o in zip(df["llm_input"], df["llm_output"])]


def main():
    df = load_data()
    cfg = TextConfig()
    tr, va, te = (df[df["split"] == s] for s in ("train", "val", "test"))
    out = {"chance_accuracy": 1 / len(FAMILIES), "tfidf_lr": {}}
    for mode in ["output", "input", "input_output"]:
        vec, clf, C, val_f1 = fit_tfidf_lr(docs(tr, mode, cfg), tr["label"].values, docs(va, mode, cfg), va["label"].values)
        pred = clf.predict(vec.transform(docs(te, mode, cfg)))
        m = metrics(te["label"].values, pred)
        m.update({"C": C, "val_macro_f1": val_f1, "pred": pred.tolist(), "gold": te["label"].tolist()})
        out["tfidf_lr"][mode] = m
        print(f"TF-IDF+LR {mode:12s}: test acc {m['accuracy']:.3f}  macro-F1 {m['macro_f1']:.3f}  (C={C})")
        if mode == "output":  # top n-grams per family, for RQ4
            names = np.array(vec.get_feature_names_out())
            top = {}
            for k, fam in enumerate(FAMILIES):
                idx = np.argsort(clf.coef_[k])[::-1][:15]
                top[fam] = [(names[i], round(float(clf.coef_[k][i]), 3)) for i in idx]
            out["top_ngrams_output"] = top
    # Why split by prompt? Compare with a random split by ROW of the same sizes (3 random splits).
    row = []
    for seed in (1, 2, 3):
        idx = np.random.default_rng(seed).permutation(len(df))
        n_tr, n_va = len(tr), len(va)
        rtr, rva, rte = df.iloc[idx[:n_tr]], df.iloc[idx[n_tr:n_tr + n_va]], df.iloc[idx[n_tr + n_va:]]
        vec, clf, _, _ = fit_tfidf_lr(docs(rtr, "output", cfg), rtr["label"].values, docs(rva, "output", cfg),
                                      rva["label"].values)
        row.append(metrics(rte["label"].values, clf.predict(vec.transform(docs(rte, "output", cfg))))["accuracy"])
    out["tfidf_lr_row_split_output"] = row
    print(f"TF-IDF+LR output with a random ROW split: {np.mean(row):.3f} (prompt split: "
          f"{out['tfidf_lr']['output']['accuracy']:.3f})")
    save_json(out, "results/baselines.json")


if __name__ == "__main__":
    main()
