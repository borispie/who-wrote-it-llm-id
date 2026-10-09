"""RQ4 (extra credit), part 2: does removing a kind of signal hurt the classifier?

Each ablation changes the response text. We measure the CNN in two ways:
  test-time : the normal CNN (trained on full responses) is tested on the ablated test responses
              -> how much does the trained model RELY on this signal?
  retrained : a new CNN is trained and tested on ablated responses
              -> is there enough OTHER signal left to identify the LLM?

Ablations
  no_format      remove markdown (#, **, bullets, numbered-list marks, `, |) and all line breaks
  words_only     lowercase words only: no punctuation, no formatting, no line breaks
  structure_only every word -> <w>, every number -> <num>; keeps punctuation, line breaks, markdown
  opening_only   only the first 30 tokens of the response
  middle_only    tokens 50-150 only (no opening, no ending, and every input has the same length)

Run:  python src/rq4_ablation.py --seeds 1 2 3
Output: results/rq4_ablation.json
"""
import argparse
import re

import numpy as np

from baselines import fit_tfidf_lr
from llmid import TextConfig, TrainConfig, Vocab, head_tail, load_data, metrics, predict, save_json, tokenize
from llmid import train_and_eval

CFG = TextConfig()


def strip_format(t):
    t = re.sub(r"(?m)^\s*#{1,6}\s*", "", t)          # headers
    t = re.sub(r"(?m)^\s*([-*•+]|\d+[.)])\s+", "", t)  # bullet and numbered-list marks
    t = re.sub(r"(?m)^\s*>\s*", "", t)                 # quotes
    t = re.sub(r"\*\*|__|`+|\|", " ", t)                # bold, code marks, table bars
    t = re.sub(r"(?<!\w)\*(?!\s)|(?<!\s)\*(?!\w)", "", t)  # single-* italics
    return re.sub(r"\s+", " ", t).strip()


def words_only(t):
    return " ".join(re.findall(r"[a-z0-9]+(?:'[a-z]+)?", t.lower()))


def structure_tokens(toks):
    out = []
    for t in toks:
        if re.fullmatch(r"[a-z]+(?:'[a-z]+)?", t):
            out.append("<w>")
        elif re.fullmatch(r"\d+", t):
            out.append("<num>")
        else:
            out.append(t)
    return out


def ablate(text, name):
    """Return the token list the model sees for one response under a given ablation."""
    if name == "original":
        return head_tail(tokenize(text), CFG.head, CFG.tail)
    if name == "no_format":
        return head_tail(tokenize(strip_format(text)), CFG.head, CFG.tail)
    if name == "words_only":
        return head_tail(tokenize(words_only(text)), CFG.head, CFG.tail)
    if name == "structure_only":
        return head_tail(structure_tokens(tokenize(text)), CFG.head, CFG.tail)
    if name == "opening_only":
        return tokenize(text)[:30]
    if name == "middle_only":
        return tokenize(text)[50:150] or ["<empty>"]
    raise ValueError(name)


ABLATIONS = ["no_format", "words_only", "structure_only", "opening_only", "middle_only"]


def encode(vocab, token_lists, labels):
    return list(zip([vocab.encode(t) for t in token_lists], labels))


def main(seeds):
    df = load_data()
    tr, va, te = (df[df["split"] == s] for s in ("train", "val", "test"))
    toks = {name: {s: [ablate(t, name) for t in d["llm_output"]] for s, d in (("train", tr), ("val", va), ("test", te))}
            for name in ["original"] + ABLATIONS}
    ys = {s: d["label"].tolist() for s, d in (("train", tr), ("val", va), ("test", te))}
    res = {"seeds": seeds, "cnn": {}, "tfidf_lr": {}}

    for seed in seeds:
        # 1) normal CNN, trained on original responses
        vocab = Vocab(toks["original"]["train"], CFG.min_freq, CFG.max_vocab)
        data = {s: encode(vocab, toks["original"][s], ys[s]) for s in ys}
        base, model = train_and_eval("cnn", vocab, data["train"], data["val"], {"test": data["test"]}, seed,
                                     TrainConfig())
        res["cnn"].setdefault("original", []).append({"test_time": base["tests"]["test"]["accuracy"],
                                                      "retrained": base["tests"]["test"]["accuracy"],
                                                      "test_time_f1": base["tests"]["test"]["macro_f1"],
                                                      "retrained_f1": base["tests"]["test"]["macro_f1"]})
        for name in ABLATIONS:
            # 2a) test-time ablation with the normal model
            g, p = predict(model, encode(vocab, toks[name]["test"], ys["test"]))
            tt = metrics(g, p)
            # 2b) retrain on ablated text
            v2 = Vocab(toks[name]["train"], CFG.min_freq, CFG.max_vocab)
            d2 = {s: encode(v2, toks[name][s], ys[s]) for s in ys}
            r2, _ = train_and_eval("cnn", v2, d2["train"], d2["val"], {"test": d2["test"]}, seed, TrainConfig())
            rt = r2["tests"]["test"]
            res["cnn"].setdefault(name, []).append({"test_time": tt["accuracy"], "retrained": rt["accuracy"],
                                                    "test_time_f1": tt["macro_f1"], "retrained_f1": rt["macro_f1"],
                                                    "retrained_per_class_f1": rt["per_class_f1"]})
            print(f"seed{seed} {name:15s} CNN acc: test-time {tt['accuracy']:.3f}  retrained {rt['accuracy']:.3f}",
                  flush=True)
        save_json(res, "results/rq4_ablation.json")  # save after every seed

    # TF-IDF + LR retrained on each ablation (fast, one run) as a second opinion
    for name in ["original"] + ABLATIONS:
        bi = lambda L: [t + [a + " " + b for a, b in zip(t, t[1:])] for t in L]
        vec, clf, _, _ = fit_tfidf_lr(bi(toks[name]["train"]), ys["train"], bi(toks[name]["val"]), ys["val"])
        m = metrics(np.array(ys["test"]), clf.predict(vec.transform(bi(toks[name]["test"]))))
        res["tfidf_lr"][name] = {"accuracy": m["accuracy"], "macro_f1": m["macro_f1"]}
        print(f"TF-IDF+LR {name:15s} retrained acc {m['accuracy']:.3f}", flush=True)

    # one example of each ablation, for the report
    ex = te["llm_output"].iloc[3]
    res["example"] = {name: " ".join(ablate(ex, name)[:60]) for name in ["original"] + ABLATIONS}
    save_json(res, "results/rq4_ablation.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    main(ap.parse_args().seeds)
