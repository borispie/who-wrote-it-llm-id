"""RQ2, part 2: when can the PROMPT alone give away the LLM?

In our main dataset every LLM answered the same 804 prompts, so the prompt carries no information about the
label and input-only accuracy should sit at chance. Many real datasets are not built like that: each prompt is
answered by only one LLM, and which LLM answers depends on where/when the prompt was collected.

This script simulates that. Each prompt gets exactly ONE label (one family's answer is kept):
  - with probability `bias` the family is chosen by the prompt's source dataset
    (selfinstruct->GPT, oasst->Claude, koala->Gemini, helpful_base->Llama, vicuna->Qwen),
  - otherwise the family is drawn uniformly at random.
bias = 0 is a fair collection; bias = 1 means the source fully decides the LLM.
We then train input-only classifiers (TF-IDF+LR and the CNN) and watch accuracy climb above chance.

Run:  python src/rq2_biased.py
Output: results/rq2_biased.json
"""
import numpy as np

from baselines import fit_tfidf_lr
from llmid import FAMILIES, TextConfig, TrainConfig, encode_frames, load_data, metrics, save_json, train_and_eval
from llmid import make_tokens

SOURCE_TO_FAMILY = {"selfinstruct": "GPT", "oasst": "Claude", "koala": "Gemini", "helpful_base": "Llama", "vicuna": "Qwen"}
BIASES = [0.0, 0.25, 0.5, 0.75, 1.0]
REPS = 5


def biased_sample(df, bias, rng):
    """Keep one row per prompt; the kept family depends on the prompt source with probability `bias`."""
    first = df.drop_duplicates("prompt_id").set_index("prompt_id")
    keep = []
    for pid, row in first.iterrows():
        if rng.random() < bias:
            fam = SOURCE_TO_FAMILY[row["prompt_source"]]
        else:
            fam = FAMILIES[rng.integers(len(FAMILIES))]
        keep.append((pid, fam))
    idx = df.set_index(["prompt_id", "llm_family"]).index
    mask = idx.isin(keep)
    return df[mask].copy()


def main():
    df = load_data()
    cfg = TextConfig()
    res = {"biases": BIASES, "reps": REPS, "source_to_family": SOURCE_TO_FAMILY, "lr": {}, "cnn": {}, "majority": {}}
    for bias in BIASES:
        lr_acc, cnn_acc, maj_acc = [], [], []
        for rep in range(REPS):
            rng = np.random.default_rng(1000 * rep + int(bias * 100))
            d = biased_sample(df, bias, rng)
            tr, va, te = (d[d["split"] == s] for s in ("train", "val", "test"))
            # majority-class baseline (the most common family in train)
            maj = tr["label"].value_counts().idxmax()
            maj_acc.append(float((te["label"] == maj).mean()))
            # TF-IDF + LR on the prompt only
            doc = lambda f: [make_tokens(i, o, "input", cfg) for i, o in zip(f["llm_input"], f["llm_output"])]
            vec, clf, _, _ = fit_tfidf_lr(doc(tr), tr["label"].values, doc(va), va["label"].values)
            lr_acc.append(metrics(te["label"].values, clf.predict(vec.transform(doc(te))))["accuracy"])
            # CNN on the prompt only
            vocab, train, (val, test) = encode_frames(tr, [va, te], "input", cfg)
            r, _ = train_and_eval("cnn", vocab, train, val, {"test": test}, seed=rep + 1, tcfg=TrainConfig(patience=5))
            cnn_acc.append(r["tests"]["test"]["accuracy"])
        res["lr"][str(bias)] = lr_acc
        res["cnn"][str(bias)] = cnn_acc
        res["majority"][str(bias)] = maj_acc
        print(f"bias {bias:.2f}: input-only acc  LR {np.mean(lr_acc):.3f}  CNN {np.mean(cnn_acc):.3f}  "
              f"majority {np.mean(maj_acc):.3f}  (chance {1/6:.3f})", flush=True)
    save_json(res, "results/rq2_biased.json")


if __name__ == "__main__":
    main()
