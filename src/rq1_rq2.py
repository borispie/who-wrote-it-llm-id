"""RQ1 + RQ2: train the CNN and the BiLSTM on three kinds of input.

  output        -> RQ1 (which LLM wrote this response?)
  input         -> RQ2 (prompt only)
  input_output  -> RQ2 (prompt + response)

Run:  python src/rq1_rq2.py                       (all models, all modes, seeds 1 2 3)
      python src/rq1_rq2.py --models cnn --modes output --seeds 1
Output: results/rq1_rq2/<model>_<mode>_seed<k>.json  and  results/rq1_rq2_summary.csv
"""
import argparse
import os
import time

import pandas as pd

from llmid import FAMILIES, TextConfig, TrainConfig, encode_frames, load_data, save_json, train_and_eval, config_dict

OUT_DIR = os.path.join("results", "rq1_rq2")


def run(model_kind, mode, seed, max_epochs=None, verbose=True):
    df = load_data()
    tr, va, te = (df[df["split"] == s] for s in ("train", "val", "test"))
    vocab, train, (val, test) = encode_frames(tr, [va, te], mode, TextConfig())
    tcfg = TrainConfig() if max_epochs is None else TrainConfig(max_epochs=max_epochs)
    res, _ = train_and_eval(model_kind, vocab, train, val, {"test": test}, seed, tcfg, verbose=verbose)
    res.update({"mode": mode, "families": FAMILIES, "config": config_dict(),
                "n_train": len(train), "n_val": len(val), "n_test": len(test),
                "test_prompt_ids": te["prompt_id"].tolist()})
    return res


def summarize():
    rows = []
    for f in sorted(os.listdir(OUT_DIR)):
        if f.endswith(".json"):
            r = pd.read_json(os.path.join(OUT_DIR, f), typ="series")
            t = r["tests"]["test"]
            rows.append({"model": r["model"], "mode": r["mode"], "seed": r["seed"], "test_acc": t["accuracy"],
                         "test_macro_f1": t["macro_f1"], "val_macro_f1": r["val_macro_f1"],
                         "best_epoch": r["best_epoch"], "train_seconds": r["train_seconds"]})
    s = pd.DataFrame(rows).sort_values(["mode", "model", "seed"])
    s.to_csv(os.path.join("results", "rq1_rq2_summary.csv"), index=False)
    print(s.groupby(["mode", "model"])[["test_acc", "test_macro_f1"]].agg(["mean", "std"]).round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["cnn", "lstm"])
    ap.add_argument("--modes", nargs="+", default=["output", "input", "input_output"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--max_epochs", type=int, default=None)
    ap.add_argument("--no_save", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    for mode in a.modes:
        for kind in a.models:
            for seed in a.seeds:
                path = os.path.join(OUT_DIR, f"{kind}_{mode}_seed{seed}.json")
                if os.path.exists(path) and not a.no_save:
                    print("skip (done):", path)
                    continue
                t0 = time.time()
                res = run(kind, mode, seed, a.max_epochs)
                t = res["tests"]["test"]
                print(f"{kind:4s} {mode:12s} seed{seed}: test acc {t['accuracy']:.3f}  macro-F1 {t['macro_f1']:.3f} "
                      f"(best epoch {res['best_epoch']}, {time.time() - t0:.0f}s)", flush=True)
                if not a.no_save:
                    save_json(res, path)
    if not a.no_save:
        summarize()
