"""RQ3 (extra credit): do LLM fingerprints carry over to a task type the model never saw in training?

Two shifts (the same ones the assignment suggests):
  writing : train on coding + math + QA/advice prompts,  test on WRITING prompts
  coding  : train on general QA/advice prompts only,      test on CODING prompts

For each shift we train two kinds of model and test both on the SAME target prompts:
  cross    - never sees the target task in training.
  control  - same number of training prompts, but half of the target-task prompts are mixed in (2 folds,
             each fold is tested on the other half). The gap  control - cross  is the cost of the task shift,
             with training-set size held fixed.
Early stopping always uses validation prompts from the source tasks only, so target prompts stay unseen.

Run:  python src/rq3_cross_task.py --models cnn --seeds 1 2 3
      python src/rq3_cross_task.py --models lstm --seeds 1 --shifts writing
Output: results/rq3/<shift>_<model>_seed<k>.json
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from llmid import TextConfig, encode_frames, load_data, metrics, save_json, train_and_eval

SHIFTS = {
    "writing": {"source": ["coding", "math", "qa_advice"], "target": "writing"},
    "coding": {"source": ["qa_advice"], "target": "coding"},
}
OUT_DIR = os.path.join("results", "rq3")


def run_parts(shift, kind, seed, parts=("cross", "control0", "control1")):
    """Train the requested parts. Running the parts in separate processes gives the same result as one full run."""
    df = load_data()
    cfg = TextConfig()
    src, tgt = SHIFTS[shift]["source"], SHIFTS[shift]["target"]
    is_src = df["task_type"].isin(src)
    is_tgt = df["task_type"] == tgt

    train_src = df[is_src & (df["split"] == "train")]
    val_src = df[is_src & (df["split"] == "val")]
    test_src = df[is_src & (df["split"] == "test")]        # in-domain reference
    target = df[is_tgt]                                       # every target-task prompt (all splits)
    n_train_prompts = train_src["prompt_id"].nunique()
    out = {"shift": shift, "model": kind, "seed": seed, "source_tasks": src, "target_task": tgt,
           "n_train_prompts": int(n_train_prompts), "n_target_prompts": int(target["prompt_id"].nunique())}

    # ---- cross-task model: source tasks only
    if "cross" in parts:
        vocab, train, (val, tgt_all, src_test) = encode_frames(train_src, [val_src, target, test_src], "output", cfg)
        cross, _ = train_and_eval(kind, vocab, train, val, {"target": tgt_all, "source_test": src_test}, seed)
        out["cross"] = {"target": dict(cross["tests"]["target"]), "source_test": dict(cross["tests"]["source_test"]),
                        "best_epoch": cross["best_epoch"], "val_macro_f1": cross["val_macro_f1"]}

    # ---- size-matched control: half of the target prompts swapped into training (2 folds)
    rng = np.random.default_rng(seed)
    tgt_ids = np.array(sorted(target["prompt_id"].unique()))
    rng.shuffle(tgt_ids)
    halves = [tgt_ids[: len(tgt_ids) // 2], tgt_ids[len(tgt_ids) // 2 :]]
    src_ids = np.array(sorted(train_src["prompt_id"].unique()))
    for k in range(2):
        in_ids, out_ids = halves[k], halves[1 - k]
        keep_src = rng.choice(src_ids, size=n_train_prompts - len(in_ids), replace=False)  # always drawn
        if f"control{k}" not in parts:
            continue
        train_ctrl = df[df["prompt_id"].isin(keep_src) | df["prompt_id"].isin(in_ids)]
        test_ctrl = df[df["prompt_id"].isin(out_ids)]
        vocab_c, train_c, (val_c, tgt_c) = encode_frames(train_ctrl, [val_src, test_ctrl], "output", cfg)
        r, _ = train_and_eval(kind, vocab_c, train_c, val_c, {"target_half": tgt_c}, seed)
        out[f"control{k}"] = {"gold": r["tests"]["target_half"]["gold"], "pred": r["tests"]["target_half"]["pred"],
                              "fold": {"best_epoch": r["best_epoch"], "val_macro_f1": r["val_macro_f1"],
                                       "n_train": len(train_c), "target_prompts_in_train": int(len(in_ids))}}
    return out


def merge(pieces):
    """Combine the cross part and both control folds into one result."""
    out = {k: v for k, v in pieces[0].items() if k not in ("cross", "control0", "control1")}
    get = lambda key: next(p[key] for p in pieces if key in p)
    out["cross"] = get("cross")
    c0, c1 = get("control0"), get("control1")
    control = metrics(np.array(c0["gold"] + c1["gold"]), np.array(c0["pred"] + c1["pred"]))
    control.update({"gold": c0["gold"] + c1["gold"], "pred": c0["pred"] + c1["pred"], "folds": [c0["fold"], c1["fold"]]})
    out["control"] = control
    return out


def run_shift(shift, kind, seed):
    return merge([run_parts(shift, kind, seed)])


def summarize():
    rows = []
    for f in sorted(os.listdir(OUT_DIR)):
        if f.endswith(".json"):
            r = pd.read_json(os.path.join(OUT_DIR, f), typ="series")
            rows.append({"shift": r["shift"], "model": r["model"], "seed": r["seed"],
                         "cross_target_acc": r["cross"]["target"]["accuracy"],
                         "control_target_acc": r["control"]["accuracy"],
                         "cross_source_test_acc": r["cross"]["source_test"]["accuracy"]})
    s = pd.DataFrame(rows)
    s.to_csv(os.path.join("results", "rq3_summary.csv"), index=False)
    print(s.groupby(["shift", "model"]).mean(numeric_only=True).round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["cnn"])
    ap.add_argument("--shifts", nargs="+", default=list(SHIFTS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--part", choices=["cross", "control0", "control1"], default=None,
                    help="train only one part (to run the parts in parallel); combine them later with --merge")
    ap.add_argument("--merge", action="store_true", help="combine saved parts into the final result files")
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    part_dir = os.path.join("results", "rq3_parts")
    for shift in a.shifts:
        for kind in a.models:
            for seed in a.seeds:
                path = os.path.join(OUT_DIR, f"{shift}_{kind}_seed{seed}.json")
                if a.part:
                    os.makedirs(part_dir, exist_ok=True)
                    save_json(run_parts(shift, kind, seed, parts=(a.part,)),
                              os.path.join(part_dir, f"{shift}_{kind}_seed{seed}_{a.part}.json"))
                    print(f"{shift} {kind} seed{seed}: part {a.part} done", flush=True)
                    continue
                if a.merge:
                    pieces = []
                    for part in ("cross", "control0", "control1"):
                        with open(os.path.join(part_dir, f"{shift}_{kind}_seed{seed}_{part}.json")) as f:
                            pieces.append(json.load(f))
                    r = merge(pieces)
                elif os.path.exists(path):
                    print("skip (done):", path)
                    continue
                else:
                    r = run_shift(shift, kind, seed)
                print(f"{shift:8s} {kind:4s} seed{seed}: target acc  cross {r['cross']['target']['accuracy']:.3f}  "
                      f"control {r['control']['accuracy']:.3f}   (cross on its own tasks "
                      f"{r['cross']['source_test']['accuracy']:.3f})", flush=True)
                save_json(r, path)
    if not a.part:
        summarize()
