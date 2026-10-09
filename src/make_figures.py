"""Make every figure and LaTeX table used in the report, straight from the files in results/.

Run (after the experiments):  python src/make_figures.py
Output: figures/*.png  and  report/tables/*.tex  and  report/numbers.tex
"""
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from llmid import FAMILIES, load_data

FIG, TAB = "figures", os.path.join("report", "tables")
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

FAM_COLORS = {"GPT": "#009E73", "Claude": "#D55E00", "Gemini": "#0072B2", "Llama": "#CC79A7", "Qwen": "#E69F00",
              "DeepSeek": "#56B4E9"}
MODEL_COLORS = {"CNN": "#0072B2", "BiLSTM": "#D55E00", "TF-IDF + LR": "#8C8C8C"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 200,
                     "axes.titleweight": "bold", "axes.titlesize": 10.5})
CHANCE = 1 / len(FAMILIES)
NUMBERS = {}  # \newcommand macros for numbers quoted in the text


def load(path):
    with open(path) as f:
        return json.load(f)


def runs(model, mode):
    return [load(p) for p in sorted(glob.glob(f"results/rq1_rq2/{model}_{mode}_seed*.json"))]


def pct(x):
    return f"{100 * x:.1f}"


def mean_std(xs):
    xs = np.array(xs, dtype=float)
    return xs.mean(), (xs.std(ddof=1) if len(xs) > 1 else 0.0)


def signed(x, plus=False):
    """Format a number with a typeset minus sign (and an optional plus sign)."""
    t = f"{x:+.1f}" if plus else f"{x:.1f}"
    return t.replace("-", "$-$")


def fmt_ms(xs):
    m, s = mean_std(xs)
    return f"{100 * m:.1f} $\\pm$ {100 * s:.1f}" if len(xs) > 1 else f"{100 * m:.1f}"


def num(name, value):
    NUMBERS[name] = value


# ----------------------------------------------------------------------------- dataset
def dataset_assets():
    df = load_data()
    df["words"] = df["llm_output"].str.split().str.len()
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2), gridspec_kw={"width_ratios": [1, 1.6]})
    tt = df.drop_duplicates("prompt_id")["task_type"].value_counts().reindex(["qa_advice", "writing", "coding", "math"])
    ax[0].bar(["QA / advice", "writing", "coding", "math"], tt.values, color="#5B7FA6")
    for i, v in enumerate(tt.values):
        ax[0].text(i, v + 8, str(v), ha="center", fontsize=9)
    ax[0].set_ylabel("number of prompts")
    ax[0].set_title("Prompts per task type")
    data = [df.loc[df["llm_family"] == f, "words"].values for f in FAMILIES]
    bp = ax[1].boxplot(data, tick_labels=FAMILIES, showfliers=False, patch_artist=True, widths=0.6)
    for patch, f in zip(bp["boxes"], FAMILIES):
        patch.set_facecolor(FAM_COLORS[f])
        patch.set_alpha(0.75)
    for med in bp["medians"]:
        med.set_color("black")
    ax[1].set_ylabel("words per response")
    ax[1].set_title("Response length by LLM family")
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig_dataset.png")
    plt.close(fig)

    names = df.drop_duplicates("llm_family").set_index("llm_family")["llm_name"]
    rows = []
    for f in FAMILIES:
        w = df.loc[df["llm_family"] == f, "words"]
        rows.append(f"{f} & \\texttt{{{names[f]}}} & {int((df['llm_family'] == f).sum())} & {int(w.median())} & "
                    f"{w.mean():.0f} \\\\")
    with open(f"{TAB}/dataset.tex", "w") as fh:
        fh.write("\n".join(rows))
    split = df.drop_duplicates("prompt_id").groupby(["task_type", "split"]).size().unstack().reindex(
        ["qa_advice", "writing", "coding", "math"])[["train", "val", "test"]]
    lines = [f"{t.replace('qa_advice', 'QA / advice')} & {r['train']} & {r['val']} & {r['test']} & {r.sum()} \\\\"
             for t, r in split.iterrows()]
    tot = split.sum()
    lines.append("\\midrule")
    lines.append(f"all prompts & {tot['train']} & {tot['val']} & {tot['test']} & {tot.sum()} \\\\")
    lines.append(f"responses ($\\times$6) & {6 * tot['train']} & {6 * tot['val']} & {6 * tot['test']} & "
                 f"{6 * tot.sum()} \\\\")
    with open(f"{TAB}/splits.tex", "w") as fh:
        fh.write("\n".join(lines))
    num("NPrompts", str(df["prompt_id"].nunique()))
    num("NRows", f"{len(df):,}")


# ----------------------------------------------------------------------------- RQ1 / RQ2
def rq1_rq2_assets():
    base = load("results/baselines.json")
    modes = ["output", "input", "input_output"]
    res = {m: {"CNN": runs("cnn", m), "BiLSTM": runs("lstm", m)} for m in modes}

    # RQ1 table: accuracy, macro-F1, per-family F1
    rows = []
    for label, rr in [("CNN", res["output"]["CNN"]), ("BiLSTM", res["output"]["BiLSTM"])]:
        if not rr:
            continue
        acc = [r["tests"]["test"]["accuracy"] for r in rr]
        f1 = [r["tests"]["test"]["macro_f1"] for r in rr]
        pcf = np.mean([r["tests"]["test"]["per_class_f1"] for r in rr], axis=0)
        rows.append(f"{label} & {fmt_ms(acc)} & {fmt_ms(f1)} & " + " & ".join(pct(x) for x in pcf) + " \\\\")
        key = "Cnn" if label == "CNN" else "Lstm"
        num(f"{key}Params", f"{rr[0]['n_params']:,}")
        num(f"{key}Vocab", f"{rr[0]['vocab_size']:,}")
        num(f"{key}Epochs", f"{np.mean([r['best_epoch'] for r in rr]):.0f}")
        num(f"{key}Minutes", f"{np.mean([r['train_seconds'] for r in rr]) / 60:.0f}")
        num(f"{key}Acc", pct(np.mean(acc)))
        num(f"{key}Min", pct(min(acc)))
        num(f"{key}Max", pct(max(acc)))
        num(f"{key}AccSd", pct(np.std(acc, ddof=1)) if len(acc) > 1 else "0.0")
        num(f"{key}Fone", pct(np.mean(f1)))
        num(f"{key}Seeds", str(len(rr)))
    if res["output"]["CNN"] and res["output"]["BiLSTM"]:
        gap = np.mean([r["tests"]["test"]["accuracy"] for r in res["output"]["CNN"]]) - \
              np.mean([r["tests"]["test"]["accuracy"] for r in res["output"]["BiLSTM"]])
        num("RqOneGap", f"{100 * gap:.0f}")
    b = base["tfidf_lr"]["output"]
    rows.append(f"TF-IDF + LR (baseline) & {pct(b['accuracy'])} & {pct(b['macro_f1'])} & " +
                " & ".join(pct(x) for x in b["per_class_f1"]) + " \\\\")
    rows.append(f"Chance & {pct(CHANCE)} & -- & " + " & ".join("--" for _ in FAMILIES) + " \\\\")
    num("LrAcc", pct(b["accuracy"]))
    with open(f"{TAB}/rq1.tex", "w") as fh:
        fh.write("\n".join(rows))

    # confusion matrices (summed over seeds, row-normalised)
    present = [(lab, rr) for lab, rr in [("CNN", res["output"]["CNN"]), ("BiLSTM", res["output"]["BiLSTM"])] if rr]
    fig, axes = plt.subplots(1, len(present), figsize=(4.6 * len(present), 4.1))
    axes = np.atleast_1d(axes)
    for ax, (lab, rr) in zip(axes, present):
        cm = np.sum([np.array(r["tests"]["test"]["confusion"]) for r in rr], axis=0).astype(float)
        cm = cm / cm.sum(1, keepdims=True)
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(FAMILIES)):
            for j in range(len(FAMILIES)):
                ax.text(j, i, f"{100 * cm[i, j]:.0f}", ha="center", va="center", fontsize=9,
                        color="white" if cm[i, j] > 0.55 else "black")
        ax.set_xticks(range(len(FAMILIES)), FAMILIES, rotation=40, ha="right")
        ax.set_yticks(range(len(FAMILIES)), FAMILIES)
        ax.set_xlabel("predicted family")
        ax.set_ylabel("true family")
        ax.set_title(f"{lab} (output only), % of each row")
        ax.spines[:].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig_confusion.png")
    plt.close(fig)

    # learning curves (seed 1)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.1))
    for lab, rr in present:
        h = pd.DataFrame(rr[0]["history"])
        ax[0].plot(h["epoch"], h["train_loss"], "-o", ms=3, color=MODEL_COLORS[lab], label=lab)
        ax[1].plot(h["epoch"], h["val_f1"], "-o", ms=3, color=MODEL_COLORS[lab], label=lab)
        ax[1].axvline(rr[0]["best_epoch"], color=MODEL_COLORS[lab], ls=":", lw=1)
    ax[0].set_title("Training loss")
    ax[1].set_title("Validation macro-F1 (dotted = epoch kept)")
    for a in ax:
        a.set_xlabel("epoch")
        a.legend(frameon=False)
    ax[1].axhline(CHANCE, color="grey", ls="--", lw=0.8)
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig_learning_curves.png")
    plt.close(fig)

    # RQ2 table + bars
    mode_names = {"input": "Input only (prompt)", "output": "Output only (response)", "input_output": "Input + output"}
    lines = []
    bars = {}
    for m in ["input", "output", "input_output"]:
        cells = []
        for lab in ["CNN", "BiLSTM"]:
            acc = [r["tests"]["test"]["accuracy"] for r in res[m][lab]]
            f1 = [r["tests"]["test"]["macro_f1"] for r in res[m][lab]]
            cells += [fmt_ms(acc) if acc else "--", fmt_ms(f1) if f1 else "--"]
            bars[(m, lab)] = mean_std(acc) if acc else (np.nan, 0)
            key = {"input": "In", "output": "Out", "input_output": "InOut"}[m] + ("Cnn" if lab == "CNN" else "Lstm")
            if acc:
                num(f"RqTwo{key}", pct(np.mean(acc)))
        b = base["tfidf_lr"][m]
        cells += [pct(b["accuracy"]), pct(b["macro_f1"])]
        bars[(m, "TF-IDF + LR")] = (b["accuracy"], 0)
        num("RqTwo" + {"input": "In", "output": "Out", "input_output": "InOut"}[m] + "Lr", pct(b["accuracy"]))
        lines.append(f"{mode_names[m]} & " + " & ".join(cells) + " \\\\")
    with open(f"{TAB}/rq2.tex", "w") as fh:
        fh.write("\n".join(lines))

    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    x = np.arange(3)
    for k, lab in enumerate(["CNN", "BiLSTM", "TF-IDF + LR"]):
        ms = [bars[(m, lab)] for m in ["input", "output", "input_output"]]
        ax.bar(x + (k - 1) * 0.26, [100 * a for a, _ in ms], 0.26, yerr=[100 * s for _, s in ms], capsize=3,
               color=MODEL_COLORS[lab], label=lab)
    ax.axhline(100 * CHANCE, color="black", ls="--", lw=0.9, label="chance (16.7%)")
    ax.set_xticks(x, ["input only\n(prompt)", "output only\n(response)", "input + output"])
    ax.set_ylabel("test accuracy (%)")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, ncol=4, loc="upper left", fontsize=9)
    ax.set_title("RQ2: what the classifier is allowed to see")
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig_rq2.png")
    plt.close(fig)


def rq2_bias_assets():
    p = "results/rq2_biased.json"
    if not os.path.exists(p):
        return
    r = load(p)
    bs = r["biases"]
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    for key, lab, col in [("lr", "TF-IDF + LR", MODEL_COLORS["TF-IDF + LR"]), ("cnn", "CNN", MODEL_COLORS["CNN"]),
                          ("majority", "majority class", "#BBBBBB")]:
        m = [100 * np.mean(r[key][str(b)]) for b in bs]
        s = [100 * np.std(r[key][str(b)], ddof=1) for b in bs]
        ax.errorbar(bs, m, yerr=s, marker="o", ms=4, capsize=3, color=col, label=lab)
    ax.axhline(100 * CHANCE, color="black", ls="--", lw=0.9)
    ax.text(1.0, 100 * CHANCE - 4.5, "chance", ha="right", fontsize=8.5)
    ax.set_xlabel("collection bias (share of prompts whose LLM is set by the prompt source)")
    ax.set_ylabel("input-only accuracy (%)")
    ax.set_ylim(0, 80)
    ax.legend(frameon=False)
    ax.set_title("RQ2: a biased collection lets the prompt leak the label")
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig_rq2_bias.png")
    plt.close(fig)
    lines = []
    for b in bs:
        lines.append(f"{b:.2f} & {fmt_ms(r['lr'][str(b)])} & {fmt_ms(r['cnn'][str(b)])} & "
                     f"{fmt_ms(r['majority'][str(b)])} \\\\")
    with open(f"{TAB}/rq2_bias.tex", "w") as fh:
        fh.write("\n".join(lines))
    num("BiasZeroLr", pct(np.mean(r["lr"]["0.0"])))
    num("BiasOneLr", pct(np.mean(r["lr"]["1.0"])))
    num("BiasZeroCnn", pct(np.mean(r["cnn"]["0.0"])))
    num("BiasOneCnn", pct(np.mean(r["cnn"]["1.0"])))
    num("BiasOneMaj", pct(np.mean(r["majority"]["1.0"])))


# ----------------------------------------------------------------------------- RQ3
def rq3_bootstrap(sel, n_boot=4000):
    """95% bootstrap interval (resampling target prompts) for control minus cross-task accuracy, seeds pooled."""
    df = load_data()
    tgt = sel[0]["target_task"]
    target = df[df["task_type"] == tgt]
    cross_pid = target["prompt_id"].values
    per_prompt = {}
    for r in sel:
        ids = np.array(sorted(target["prompt_id"].unique()))
        np.random.default_rng(r["seed"]).shuffle(ids)
        halves = [ids[: len(ids) // 2], ids[len(ids) // 2 :]]
        ctrl_pid = np.concatenate([df[df["prompt_id"].isin(halves[1])]["prompt_id"].values,
                                   df[df["prompt_id"].isin(halves[0])]["prompt_id"].values])
        gold_c = np.concatenate([df[df["prompt_id"].isin(halves[1])]["label"].values,
                                 df[df["prompt_id"].isin(halves[0])]["label"].values])
        assert list(gold_c) == r["control"]["gold"] and list(target["label"].values) == r["cross"]["target"]["gold"]
        ok_x = np.array(r["cross"]["target"]["pred"]) == np.array(r["cross"]["target"]["gold"])
        ok_c = np.array(r["control"]["pred"]) == np.array(r["control"]["gold"])
        for pid in np.unique(cross_pid):
            d = per_prompt.setdefault(pid, [])
            d.append(ok_c[ctrl_pid == pid].mean() - ok_x[cross_pid == pid].mean())
    diffs = np.array([np.mean(v) for v in per_prompt.values()])
    rng = np.random.default_rng(0)
    boots = [diffs[rng.integers(len(diffs), size=len(diffs))].mean() for _ in range(n_boot)]
    return np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def rq3_assets():
    files = sorted(glob.glob("results/rq3/*.json"))
    if not files:
        return
    rr = [load(p) for p in files]
    rows, plot = [], {}
    for shift in ["writing", "coding"]:
        for model in ["cnn", "lstm"]:
            sel = [r for r in rr if r["shift"] == shift and r["model"] == model]
            if not sel:
                continue
            ctrl = [r["control"]["accuracy"] for r in sel]
            cross = [r["cross"]["target"]["accuracy"] for r in sel]
            src = [r["cross"]["source_test"]["accuracy"] for r in sel]
            drop = [c - x for c, x in zip(ctrl, cross)]
            lo, hi = rq3_bootstrap(sel)
            lab = "CNN" if model == "cnn" else "BiLSTM"
            rows.append(f"{shift.capitalize()} shift & {lab} ({len(sel)}) & {fmt_ms(src)} & {fmt_ms(ctrl)} & "
                        f"{fmt_ms(cross)} & {signed(100 * np.mean(drop), True)} [{signed(100 * lo, True)}, {signed(100 * hi, True)}] \\\\")
            plot[(shift, lab)] = (ctrl, cross, sel)
            key = shift.capitalize() + ("Cnn" if model == "cnn" else "Lstm")
            num(f"RqThree{key}Ctrl", pct(np.mean(ctrl)))
            num(f"RqThree{key}Cross", pct(np.mean(cross)))
            num(f"RqThree{key}Src", pct(np.mean(src)))
            num(f"RqThree{key}Drop", f"{100 * np.mean(drop):.1f}")
            num(f"RqThree{key}NTarget", str(sel[0]["n_target_prompts"]))
            num(f"RqThree{key}CiLo", signed(100 * lo))
            num(f"RqThree{key}CiHi", signed(100 * hi))
    with open(f"{TAB}/rq3.tex", "w") as fh:
        fh.write("\n".join(rows))

    # per-family recall on the target task (CNN, all seeds pooled): control vs cross
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.2), sharey=True)
    for ax, shift in zip(axes, ["writing", "coding"]):
        if ("%s" % shift, "CNN") not in plot:
            continue
        _, _, sel = plot[(shift, "CNN")]

        def recall(gold, pred):
            gold, pred = np.array(gold), np.array(pred)
            return [float((pred[gold == k] == k).mean()) for k in range(len(FAMILIES))]

        rc = recall(sum([r["control"]["gold"] for r in sel], []), sum([r["control"]["pred"] for r in sel], []))
        rx = recall(sum([r["cross"]["target"]["gold"] for r in sel], []),
                    sum([r["cross"]["target"]["pred"] for r in sel], []))
        for k, fam in enumerate(FAMILIES):
            num(f"RecallCtrl{shift.capitalize()}{fam}", f"{100 * rc[k]:.0f}")
            num(f"RecallCross{shift.capitalize()}{fam}", f"{100 * rx[k]:.0f}")
        x = np.arange(len(FAMILIES))
        ax.bar(x - 0.2, [100 * v for v in rc], 0.4, color="#9ECAE1", label="control (saw the task)")
        ax.bar(x + 0.2, [100 * v for v in rx], 0.4, color="#08519C", label="cross-task (never saw it)")
        ax.set_xticks(x, FAMILIES, rotation=30)
        src_names = {"writing": "train: coding+math+QA, test: writing", "coding": "train: QA, test: coding"}[shift]
        ax.set_title(f"CNN, {src_names}", fontsize=9.5)
        ax.axhline(100 * CHANCE, color="black", ls="--", lw=0.8)
    axes[0].set_ylabel("recall on target task (%)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=9, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(f"{FIG}/fig_rq3.png")
    plt.close(fig)


# ----------------------------------------------------------------------------- RQ4
def rq4_assets():
    feats = load("results/rq4_features.json")
    means = pd.read_csv("results/rq4_feature_means.csv", index_col=0).reindex(FAMILIES)
    shares = pd.read_csv("results/rq4_feature_shares.csv", index_col=0).reindex(FAMILIES)
    show = [(shares, "bold", "% of answers using bold (**...**)", 100), (shares, "headers", "% using markdown headers (#)", 100),
            (shares, "bullets", "% using bullet lists", 100), (shares, "numbered", "% using numbered lists", 100),
            (means, "n_words", "words per answer (mean)", 1), (means, "words_per_sentence", "words per sentence", 1),
            (means, "dash_per100", "dashes per 100 words", 1), (means, "exclam_per100", "'!' per 100 words", 1)]
    fig, axes = plt.subplots(2, 4, figsize=(10, 4.7))
    for ax, (tab, col, title, scale) in zip(axes.ravel(), show):
        ax.bar(range(len(FAMILIES)), scale * tab[col].values, color=[FAM_COLORS[f] for f in FAMILIES])
        ax.set_xticks(range(len(FAMILIES)), FAMILIES, rotation=55, fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.tick_params(axis="y", labelsize=8)
    fig.suptitle("RQ4: style differences between LLM families (all 804 answers per family)", fontsize=10.5,
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig_rq4_style.png")
    plt.close(fig)

    lines = []
    for f in FAMILIES:
        cells = [f"{means.loc[f, 'n_words']:.0f}", f"{means.loc[f, 'words_per_sentence']:.1f}",
                 f"{means.loc[f, 'mattr50']:.3f}"]
        cells += [f"{100 * shares.loc[f, c]:.0f}" for c in ["bold", "headers", "bullets", "numbered"]]
        cells += [f"{means.loc[f, 'dash_per100']:.2f}", f"{means.loc[f, 'exclam_per100']:.2f}"]
        lines.append(f"{f} & " + " & ".join(cells) + " \\\\")
        for c in ["bold", "headers", "bullets", "numbered"]:
            num(f"Share{c.capitalize()}{f}", f"{100 * shares.loc[f, c]:.0f}")
    with open(f"{TAB}/rq4_means.tex", "w") as fh:
        fh.write("\n".join(lines))
    lines = [f"{k} & {v['n_features']} & {pct(v['accuracy'])} & {pct(v['macro_f1'])} \\\\"
             for k, v in feats["feature_classifiers"].items()]
    with open(f"{TAB}/rq4_featclf.tex", "w") as fh:
        fh.write("\n".join(lines))
    for k, v in feats["feature_classifiers"].items():
        num("Feat" + "".join(w.capitalize() for w in k.split()), pct(v["accuracy"]))
    # openers
    lines = []
    for f in FAMILIES:
        tot = 804
        o = ", ".join(f"\\textit{{{w}}} ({100 * c / tot:.0f}\\%)" for w, c in feats["openers"][f][:4])
        lines.append(f"{f} & {o} \\\\")
    with open(f"{TAB}/rq4_openers.tex", "w") as fh:
        fh.write("\n".join(lines))

    mk = pd.read_csv("results/rq4_markers.csv", index_col=0).reindex(FAMILIES)
    nice = {"curly apostrophe (’)": "uses the curly apostrophe (\\textquoteright)",
            "bullets written as '* '": "writes bullets as \\texttt{*}",
            "bullets written as '- '": "writes bullets as \\texttt{-}",
            "starts with Sure/Certainly/Absolutely/Of course": "opens with \\emph{Sure / Certainly / Absolutely / Of course}",
            "starts with Here's/Here is/Here are": "opens with \\emph{Here's / Here is / Here are}",
            "starts with What a/What an": "opens with \\emph{What a / What an}"}
    lines = []
    for col in mk.columns:
        lines.append(f"{nice[col]} & " + " & ".join(f"{100 * v:.0f}" for v in mk[col].values) + " \\\\")
    with open(f"{TAB}/rq4_markers.tex", "w") as fh:
        fh.write("\n".join(lines))
    num("CurlyGpt", f"{100 * mk.loc['GPT'].iloc[0]:.0f}")
    num("CurlyQwen", f"{100 * mk.loc['Qwen'].iloc[0]:.0f}")

    # top n-grams from the TF-IDF model
    base = load("results/baselines.json")
    esc = lambda s: (s.replace("\\", "\\textbackslash{}").replace("&", "\\&").replace("%", "\\%").replace("#", "\\#")
                     .replace("_", "\\_").replace("{", "\\{").replace("}", "\\}").replace("$", "\\$")
                     .replace("^", "\\^{}").replace("~", "\\~{}").replace("<nl>", "\\textlangle nl\\textrangle")
                     .replace("<cut>", "\\textlangle cut\\textrangle").replace("<", "\\textless{}")
                     .replace(">", "\\textgreater{}"))
    import re as _re
    emoji = _re.compile("[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F]+")
    lines = []
    for f in FAMILIES:
        grams = [emoji.sub("[emoji]", g) for g, _ in base["top_ngrams_output"][f][:12]]
        lines.append(f"{f} & " + " \\textbar{} ".join(f"\\texttt{{{esc(g)}}}" for g in grams) + " \\\\")
    with open(f"{TAB}/rq4_ngrams.tex", "w") as fh:
        fh.write("\n".join(lines))

    p = "results/rq4_ablation.json"
    if not os.path.exists(p):
        return
    ab = load(p)
    names = [("original", "none (full response)"), ("no_format", "remove markdown + line breaks"),
             ("words_only", "words only (no punctuation/format)"), ("structure_only", "structure only (words $\\to$ \\textlangle w\\textrangle)"),
             ("opening_only", "first 30 tokens only"), ("middle_only", "tokens 50--150 only")]
    lines, plot = [], []
    for key, lab in names:
        if key not in ab["cnn"]:
            continue
        tt = [x["test_time"] for x in ab["cnn"][key]]
        rt = [x["retrained"] for x in ab["cnn"][key]]
        lr = ab["tfidf_lr"].get(key, {}).get("accuracy")
        lines.append(f"{lab} & {fmt_ms(tt) if key != 'original' else '--'} & {fmt_ms(rt)} & "
                     f"{pct(lr) if lr is not None else '--'} \\\\")
        plot.append((key, lab, mean_std(tt), mean_std(rt)))
        k = "".join(w.capitalize() for w in key.split("_"))
        num(f"Abl{k}Test", pct(np.mean(tt)))
        num(f"Abl{k}Retrain", pct(np.mean(rt)))
        if lr is not None:
            num(f"Abl{k}Lr", pct(lr))
    num("AblSeeds", str(len(ab["cnn"]["original"])))
    with open(f"{TAB}/rq4_ablation.tex", "w") as fh:
        fh.write("\n".join(lines))

    fig, ax = plt.subplots(figsize=(8.2, 3.3))
    keys = [p_ for p_ in plot if p_[0] != "original"]
    x = np.arange(len(keys))
    ax.bar(x - 0.2, [100 * p_[2][0] for p_ in keys], 0.4, yerr=[100 * p_[2][1] for p_ in keys], capsize=3,
           color="#9ECAE1", label="test-time (normal CNN, ablated test text)")
    ax.bar(x + 0.2, [100 * p_[3][0] for p_ in keys], 0.4, yerr=[100 * p_[3][1] for p_ in keys], capsize=3,
           color="#08519C", label="retrained on ablated text")
    orig = plot[0][3][0]
    ax.axhline(100 * orig, color="#08519C", ls=":", lw=1.2, label=f"full response ({100 * orig:.1f}%)")
    ax.axhline(100 * CHANCE, color="black", ls="--", lw=0.9, label="chance (16.7%)")
    short = {"no_format": "no markdown /\nline breaks", "words_only": "words only", "structure_only": "structure only\n(no words)",
             "opening_only": "first 30\ntokens", "middle_only": "middle only\n(tokens 50-150)"}
    ax.set_xticks(x, [short[p_[0]] for p_ in keys], fontsize=9)
    ax.set_ylabel("CNN test accuracy (%)")
    ax.set_ylim(0, 95)
    ax.legend(frameon=False, fontsize=8.5, loc="upper center", ncol=2)
    ax.set_title("RQ4: removing one kind of signal at a time")
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig_rq4_ablation.png")
    plt.close(fig)


def error_analysis_assets():
    """Where do the RQ1 models fail? Accuracy by response length and by task type, plus a simple vote ensemble."""
    cnn, lstm = runs("cnn", "output"), runs("lstm", "output")
    if not cnn or not lstm:
        return
    df = load_data()
    te = df[df["split"] == "test"].reset_index(drop=True)
    gold = te["label"].values
    assert list(gold) == cnn[0]["tests"]["test"]["gold"]
    base = load("results/baselines.json")["tfidf_lr"]["output"]
    preds = {"CNN": [np.array(r["tests"]["test"]["pred"]) for r in cnn],
             "BiLSTM": [np.array(r["tests"]["test"]["pred"]) for r in lstm],
             "TF-IDF + LR": [np.array(base["pred"])]}
    words = te["llm_output"].str.split().str.len().values
    bins = [0, 100, 200, 300, 10_000]
    labels = ["under 100", "100--199", "200--299", "300+"]
    te["len_bin"] = pd.cut(words, bins=bins, right=False, labels=labels)
    lines = []
    for b in labels:
        m = (te["len_bin"] == b).values
        cells = [pct(np.mean([(p[m] == gold[m]).mean() for p in preds[k]])) for k in preds]
        lines.append(f"{b} words & {m.sum()} & " + " & ".join(cells) + " \\\\")
    lines.append("\\midrule")
    names = {"qa_advice": "QA / advice", "writing": "writing", "coding": "coding", "math": "math"}
    for t in ["qa_advice", "writing", "coding", "math"]:
        m = (te["task_type"] == t).values
        cells = [pct(np.mean([(p[m] == gold[m]).mean() for p in preds[k]])) for k in preds]
        lines.append(f"{names[t]} prompts & {m.sum()} & " + " & ".join(cells) + " \\\\")
    with open(f"{TAB}/rq1_breakdown.tex", "w") as fh:
        fh.write("\n".join(lines))
    short = (te["len_bin"] == "under 100").values
    num("ShortN", str(short.sum()))
    num("ShortCnnAcc", pct(np.mean([(p[short] == gold[short]).mean() for p in preds["CNN"]])))
    long_ = (te["len_bin"] == "300+").values
    num("LongCnnAcc", pct(np.mean([(p[long_] == gold[long_]).mean() for p in preds["CNN"]])))

    # majority vote over every trained model (3 CNN + 3 BiLSTM seeds + TF-IDF): ties -> CNN seed 1
    allp = np.stack(preds["CNN"] + preds["BiLSTM"] + preds["TF-IDF + LR"])
    vote = []
    for j in range(allp.shape[1]):
        c = np.bincount(allp[:, j], minlength=len(FAMILIES))
        best = np.flatnonzero(c == c.max())
        vote.append(allp[0, j] if allp[0, j] in best else best[0])
    vote = np.array(vote)
    num("EnsAcc", pct((vote == gold).mean()))
    best_single = max(np.mean([(p == gold).mean() for p in preds[k]]) for k in preds)
    num("EnsGain", f"{100 * ((vote == gold).mean() - best_single):.1f}")
    best_run = max((p == gold).mean() for k in preds for p in preds[k])
    num("EnsGainRun", f"{100 * ((vote == gold).mean() - best_run):.1f}")
    num("BestRunAcc", pct(best_run))
    pooled_g = np.concatenate([gold] * len(preds["CNN"]))
    pooled_p = np.concatenate(preds["CNN"])
    for k, fam in enumerate(FAMILIES):
        num(f"RecallCnn{fam}", pct((pooled_p[pooled_g == k] == k).mean()))
    for t, key in [("writing", "Writing"), ("qa_advice", "Qa")]:
        m = (te["task_type"] == t).values
        num(f"{key}CnnAcc", pct(np.mean([(p[m] == gold[m]).mean() for p in preds["CNN"]])))
        num(f"{key}LrAcc", pct(np.mean([(p[m] == gold[m]).mean() for p in preds["TF-IDF + LR"]])))
    num("ShortLrAcc", pct(np.mean([(p[short] == gold[short]).mean() for p in preds["TF-IDF + LR"]])))
    num("EnsN", str(allp.shape[0]))
    agree = (preds["CNN"][0] == preds["BiLSTM"][0])
    num("AgreeRate", pct(agree.mean()))
    num("AgreeAcc", pct((preds["CNN"][0][agree] == gold[agree]).mean()))
    num("DisagreeAcc", pct((preds["CNN"][0][~agree] == gold[~agree]).mean()))


def misc_numbers():
    df = load_data()
    num("MedianWords", f"{df['llm_output'].str.split().str.len().median():.0f}")
    from llmid import tokenize
    L = df["llm_output"].map(lambda t: len(tokenize(t)))
    num("ShareUnderOneFifty", f"{100 * (L < 150).mean():.0f}")
    num("ShareUnderFifty", f"{100 * (L < 50).mean():.0f}")
    by = (L < 150).groupby(df["llm_family"]).mean()
    num("ShareUnderOneFiftyDeepSeek", f"{100 * by['DeepSeek']:.0f}")
    num("ShareUnderOneFiftyClaude", f"{100 * by['Claude']:.0f}")
    for kind, key in [("cnn", "Cnn"), ("lstm", "Lstm")]:
        rr = runs(kind, "output")
        if rr:
            per = [r["train_seconds"] / len(r["history"]) for r in rr]
            num(f"{key}SecEpochLo", f"{min(per):.0f}")
            num(f"{key}SecEpochHi", f"{max(per):.0f}")
            num(f"{key}BestEpochs", ", ".join(str(r["best_epoch"]) for r in rr))
    base = load("results/baselines.json")
    if "tfidf_lr_row_split_output" in base:
        num("RowSplitLr", pct(np.mean(base["tfidf_lr_row_split_output"])))
        num("RowSplitGap", f"{100 * (base['tfidf_lr']['output']['accuracy'] - np.mean(base['tfidf_lr_row_split_output'])):.0f}")
    mk = pd.read_csv("results/rq4_markers.csv", index_col=0)
    dash_col = [c for c in mk.columns if c.startswith("bullets written as '-")][0]
    num("DashBulletLlama", f"{100 * mk.loc['Llama', dash_col]:.1f}")


def cnn_filter_assets():
    p = "results/rq4_cnn_filters.json"
    if not os.path.exists(p):
        return
    r = load(p)
    esc = lambda s: (s.replace("\\", "\\textbackslash{}").replace("&", "\\&").replace("%", "\\%").replace("#", "\\#")
                     .replace("_", "\\_").replace("{", "\\{").replace("}", "\\}").replace("$", "\\$")
                     .replace("^", "\\^{}").replace("~", "\\~{}").replace("<nl>", "\\textlangle nl\\textrangle{}")
                     .replace("<cut>", "\\textlangle cut\\textrangle{}").replace("<unk>", "\\textlangle unk\\textrangle{}")
                     .replace("<", "\\textless{}").replace(">", "\\textgreater{}").replace("’", "\\textquoteright{}").replace("`", "\\textasciigrave{}"))
    import re as _re
    emoji = _re.compile("[\\U0001F000-\\U0001FAFF\\u2600-\\u27BF\\uFE0F]+")
    lines = []
    for fam in FAMILIES:
        wins = []
        for fl in r["filters"][fam]:
            for w, c in fl["top_windows"][:1]:
                w = emoji.sub("[emoji]", w)
                if w not in wins:
                    wins.append(w)
        lines.append(f"{fam} & " + " \\textbar{} ".join(f"\\texttt{{{esc(w)}}}" for w in wins[:5]) + " \\\\")
    with open(f"{TAB}/rq4_filters.tex", "w") as fh:
        fh.write("\n".join(lines))
    num("FilterCnnAcc", pct(r["test_accuracy"]))


def write_numbers():
    with open(os.path.join("report", "numbers.tex"), "w") as fh:
        for k, v in sorted(NUMBERS.items()):
            fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")


if __name__ == "__main__":
    dataset_assets()
    rq1_rq2_assets()
    rq2_bias_assets()
    rq3_assets()
    rq4_assets()
    error_analysis_assets()
    misc_numbers()
    cnn_filter_assets()
    write_numbers()
    print("figures:", sorted(os.listdir(FIG)))
    print("tables:", sorted(os.listdir(TAB)))
