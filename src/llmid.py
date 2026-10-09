"""Shared code: tokenizer, vocabulary, CNN and BiLSTM models, training and evaluation.

Every experiment script in src/ imports from this file.
"""
import json
import os
import random
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

FAMILIES = ["GPT", "Claude", "Gemini", "Llama", "Qwen", "DeepSeek"]
DATA_CSV = os.path.join("data", "llm_responses.csv")
torch.set_num_threads(int(os.environ.get("LLMID_THREADS", os.cpu_count() or 1)))


# ----------------------------------------------------------------------------- text
# Words, numbers, single punctuation marks, and formatting marks (newline, **, #, ```)
# are all kept as tokens, because formatting is part of an LLM's style.
TOKEN_RE = re.compile(r"\n|\*\*|#{1,6}|```|`|\w+(?:'\w+)?|[^\w\s]")


def tokenize(text: str):
    return ["<nl>" if t == "\n" else t.lower() for t in TOKEN_RE.findall(text or "")]


@dataclass
class TextConfig:
    head: int = 200         # keep the first 200 output tokens ...
    tail: int = 50          # ... and the last 50 (openings and endings both carry style)
    input_len: int = 64     # prompt tokens used when the prompt is part of the input
    input_only_len: int = 128
    min_freq: int = 2
    max_vocab: int = 30000


def head_tail(tokens, head, tail):
    if len(tokens) <= head + tail:
        return tokens
    return tokens[:head] + ["<cut>"] + tokens[-tail:]


def make_tokens(inp: str, out: str, mode: str, cfg: TextConfig):
    """mode: 'output' (RQ1), 'input' (prompt only) or 'input_output' (prompt + response)."""
    if mode == "output":
        return head_tail(tokenize(out), cfg.head, cfg.tail)
    if mode == "input":
        return tokenize(inp)[: cfg.input_only_len]
    if mode == "input_output":
        return tokenize(inp)[: cfg.input_len] + ["<sep>"] + head_tail(tokenize(out), cfg.head, cfg.tail)
    raise ValueError(mode)


class Vocab:
    def __init__(self, token_lists, min_freq=2, max_size=30000):
        counts = Counter(t for toks in token_lists for t in toks)
        keep = [t for t, c in counts.most_common(max_size) if c >= min_freq]
        self.itos = ["<pad>", "<unk>"] + [t for t in keep if t not in ("<pad>", "<unk>")]
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, toks):
        ids = [self.stoi.get(t, 1) for t in toks]
        return ids if ids else [1]


# ----------------------------------------------------------------------------- data
def load_data(path=DATA_CSV):
    df = pd.read_csv(path, keep_default_na=False)
    df["label"] = df["llm_family"].map({f: i for i, f in enumerate(FAMILIES)})
    return df


def encode_frames(train_df, eval_dfs, mode, cfg: TextConfig, text_fn=None):
    """Tokenize, build the vocabulary on the TRAIN rows only, and encode every frame.

    text_fn (optional) transforms the raw output text first (used for ablations)."""
    def toks(df):
        outs = df["llm_output"] if text_fn is None else df["llm_output"].map(text_fn)
        return [make_tokens(i, o, mode, cfg) for i, o in zip(df["llm_input"], outs)]

    train_toks = toks(train_df)
    vocab = Vocab(train_toks, cfg.min_freq, cfg.max_vocab)
    enc = lambda tl, df: list(zip([vocab.encode(t) for t in tl], df["label"].tolist()))
    train = enc(train_toks, train_df)
    evals = [enc(toks(d), d) for d in eval_dfs]
    return vocab, train, evals


def batches(data, batch_size, shuffle, rng=None, min_len=5):
    idx = list(range(len(data)))
    if shuffle:
        rng.shuffle(idx)
    for s in range(0, len(idx), batch_size):
        chunk = [data[i] for i in idx[s : s + batch_size]]
        lengths = [len(x) for x, _ in chunk]
        T = max(max(lengths), min_len)
        x = torch.zeros(len(chunk), T, dtype=torch.long)
        for r, (ids, _) in enumerate(chunk):
            x[r, : len(ids)] = torch.tensor(ids)
        yield x, torch.tensor(lengths), torch.tensor([y for _, y in chunk])


# ----------------------------------------------------------------------------- models
class TextCNN(nn.Module):
    """Kim (2014)-style CNN: embeddings -> parallel 1D convolutions (widths 3/4/5) -> max-pool -> linear."""

    def __init__(self, vocab_size, n_classes, emb_dim=128, n_filters=100, kernel_sizes=(3, 4, 5),
                 emb_dropout=0.2, dropout=0.5):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(emb_dim, n_filters, k) for k in kernel_sizes])
        self.emb_drop = nn.Dropout(emb_dropout)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(kernel_sizes), n_classes)

    def forward(self, x, lengths):
        e = self.emb_drop(self.emb(x)).transpose(1, 2)            # (B, E, T)
        pooled = []
        for conv in self.convs:
            k = conv.kernel_size[0]
            h = torch.relu(conv(e))                               # (B, F, T-k+1)
            pos = torch.arange(h.size(2), device=x.device)[None, :]
            valid = (pos + k <= lengths[:, None]) | (pos == 0)    # ignore windows that cover padding
            pooled.append((h * valid[:, None, :]).max(dim=2).values)
        return self.fc(self.drop(torch.cat(pooled, dim=1)))


class BiLSTMClassifier(nn.Module):
    """Embeddings -> 1-layer bidirectional LSTM -> mean+max pooling over time -> linear."""

    def __init__(self, vocab_size, n_classes, emb_dim=128, hidden=64, emb_dropout=0.2, dropout=0.5):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden, batch_first=True, bidirectional=True)
        self.emb_drop = nn.Dropout(emb_dropout)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(4 * hidden, n_classes)

    def forward(self, x, lengths):
        e = self.emb_drop(self.emb(x))
        packed = pack_padded_sequence(e, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=x.size(1))  # (B, T, 2H)
        mask = (torch.arange(x.size(1))[None, :] < lengths[:, None]).unsqueeze(-1)
        mean = (out * mask).sum(1) / lengths[:, None].clamp(min=1)
        mx = out.masked_fill(~mask, -1e4).max(1).values
        return self.fc(self.drop(torch.cat([mean, mx], dim=1)))


def build_model(kind, vocab_size, n_classes):
    if kind == "cnn":
        return TextCNN(vocab_size, n_classes)
    if kind == "lstm":
        return BiLSTMClassifier(vocab_size, n_classes)
    raise ValueError(kind)


# ----------------------------------------------------------------------------- training
@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 32
    max_epochs: int = 15
    patience: int = 3       # early stopping on validation macro-F1
    clip: float = 1.0


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def predict(model, data, batch_size=128):
    model.eval()
    preds, gold = [], []
    for x, lengths, y in batches(data, batch_size, shuffle=False):
        preds.append(model(x, lengths).argmax(1))
        gold.append(y)
    return torch.cat(gold).numpy(), torch.cat(preds).numpy()


def metrics(gold, pred, n_classes=len(FAMILIES)):
    return {
        "accuracy": float(accuracy_score(gold, pred)),
        "macro_f1": float(f1_score(gold, pred, average="macro", labels=list(range(n_classes)), zero_division=0)),
        "per_class_f1": f1_score(gold, pred, average=None, labels=list(range(n_classes)), zero_division=0).tolist(),
        "confusion": confusion_matrix(gold, pred, labels=list(range(n_classes))).tolist(),
    }


def train_and_eval(kind, vocab, train, val, tests: dict, seed, tcfg=TrainConfig(), n_classes=len(FAMILIES),
                   verbose=False):
    """Train with early stopping on validation macro-F1, then evaluate the best checkpoint on each test set."""
    set_seed(seed)
    rng = random.Random(seed)
    model = build_model(kind, len(vocab), n_classes)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    best_f1, best_state, best_epoch, bad = -1.0, None, 0, 0
    history = []
    t0 = time.time()
    for epoch in range(1, tcfg.max_epochs + 1):
        model.train()
        total, n = 0.0, 0
        for x, lengths, y in batches(train, tcfg.batch_size, shuffle=True, rng=rng):
            opt.zero_grad()
            loss = loss_fn(model(x, lengths), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), tcfg.clip)
            opt.step()
            total += loss.item() * len(y)
            n += len(y)
        g, p = predict(model, val)
        vm = metrics(g, p, n_classes)
        history.append({"epoch": epoch, "train_loss": total / n, "val_acc": vm["accuracy"], "val_f1": vm["macro_f1"]})
        if verbose:
            print(f"  [{kind} seed{seed}] epoch {epoch:2d} loss {total / n:.3f} val_acc {vm['accuracy']:.3f} "
                  f"val_f1 {vm['macro_f1']:.3f} ({time.time() - t0:.0f}s)", flush=True)
        if vm["macro_f1"] > best_f1:
            best_f1, best_epoch, bad = vm["macro_f1"], epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= tcfg.patience:
                break
    model.load_state_dict(best_state)
    out = {"model": kind, "seed": seed, "best_epoch": best_epoch, "val_macro_f1": best_f1,
           "train_seconds": round(time.time() - t0, 1), "n_params": sum(p.numel() for p in model.parameters()),
           "vocab_size": len(vocab), "history": history, "tests": {}}
    for name, data in tests.items():
        g, p = predict(model, data)
        out["tests"][name] = metrics(g, p, n_classes)
        out["tests"][name]["pred"] = p.tolist()
        out["tests"][name]["gold"] = g.tolist()
    return out, model


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)


def config_dict():
    return {"text": asdict(TextConfig()), "train": asdict(TrainConfig())}
