"""RQ4, part 3: what does the CNN itself look at?

Each CNN filter is a detector for a short window of 3, 4 or 5 tokens. This script re-trains the RQ1 CNN (output only,
seed 1, single thread so it matches the RQ1 run), then for every family takes the 3 filters with the largest weight
toward that family and lists the text windows that switch each filter on most strongly in the training answers.

Run:  LLMID_THREADS=1 python src/rq4_cnn_filters.py
Output: results/rq4_cnn_filters.json
"""
import collections

import numpy as np
import torch

from llmid import FAMILIES, TextConfig, TrainConfig, batches, encode_frames, load_data, save_json, train_and_eval


def main(seed=1, top_filters=3, top_docs=40):
    df = load_data()
    tr, va, te = (df[df["split"] == s] for s in ("train", "val", "test"))
    vocab, train, (val, test) = encode_frames(tr, [va, te], "output", TextConfig())
    res, model = train_and_eval("cnn", vocab, train, val, {"test": test}, seed, TrainConfig())
    acc = res["tests"]["test"]["accuracy"]
    print(f"re-trained CNN (seed {seed}) test accuracy {acc:.3f}")
    model.eval()
    weights = model.fc.weight.detach().numpy()            # (6 families, 300 filters)
    n_f = model.convs[0].out_channels
    acts = np.zeros((len(train), n_f * len(model.convs)))
    where = np.zeros_like(acts, dtype=int)
    with torch.no_grad():
        i = 0
        for x, lengths, y in batches(train, 128, shuffle=False):
            e = model.emb(x).transpose(1, 2)
            for c, conv in enumerate(model.convs):
                k = conv.kernel_size[0]
                h = torch.relu(conv(e))
                pos = torch.arange(h.size(2))[None, :]
                h = h * ((pos + k <= lengths[:, None]) | (pos == 0))[:, None, :]
                m, p = h.max(dim=2)
                acts[i : i + len(y), c * n_f : (c + 1) * n_f] = m.numpy()
                where[i : i + len(y), c * n_f : (c + 1) * n_f] = p.numpy()
            i += len(y)

    out = {}
    for k_cls, fam in enumerate(FAMILIES):
        filters = []
        for f in np.argsort(weights[k_cls])[::-1][:top_filters]:
            width = model.convs[f // n_f].kernel_size[0]
            counts = collections.Counter()
            for d in np.argsort(acts[:, f])[::-1][:top_docs]:
                ids = train[d][0]
                p = where[d, f]
                counts[" ".join(vocab.itos[t] for t in ids[p : p + width])] += 1
            filters.append({"filter": int(f), "width": int(width), "weight": round(float(weights[k_cls, f]), 3),
                            "top_windows": counts.most_common(3)})
        out[fam] = filters
        print(fam, "|", " || ".join(" / ".join(w for w, _ in fl["top_windows"][:2]) for fl in filters))
    save_json({"seed": seed, "test_accuracy": acc, "filters": out}, "results/rq4_cnn_filters.json")


if __name__ == "__main__":
    main()
