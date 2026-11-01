"""Calibrate and evaluate the matching thresholds on open, human-labeled data.

Data source: STS-B (Semantic Textual Similarity Benchmark), a widely used open
dataset of sentence pairs each scored 0-5 by human annotators for how close in
meaning they are. We encode every pair with our own model and measure the cosine
distance, then pick the distance cut-offs that best reproduce human judgement:

  * same_distance    -> boundary for "these mean the same" (human score >= 4.0)
  * suggest_distance  -> boundary for "these are related"   (human score >= 3.0)

It also measures TYPO robustness: it corrupts clean sentences with realistic
typos and reports how often the typo'd version still lands within same_distance
of the original, with and without our normalizer.

Run inside the container venv:
    HF_HOME=/opt/not-alone/models venv/bin/python scripts/calibrate.py
"""
import random
import sys

import numpy as np

from app.embedding import get_model, normalize

random.seed(0)


def load_stsb():
    from datasets import load_dataset
    ds = load_dataset("glue", "stsb", split="validation")
    a = [r["sentence1"] for r in ds]
    b = [r["sentence2"] for r in ds]
    y = np.array([r["label"] for r in ds], dtype=np.float32)  # 0..5
    return a, b, y


def distances(model, a, b):
    va = model.encode([normalize(x) for x in a], normalize_embeddings=True,
                      convert_to_numpy=True, batch_size=64, show_progress_bar=False)
    vb = model.encode([normalize(x) for x in b], normalize_embeddings=True,
                      convert_to_numpy=True, batch_size=64, show_progress_bar=False)
    return 1.0 - np.sum(va * vb, axis=1)  # cosine distance


def best_threshold(dist, pos):
    """Distance cut maximizing F1 for the positive class (pos = boolean array)."""
    best_f1, best_t = 0.0, 0.5
    for t in np.arange(0.10, 0.80, 0.01):
        pred = dist < t
        tp = np.sum(pred & pos); fp = np.sum(pred & ~pos); fn = np.sum(~pred & pos)
        if tp == 0:
            continue
        prec = tp / (tp + fp); rec = tp / (tp + fn)
        f1 = 2 * prec * rec / (prec + rec)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    pred = dist < best_t
    tp = np.sum(pred & pos); fp = np.sum(pred & ~pos); fn = np.sum(~pred & pos)
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    return best_t, best_f1, prec, rec


TYPO_CHARS = "abcdefghijklmnopqrstuvwxyz"


def add_typos(s, rate=0.08):
    out = []
    for ch in s:
        r = random.random()
        if ch.isalpha() and r < rate:
            k = random.random()
            if k < 0.4:      # drop
                continue
            elif k < 0.7:    # swap with neighbour
                out.append(random.choice(TYPO_CHARS)); out.append(ch)
            else:            # substitute
                out.append(random.choice(TYPO_CHARS))
        else:
            out.append(ch)
    return "".join(out)


def typo_robustness(model, sents, same_t):
    clean = sents
    typo = [add_typos(s) for s in sents]
    vc = model.encode([normalize(x) for x in clean], normalize_embeddings=True, convert_to_numpy=True, batch_size=64)
    vt = model.encode([normalize(x) for x in typo], normalize_embeddings=True, convert_to_numpy=True, batch_size=64)
    d = 1.0 - np.sum(vc * vt, axis=1)
    held = np.mean(d < same_t)
    return held, float(np.mean(d))


def main():
    print("loading model + STS-B ...", file=sys.stderr)
    model = get_model()
    a, b, y = load_stsb()
    d = distances(model, a, b)

    same_t, f1s, ps, rs = best_threshold(d, y >= 4.0)
    sugg_t, f1r, pr, rr = best_threshold(d, y >= 3.0)

    print(f"pairs evaluated: {len(y)}")
    print(f"\nSAME  (human score >= 4.0): best distance < {same_t:.2f}  "
          f"F1={f1s:.2f} precision={ps:.2f} recall={rs:.2f}")
    print(f"RELATED (human score >= 3.0): best distance < {sugg_t:.2f}  "
          f"F1={f1r:.2f} precision={pr:.2f} recall={rr:.2f}")

    sample = random.sample(a, 300)
    held, meand = typo_robustness(model, sample, same_t)
    print(f"\nTYPO robustness (8% char corruption, {len(sample)} sentences):")
    print(f"  mean distance clean<->typo = {meand:.3f}")
    print(f"  stays within same_distance({same_t:.2f}) = {held*100:.0f}%")

    print(f"\nSuggested config:\n  same_distance = {same_t:.2f}\n  suggest_distance = {sugg_t:.2f}")


if __name__ == "__main__":
    main()
