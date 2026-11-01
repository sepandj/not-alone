"""Nightly self-tuning, no LLM involved. Two jobs:

1. THRESHOLDS. Real thoughts that mean the same thing sit close together, and
   unrelated ones sit far apart, so the distance from each thought to the
   nearest thought by *another person* has a dip between those two groups. The
   lowest point of that dip is the natural "same thought" cut-off for the way
   people here actually write. The tuner moves same_distance toward it, but:
     * only once there are enough real people (MIN_PEOPLE),
     * only inside fixed bounds around the STS-B calibration,
     * at most STEP per night,
     * and only if the result still passes two fixed checks: the hand-labeled
       pairs in scripts/tuning_pairs.json must score at least as well as the
       calibrated defaults, and typo'd versions of real thoughts must still
       match their originals at least TYPO_MIN of the time.
   suggest_distance keeps the calibrated gap above same_distance.
   Every run is recorded in the `tuning` table, applied or not, with the reason.

2. MEANING-GROUPS. Clusters all non-blocked thoughts so that every member is
   within cluster_distance of every other (complete linkage), keeps different
   starter phrasings apart, gives each group a label (its starter phrasing if
   it has one, else the moderated member nearest its centre) and stores the
   group on each thought. Suggestions then merge wordings of one feeling.

Runs in the container as the app user:
    PYTHONPATH=/opt/not-alone/app venv/bin/python scripts/autotune.py [--dry-run]
"""
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sqlalchemy import text as sql

from app.config import settings
from app.db import engine
from app.safety import has_crisis_keywords
from app.tuning import thresholds

SAME_BOUNDS = (0.38, 0.52)
GAP = round(settings.suggest_distance - settings.same_distance, 4)
STEP = 0.01
MIN_PEOPLE = 150
TYPO_MIN = 0.85
ANCHOR_SLACK = 0.02
MAX_CLUSTER = 8000  # beyond this, the rest join their nearest group

PAIRS = json.loads((Path(__file__).parent / "tuning_pairs.json").read_text())


def load():
    with engine.connect() as c:
        rows = c.execute(sql(
            "SELECT id, text, status, is_seed, visitor_hash, embedding::text AS e "
            "FROM thoughts WHERE status <> 'blocked' ORDER BY id DESC")).mappings().all()
    if not rows:
        return [], np.zeros((0, settings.embedding_dim), np.float32)
    emb = np.array([json.loads(r["e"]) for r in rows], dtype=np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
    return rows, emb


# ---- thresholds ---------------------------------------------------------------

def nn_other_person(rows, emb):
    """For each real thought, cosine distance to the nearest real thought
    written by someone else."""
    idx = [i for i, r in enumerate(rows) if not r["is_seed"]]
    if len(idx) < 2:
        return np.array([])
    E = emb[idx]
    who = np.array([rows[i]["visitor_hash"] for i in idx])
    out = np.full(len(idx), np.inf, dtype=np.float32)
    for s in range(0, len(idx), 1024):
        d = 1.0 - E[s:s + 1024] @ E.T
        d[who[s:s + 1024, None] == who[None, :]] = np.inf
        out[s:s + 1024] = d.min(axis=1)
    return out[np.isfinite(out)]


def valley(d):
    """Lowest point of the smoothed distance distribution inside SAME_BOUNDS,
    or None when there is no clear dip."""
    grid = np.arange(0.15, 0.85, 0.005)
    bw = 0.03
    dens = np.exp(-0.5 * ((grid[:, None] - d[None, :]) / bw) ** 2).sum(axis=1)
    lo, hi = SAME_BOUNDS
    band = (grid >= lo) & (grid <= hi)
    i = np.argmin(np.where(band, dens, np.inf))
    # A real dip has higher density on both sides of it.
    left, right = dens[grid < grid[i]].max(initial=0), dens[grid > grid[i]].max(initial=0)
    if dens[i] > 0.6 * min(left, right):
        return None
    return float(grid[i])


def anchor_f1(model, t):
    from app.embedding import normalize
    pairs = [(a, b, 1) for a, b in PAIRS["same"]] + [(a, b, 0) for a, b in PAIRS["different"]]
    A = model.encode([normalize(a) for a, _, _ in pairs], normalize_embeddings=True, show_progress_bar=False)
    B = model.encode([normalize(b) for _, b, _ in pairs], normalize_embeddings=True, show_progress_bar=False)
    dist = 1.0 - np.sum(A * B, axis=1)
    y = np.array([p[2] for p in pairs], bool)
    pred = dist < t
    tp, fp, fn = np.sum(pred & y), np.sum(pred & ~y), np.sum(~pred & y)
    return float(2 * tp / (2 * tp + fp + fn)) if tp else 0.0


def typo_hold(model, texts, t):
    from app.embedding import normalize
    sys.path.insert(0, str(Path(__file__).parent))
    from calibrate import add_typos
    random.seed(1)
    texts = random.sample(texts, min(200, len(texts)))
    if not texts:
        return 1.0
    a = model.encode([normalize(x) for x in texts], normalize_embeddings=True, show_progress_bar=False)
    b = model.encode([normalize(add_typos(x)) for x in texts], normalize_embeddings=True, show_progress_bar=False)
    return float(np.mean(1.0 - np.sum(a * b, axis=1) < t))


def tune(rows, emb):
    cur_same, cur_sugg = thresholds()
    people = len({r["visitor_hash"] for r in rows if not r["is_seed"]})
    report = {"people": people, "current": [cur_same, cur_sugg]}
    if people < MIN_PEOPLE:
        report["skipped"] = f"only {people} people so far, need {MIN_PEOPLE}"
        return cur_same, cur_sugg, False, report

    d = nn_other_person(rows, emb)
    target = valley(d)
    report["nn_median"] = round(float(np.median(d)), 3) if len(d) else None
    report["valley"] = target
    if target is None:
        report["skipped"] = "no clear dip in the distance distribution"
        return cur_same, cur_sugg, False, report

    step = float(np.clip(target - cur_same, -STEP, STEP))
    new_same = round(float(np.clip(cur_same + step, *SAME_BOUNDS)), 3)
    if abs(new_same - cur_same) < 1e-4:
        report["skipped"] = "already at the dip"
        return cur_same, cur_sugg, False, report

    from app.embedding import get_model
    model = get_model()
    f1_new, f1_base = anchor_f1(model, new_same), anchor_f1(model, settings.same_distance)
    real_ok = [r["text"] for r in rows if not r["is_seed"] and r["status"] == "ok"]
    hold = typo_hold(model, real_ok, new_same)
    report.update(anchor_f1=round(f1_new, 3), anchor_f1_default=round(f1_base, 3), typo_hold=round(hold, 3))
    if f1_new < f1_base - ANCHOR_SLACK:
        report["skipped"] = "would match the labeled pairs worse than the defaults"
        return cur_same, cur_sugg, False, report
    if hold < TYPO_MIN:
        report["skipped"] = f"typo'd thoughts would match only {hold:.0%} of the time"
        return cur_same, cur_sugg, False, report
    return new_same, round(new_same + GAP, 3), True, report


# ---- meaning-groups -------------------------------------------------------------

def cluster(rows, emb, cut=settings.cluster_distance):
    n = len(rows)
    if n == 0:
        return np.array([], int)
    core = min(n, MAX_CLUSTER)  # rows are newest first
    if core == 1:
        labels = np.array([1])
    else:
        D = np.clip(1.0 - emb[:core] @ emb[:core].T, 0, 2).astype(np.float64)
        np.fill_diagonal(D, 0)
        Z = linkage(squareform(D, checks=False), method="complete")
        labels = fcluster(Z, t=cut, criterion="distance")
    if n > core:  # older thoughts join the group of their nearest core thought
        rest = []
        for s in range(core, n, 1024):
            sim = emb[s:s + 1024] @ emb[:core].T
            j = sim.argmax(axis=1)
            near = 1.0 - sim[np.arange(len(j)), j] < cut
            rest.append(np.where(near, labels[j], -1))
        rest = np.concatenate(rest)
        labels = labels.copy()
        # stragglers with no near neighbour get their own group each
        extra = np.arange(labels.max() + 1, labels.max() + 1 + np.sum(rest == -1))
        rest[rest == -1] = extra
        labels = np.concatenate([labels, rest])
    return split_seeds(rows, emb, labels)


def split_seeds(rows, emb, labels):
    """Each starter phrasing is its own suggestion: a group holding several
    seeds is split, every other member joining the seed it is closest to."""
    labels = np.asarray(labels).copy()
    nxt = int(labels.max()) + 1
    for l in np.unique(labels):
        ix = np.flatnonzero(labels == l)
        seeds = [i for i in ix if rows[i]["is_seed"]]
        if len(seeds) < 2:
            continue
        near = np.array(seeds)[np.argmax(emb[ix] @ emb[seeds].T, axis=1)]
        for k, s in enumerate(seeds[1:]):
            labels[ix[near == s]] = nxt + k
        nxt += len(seeds) - 1
    return labels


def groups(rows, emb, labels):
    members = defaultdict(list)
    for i, l in enumerate(labels):
        members[int(l)].append(i)
    out = []
    for l, ix in members.items():
        seeds = [rows[i]["text"] for i in ix if rows[i]["is_seed"]]
        label = None
        if seeds:
            label = min(seeds, key=len)
        else:
            ok = [i for i in ix if rows[i]["status"] == "ok" and not has_crisis_keywords(rows[i]["text"])]
            if ok:
                centre = emb[ix].mean(axis=0)
                label = rows[max(ok, key=lambda i: float(emb[i] @ centre))]["text"]
        people = len({rows[i]["visitor_hash"] for i in ix if not rows[i]["is_seed"]})
        out.append((l, label, people, [rows[i]["id"] for i in ix]))
    return out


def save(groups_, same, sugg, applied, report, dry):
    multi = [g for g in groups_ if len(g[3]) > 1]
    report["groups"] = len(groups_)
    report["groups_multi"] = len(multi)
    report["largest"] = max((len(g[3]) for g in groups_), default=0)
    if dry:
        return
    with engine.begin() as c:
        c.execute(sql(
            "INSERT INTO tuning (same_distance, suggest_distance, applied, report) "
            "VALUES (:s, :g, :a, CAST(:r AS jsonb))"),
            {"s": same, "g": sugg, "a": applied, "r": json.dumps(report)})
        # Fresh ids every night, so a thought posted mid-run that copied an old
        # id simply points at nothing (no label) until the next run.
        base = c.execute(sql("SELECT COALESCE(MAX(id), 0) FROM clusters")).scalar() + 1
        c.execute(sql("DELETE FROM clusters"))
        if groups_:
            c.execute(sql("INSERT INTO clusters (id, label, people) VALUES (:id, :label, :people)"),
                      [{"id": base + i, "label": lab, "people": p} for i, (_, lab, p, _) in enumerate(groups_)])
            c.execute(sql("CREATE TEMP TABLE _cl (tid int, cid int) ON COMMIT DROP"))
            c.execute(sql("INSERT INTO _cl VALUES (:t, :c)"),
                      [{"t": tid, "c": base + i} for i, g in enumerate(groups_) for tid in g[3]])
            c.execute(sql("UPDATE thoughts t SET cluster_id = _cl.cid FROM _cl "
                          "WHERE t.id = _cl.tid AND t.cluster_id IS DISTINCT FROM _cl.cid"))
        c.execute(sql("UPDATE thoughts SET cluster_id = NULL WHERE cluster_id IS NOT NULL "
                      "AND cluster_id NOT IN (SELECT id FROM clusters)"))


def main(dry=False):
    rows, emb = load()
    same, sugg, applied, report = tune(rows, emb)
    labels = cluster(rows, emb)
    g = groups(rows, emb, labels)
    save(g, same, sugg, applied, report, dry)
    print(json.dumps({"same": same, "suggest": sugg, "applied": applied, **report}, ensure_ascii=False))


if __name__ == "__main__":
    main(dry="--dry-run" in sys.argv)
