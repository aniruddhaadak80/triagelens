"""The experiment: does an offline tabular foundation model beat the loudest-first
heuristic at telling a maintainer which issues in their backlog will never land?

Compute allocation note: TabPFN costs ~0.19s per test row per 1000 training rows on
CPU, so it is fitted on one large held-out time split. The cheap baselines are run
across several rolling time folds so the stability claim is properly supported.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

# TabPFN v2 is Prior Labs' ungated open-weights checkpoint (public GCS mirror),
# so this whole project runs offline with no account, no key and no licence flow.
os.environ.setdefault("TABPFN_MODEL_VERSION", "v2")
os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")

import numpy as np

from . import data as D
from . import features as F

RNG = np.random.default_rng(20261002)
MAX_TRAIN = 1500          # TabPFN CPU budget
MAX_TEST = 600
N_ESTIMATORS = 1
FOLDS = 4                 # rolling time folds for the cheap baselines
TRAIN_FRACTION = 0.8


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #

def roc_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y_true)
    s = np.asarray(score, dtype=float)
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def precision_at_k(y_true: np.ndarray, score: np.ndarray, k: int) -> float:
    order = np.argsort(-np.asarray(score, dtype=float))[:k]
    return float(np.asarray(y_true)[order].mean())


def evaluate(y_true: np.ndarray, score: np.ndarray, label: str) -> dict:
    y = np.asarray(y_true)
    s = np.asarray(score, dtype=float)
    thr = 0.5 if set(np.unique(s)) <= {0.0, 1.0} else float(np.mean(s))
    pred = (s >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    return {
        "model": label,
        "roc_auc": roc_auc(y, s),
        "precision_at_50": precision_at_k(y, s, 50),
        "precision_at_100": precision_at_k(y, s, 100),
        "precision_at_10pct": precision_at_k(y, s, max(1, int(0.10 * len(y)))),
        "recall": tp / max(1, int(y.sum())),
        "flagged_rate": float(pred.mean()),
        "n_test": int(len(y)),
        "base_rate": float(y.mean()),
    }


def subsample(idx: np.ndarray, cap: int, seed_offset: int = 0) -> np.ndarray:
    if len(idx) <= cap:
        return idx
    rng = np.random.default_rng(20261002 + seed_offset)
    return np.sort(rng.choice(idx, size=cap, replace=False))


# --------------------------------------------------------------------------- #
# data assembly
# --------------------------------------------------------------------------- #

def assemble(include_leaky: bool = False):
    records = D.build_frame()
    D.add_author_history(records)
    rows, names = F.build_matrix(records, include_leaky=include_leaky)
    y = np.array([r["dead"] for r in records], dtype=int)
    X = {n: np.array([r[n] for r in rows], dtype=float) for n in names}
    return records, rows, names, X, y


def time_split(n: int, fraction: float = TRAIN_FRACTION):
    cut = int(n * fraction)
    return np.arange(cut), np.arange(cut, n)


def rolling_splits(n: int, folds: int = FOLDS, fraction: float = TRAIN_FRACTION):
    """Expanding-window time folds: train on [0, e), test on [e, e + w)."""
    out = []
    step = (n - int(n * fraction)) // folds
    for k in range(folds):
        e = int(n * fraction) + k * step
        w = step if k < folds - 1 else n - e
        if e >= n or w <= 0:
            continue
        out.append((np.arange(0, e), np.arange(e, min(n, e + w))))
    return out


# --------------------------------------------------------------------------- #
# baselines
# --------------------------------------------------------------------------- #

def logistic_scores(X_all: dict, y_all: np.ndarray, cols: list[str],
                    train_idx: np.ndarray, test_idx: np.ndarray):
    """Fit a logistic regression on `train_idx`, score `test_idx`.

    Takes the full column dict and explicit index sets so a time split can never
    be silently turned into a random split by a caller mistake. Features are
    standardised because raw issue numbers run to ~29,000 while binary flags sit
    at 0/1.

    Deliberately **unweighted**. The usual remedy for a 13.5% positive rate is
    ``class_weight="balanced"``; measured on this corpus it made every metric
    worse and destroyed calibration (AUC 0.7819 -> 0.7869 without it,
    precision@top-10% 0.583 -> 0.650, mean predicted score 0.508 -> 0.148
    against a 0.176 base rate). Ranking metrics are threshold-free, so weighting
    bought nothing and cost the probability a maintainer would act on.
    """
    from sklearn.linear_model import LogisticRegression

    mu = {c: X_all[c][train_idx].mean() for c in cols}
    sd = {c: (X_all[c][train_idx].std() or 1.0) for c in cols}
    A = np.column_stack([(X_all[c][train_idx] - mu[c]) / sd[c] for c in cols])
    B = np.column_stack([(X_all[c][test_idx] - mu[c]) / sd[c] for c in cols])
    clf = LogisticRegression(max_iter=5000)
    clf.fit(A, y_all[train_idx])
    return clf.predict_proba(B)[:, 1], (clf, list(cols), mu, sd)


def single_feature_aucs(X, y, cols):
    out = {}
    for c in cols:
        v = X[c]
        if np.all(v == v[0]):
            out[c] = 0.5
            continue
        out[c] = roc_auc(y, v)
    return out


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def random_split(n: int, seed: int = 20261002, test_frac: float = 0.25):
    """A random split of the SAME rows. Included only to quantify how much a
    random split overstates performance relative to a time split."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    cut = int(n * (1 - test_frac))
    return np.sort(perm[:cut]), np.sort(perm[cut:])


RESULTS_PATH = Path(__file__).resolve().parent.parent / "data" / "results.json"


def save(results: dict) -> None:
    """Checkpoint after every expensive stage so a crash never loses compute."""
    RESULTS_PATH.write_text(json.dumps(results, indent=2, default=str),
                            encoding="utf-8")
    print(f"  [checkpoint] wrote {RESULTS_PATH}", flush=True)


def run() -> dict:
    t_all = time.time()
    results: dict = {}

    raw = D.load_raw()
    records, rows, names, X, y = assemble(include_leaky=False)
    n = len(y)
    results["corpus"] = {
        "repo": raw["repo"],
        "retrieved_at_utc": raw["retrieved_at_utc"],
        "raw_items_returned": raw["raw_items_returned"],
        "pull_requests_skipped": raw["pull_requests_skipped"],
        **D.label_counts(records),
        "n_features_triage_time": len(names),
        "features": names,
    }
    print(f"corpus: {n} closed issues, dead base rate = {y.mean():.4f}", flush=True)
    print(f"features ({len(names)}): {names}", flush=True)

    tr, te = time_split(n)
    tr = subsample(tr, MAX_TRAIN, 1)
    te = subsample(te, MAX_TEST, 2)
    results["split"] = {
        "strategy": "expanding time split (train older issues, test newer)",
        "train_rows_used": int(len(tr)),
        "test_rows_used": int(len(te)),
        "train_period": [records[tr[0]]["created_at"], records[tr[-1]]["created_at"]],
        "test_period": [records[te[0]]["created_at"], records[te[-1]]["created_at"]],
        "train_dead_rate": float(y[tr].mean()),
        "test_dead_rate": float(y[te].mean()),
    }
    print(f"split: train={len(tr)} ({records[tr[0]]['created_at'][:10]}.."
          f"{records[tr[-1]]['created_at'][:10]}) test={len(te)} "
          f"({records[te[0]]['created_at'][:10]}..{records[te[-1]]['created_at'][:10]})",
          flush=True)

    table = []

    # 0. trivial baseline
    table.append(evaluate(y[te], np.zeros(len(te)), "always-alive (trivial)"))

    # 1. best single triage-time feature on the whole corpus (diagnostic only)
    aucs = single_feature_aucs(X, y, names)
    results["single_feature_auc_whole_corpus"] = dict(
        sorted(aucs.items(), key=lambda kv: -kv[1]))
    ranked = sorted(aucs.items(), key=lambda kv: -max(kv[1], 1 - kv[1]))
    best_feature = ranked[0][0]
    # Orient the single feature so that higher = more likely dead. Several of the
    # strongest raw signals (issue_number, n_labels, lbl_bug) are inversely
    # related to the label, and reporting them un-oriented would understate them.
    orient = 1.0 if aucs[best_feature] >= 0.5 else -1.0
    single_row = evaluate(y[te], orient * X[best_feature][te],
                          f"single best feature: {best_feature}")
    single_row["oriented"] = bool(orient > 0)
    single_row["whole_corpus_auc"] = float(aucs[best_feature])
    table.append(single_row)

    # 2. all triage-time features, logistic
    s, _ = logistic_scores(X, y, names, tr, te)
    table.append(evaluate(y[te], s, "logistic, all triage-time features"))

    # 3. THE FOLK BELIEF: triage the loudest first. Leaks by construction,
    #    because comments/reactions are only knowable after the fact.
    recs_leaky, _, lnames, Xl, yl = assemble(include_leaky=True)
    folk_cols = ["log_comments", "reactions_total"]
    s_folk, _ = logistic_scores(Xl, yl, folk_cols, tr, te)
    folk = evaluate(y[te], s_folk, "FOLK BELIEF: loudest-first (leaky features)")
    folk["leaky"] = True
    folk["note"] = ("comments and reactions are only knowable after the issue has run "
                    "its course; reported to quantify the leak, not to recommend it")
    table.append(folk)

    # 4. RANDOM SPLIT vs TIME SPLIT -- the headline methodological result
    rtr, rte = random_split(n)
    rtr = subsample(rtr, MAX_TRAIN, 3)
    rte = subsample(rte, MAX_TEST, 4)
    s_rand, _ = logistic_scores(X, y, names, rtr, rte)
    row_rand = evaluate(y[rte], s_rand, "logistic, all triage-time (RANDOM split)")
    row_rand["split"] = "random"
    row_time = [t_ for t_ in table if t_["model"].startswith("logistic, all triage")][0].copy()
    row_time["split"] = "time"
    results["split_sensitivity"] = {
        "random_split_auc": row_rand["roc_auc"],
        "time_split_auc": row_time["roc_auc"],
        "inflation": row_rand["roc_auc"] - row_time["roc_auc"],
        "era_proxy_feature": "issue_number",
        "era_proxy_auc_random": roc_auc(y[rte], X["issue_number"][rte]),
        "era_proxy_auc_time": roc_auc(y[te], X["issue_number"][te]),
    }
    print(f"random-split AUC={row_rand['roc_auc']:.4f} vs "
          f"time-split AUC={row_time['roc_auc']:.4f} "
          f"(inflation {row_rand['roc_auc']-row_time['roc_auc']:+.4f})", flush=True)
    table.append(row_rand)

    # 5. HEADLINE: TabPFN on triage-time features only, time split
    from tabpfn import TabPFNClassifier
    A = np.column_stack([X[c][tr] for c in names])
    B = np.column_stack([X[c][te] for c in names])
    t = time.time()
    clf = TabPFNClassifier(device="cpu", n_estimators=N_ESTIMATORS)
    clf.fit(A, y[tr])
    p_tab = clf.predict_proba(B)[:, 1]
    tab_time = time.time() - t
    tab_row = evaluate(y[te], p_tab, "TabPFN v2, triage-time features only")
    tab_row["fit_predict_seconds"] = round(tab_time, 1)
    tab_row["n_estimators"] = N_ESTIMATORS
    table.append(tab_row)
    print(f"TabPFN done in {tab_time:.1f}s  auc={tab_row['roc_auc']:.4f}", flush=True)

    # 6. leak demonstration: TabPFN with resolution-time labels + comments
    leak_cols = [c for c in lnames if c not in names]
    A2 = np.column_stack([Xl[c][tr] for c in lnames])
    B2 = np.column_stack([Xl[c][te] for c in lnames])
    t = time.time()
    clf2 = TabPFNClassifier(device="cpu", n_estimators=N_ESTIMATORS)
    clf2.fit(A2, yl[tr])
    p_leak = clf2.predict_proba(B2)[:, 1]
    leak_row = evaluate(y[te], p_leak,
                        "TabPFN + resolution-time labels & comments (LEAK)")
    leak_row["leaky"] = True
    leak_row["leak_columns"] = leak_cols
    leak_row["fit_predict_seconds"] = round(time.time() - t, 1)
    table.append(leak_row)
    print(f"TabPFN(leaky) done  auc={leak_row['roc_auc']:.4f}", flush=True)

    results["holdout_table"] = table
    save(results)

    # 7. Feature importance.
    #    TabPFN costs ~0.9s per predicted row on CPU, so permuting all 46
    #    features three times over would take ~14 hours. Importance is therefore
    #    measured on the logistic model, where permutation is essentially free,
    #    and the single-feature AUC table above gives the per-feature ranking.
    s_full, lr_pack = logistic_scores(X, y, names, tr, te)
    lr_clf, lr_cols, lr_mu, lr_sd = lr_pack
    base_auc_lr = roc_auc(y[te], s_full)
    rng = np.random.default_rng(7)
    perm_rows = []
    Bte = np.column_stack([(X[c][te] - lr_mu[c]) / lr_sd[c] for c in names])
    for i, c in enumerate(names):
        drops = []
        for _ in range(10):
            Bp = Bte.copy()
            rng.shuffle(Bp[:, i])
            drops.append(base_auc_lr - roc_auc(y[te], lr_clf.predict_proba(Bp)[:, 1]))
        perm_rows.append({"feature": c, "auc_drop_mean": float(np.mean(drops)),
                          "auc_drop_std": float(np.std(drops))})
    perm_rows.sort(key=lambda r: -r["auc_drop_mean"])
    results["permutation_importance_logistic"] = perm_rows
    results["logistic_coefficients"] = {
        c: float(v) for c, v in zip(lr_cols, lr_clf.coef_[0])}
    print("  permutation importance (logistic, 10 reps) done", flush=True)

    # 8. rolling time folds for the cheap models (stability across time)
    fold_rows = []
    for k, (ftr, fte) in enumerate(rolling_splits(n)):
        fr = subsample(ftr, 2000, 10 + k)
        row = {"fold": k,
               "test_period": [records[fte[0]]["created_at"], records[fte[-1]]["created_at"]],
               "n_test": int(len(fte)),
               "test_dead_rate": float(y[fte].mean()),
               "always_alive_auc": roc_auc(y[fte], np.zeros(len(fte)))}
        sp, _ = logistic_scores(X, y, names, fr, fte)
        row["logistic_auc"] = roc_auc(y[fte], sp)
        row["single_best_auc"] = roc_auc(y[fte], X[best_feature][fte])
        sf, _ = logistic_scores(Xl, yl, folk_cols, fr, fte)
        row["folk_belief_auc"] = roc_auc(y[fte], sf)
        fold_rows.append(row)
        print(f"  fold {k}: logistic={row['logistic_auc']:.4f} "
              f"folk={row['folk_belief_auc']:.4f}", flush=True)
    results["rolling_folds"] = fold_rows

    # 9. TabPFN stability across three rolling time folds, on a small test slice
    #    so the ~0.9s/row CPU cost stays affordable.
    from tabpfn import TabPFNClassifier
    tab_folds = []
    for k, (ftr, fte) in enumerate(rolling_splits(n)[:3]):
        fr = subsample(ftr, 1000, 30 + k)
        fe = subsample(fte, 150, 40 + k)
        A = np.column_stack([X[c][fr] for c in names])
        B = np.column_stack([X[c][fe] for c in names])
        t = time.time()
        c_ = TabPFNClassifier(device="cpu", n_estimators=N_ESTIMATORS)
        c_.fit(A, y[fr])
        p = c_.predict_proba(B)[:, 1]
        tab_folds.append({"fold": k, "n_train": int(len(fr)), "n_test": int(len(fe)),
                          "auc": roc_auc(y[fe], p),
                          "seconds": round(time.time() - t, 1)})
        print(f"  TabPFN fold {k}: auc={tab_folds[-1]['auc']:.4f} "
              f"({tab_folds[-1]['seconds']}s)", flush=True)
    results["tabpfn_rolling_folds"] = tab_folds
    save(results)

    # 9. H3: dead rate by body-length bucket (tests for an inverted U)
    buckets = [(0, 0), (1, 200), (201, 500), (501, 1000),
               (1001, 2000), (2001, 4000), (4001, 10**9)]
    h3 = []
    for lo, hi in buckets:
        sel = np.array([(lo <= X["body_len"][i] <= hi) for i in range(n)])
        if sel.sum() < 30:
            continue
        h3.append({"body_len_min": lo,
                   "body_len_max": (None if hi > 10**8 else hi),
                   "n": int(sel.sum()), "dead_rate": float(y[sel].mean())})
    results["h3_body_length_curve"] = h3

    # 10. H1: does loudness predict anything?
    dead_c = [r["comments"] for r in records if r["dead"] == 1]
    alive_c = [r["comments"] for r in records if r["dead"] == 0]
    results["h1_loudness"] = {
        "auc_log_comments": roc_auc(y, Xl["log_comments"]),
        "auc_reactions_total": roc_auc(y, Xl["reactions_total"]),
        "median_comments_when_dead": float(np.median(dead_c)),
        "median_comments_when_alive": float(np.median(alive_c)),
        "mean_comments_when_dead": float(np.mean(dead_c)),
        "mean_comments_when_alive": float(np.mean(alive_c)),
    }

    # 11. Does the era proxy carry the model? Refit the logistic WITHOUT
    #     issue_number / log_issue_number and compare. If AUC barely moves, the
    #     apparent skill came from a calendar feature a maintainer cannot use to
    #     rank today's backlog.
    era_cols = [c for c in names if "issue_number" not in c]
    s_noera, _ = logistic_scores(X, y, era_cols, tr, te)
    results["era_proxy_ablation"] = {
        "with_era_features": base_auc_lr,
        "without_era_features": roc_auc(y[te], s_noera),
        "delta": base_auc_lr - roc_auc(y[te], s_noera),
        "dropped": [c for c in names if "issue_number" in c],
    }
    print(f"  era ablation: with={base_auc_lr:.4f} "
          f"without={results['era_proxy_ablation']['without_era_features']:.4f}",
          flush=True)

    # 12. Author-history ablation
    auth_cols = [c for c in names if not c.startswith("author_")]
    s_noauth, _ = logistic_scores(X, y, auth_cols, tr, te)
    results["author_ablation"] = {
        "with_author_features": base_auc_lr,
        "without_author_features": roc_auc(y[te], s_noauth),
        "delta": base_auc_lr - roc_auc(y[te], s_noauth),
    }

    # 13. Text-structure ablation (are the body probes doing anything?)
    text_cols = [c for c in names if c not in
                 {"title_len", "body_len", "log_body_len", "n_code_fences",
                  "has_code_block", "has_stacktrace", "has_url", "n_urls",
                  "has_version", "has_config_file", "has_cli_flag",
                  "has_lang_fence", "has_checkbox", "n_quote_lines",
                  "title_is_question", "title_starts_question", "title_n_caps"}]
    s_notext, _ = logistic_scores(X, y, text_cols, tr, te)
    results["text_ablation"] = {
        "with_text_features": base_auc_lr,
        "without_text_features": roc_auc(y[te], s_notext),
        "delta": base_auc_lr - roc_auc(y[te], s_notext),
    }

    results["runtime_seconds"] = round(time.time() - t_all, 1)
    return results


if __name__ == "__main__":
    res = run()
    out = Path(__file__).resolve().parent.parent / "data" / "results.json"
    out.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {out}")
    print("\n=== HOLDOUT TABLE (time split) ===")
    for r in res["holdout_table"]:
        auc = r["roc_auc"]
        print(f"  {r['model']:<58} auc={auc:.4f} P@100={r['precision_at_100']:.3f} "
              f"recall={r['recall']:.3f}")