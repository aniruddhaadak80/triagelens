"""Pretty-print the key findings from data/results.json."""
import json, pathlib

r = json.loads(pathlib.Path("data/results.json").read_text(encoding="utf-8"))

print("=== CORPUS ===")
c = r["corpus"]
for k in ("repo", "retrieved_at_utc", "raw_items_returned", "pull_requests_skipped",
          "n_closed", "n_dead", "n_alive", "dead_rate", "n_slow",
          "span_first", "span_last", "n_features_triage_time"):
    print(f"  {k}: {c.get(k, 'n/a')}")

print("\n=== SPLIT ===")
for k, v in r["split"].items():
    print(f"  {k}: {v}")

print("\n=== H1 LOUDNESS (the folk belief) ===")
for k, v in r["h1_loudness"].items():
    print(f"  {k}: {v}")

print("\n=== SPLIT SENSITIVITY ===")
for k, v in r["split_sensitivity"].items():
    print(f"  {k}: {v}")

print("\n=== ABLATIONS ===")
for blk in ("era_proxy_ablation", "author_ablation", "text_ablation"):
    print(f"  {blk}:")
    for k, v in r[blk].items():
        print(f"    {k}: {v}")

print("\n=== H3 dead rate by body_len ===")
for b in r["h3_body_length_curve"]:
    hi = b["body_len_max"] if b["body_len_max"] is not None else "inf"
    print(f"  {b['body_len_min']:>5}-{hi:<6} n={b['n']:>5} dead_rate={b['dead_rate']:.4f}")

print("\n=== PERMUTATION IMPORTANCE (logistic, top 14) ===")
for p in r["permutation_importance_logistic"][:14]:
    print(f"  {p['feature']:28} drop={p['auc_drop_mean']:+.4f} +/- {p['auc_drop_std']:.4f}")
print("  ... bottom 5:")
for p in r["permutation_importance_logistic"][-5:]:
    print(f"  {p['feature']:28} drop={p['auc_drop_mean']:+.4f}")

print("\n=== ROLLING TIME FOLDS (cheap models) ===")
for f in r["rolling_folds"]:
    print(f"  fold {f['fold']}: logistic={f['logistic_auc']:.4f} "
          f"folk={f['folk_belief_auc']:.4f} single_best={f['single_best_auc']:.4f} "
          f"n={f['n_test']} dead={f['test_dead_rate']:.3f}")

print("\n=== TABPFN ROLLING FOLDS ===")
for f in r["tabpfn_rolling_folds"]:
    print(f"  fold {f['fold']}: auc={f['auc']:.4f} train={f['n_train']} "
          f"test={f['n_test']} {f['seconds']}s")

print("\n=== SINGLE-FEATURE AUC (top 8 / bottom 4) ===")
sf = r["single_feature_auc_whole_corpus"]
items = list(sf.items())
for k, v in items[:8]:
    print(f"  {k:28} {v:+.4f}")
print("  ...")
for k, v in items[-4:]:
    print(f"  {k:28} {v:+.4f}")

print("\n=== HOLDOUT TABLE ===")
for t in r["holdout_table"]:
    print(f"  {t['model']}")
    print(f"     auc={t['roc_auc']:.4f} P@100={t['precision_at_100']:.3f} "
          f"P@10%={t['precision_at_10pct']:.3f} recall={t['recall']:.3f} "
          f"base={t['base_rate']:.3f} n={t['n_test']}")
print(f"\nruntime_seconds: {r.get('runtime_seconds')}")