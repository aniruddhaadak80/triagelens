"""Fast validation of the data pipeline (no TabPFN)."""
import numpy as np
from triagelens import experiments as E
from triagelens import data as D

records, rows, names, X, y = E.assemble()
print(f"n={len(y)} dead_rate={y.mean():.4f} n_features={len(names)}")
print("labels:", D.label_counts(records))
print("\nfeatures:")
for n in names:
    col = X[n]
    print(f"  {n:28} mean={col.mean():9.3f} min={col.min():8.2f} max={col.max():9.2f}")

aucs = E.single_feature_aucs(X, y, names)
print("\n--- single-feature AUC (top 12) ---")
for k, v in sorted(aucs.items(), key=lambda kv: -kv[1])[:12]:
    print(f"  {k:28} {v:.4f}")
print("--- single-feature AUC (bottom 5) ---")
for k, v in sorted(aucs.items(), key=lambda kv: kv[1])[:5]:
    print(f"  {k:28} {v:.4f}")

tr, te = E.time_split(len(y))
tr = E.subsample(tr, E.MAX_TRAIN, 1); te = E.subsample(te, E.MAX_TEST, 2)
print(f"\nsplit train={len(tr)} test={len(te)} "
      f"train_period={records[tr[0]]['created_at'][:10]}..{records[tr[-1]]['created_at'][:10]} "
      f"test_period={records[te[0]]['created_at'][:10]}..{records[te[-1]]['created_at'][:10]}")

for r in [
    E.evaluate(y[te], np.zeros(len(te)), "always-alive"),
    E.evaluate(y[te], X[max(aucs, key=lambda k: aucs[k])][te], "single best"),
]:
    print(f"  {r['model']:<28} auc={r['roc_auc']:.4f} P@100={r['precision_at_100']:.3f}")

s = E.logistic_scores(X, y[tr], names, X)
r = E.evaluate(y[te], s[te], "logistic all triage-time")
print(f"  {'logistic all triage-time':<28} auc={r['roc_auc']:.4f} "
      f"P@100={r['precision_at_100']:.3f} recall={r['recall']:.3f}")

print("\n--- H3 dead rate by body_len ---")
for b in [(0, 0), (1, 200), (201, 500), (501, 1000), (1001, 2000),
          (2001, 4000), (4001, 10**9)]:
    sel = np.array([(X["body_len"][i] >= b[0]) and (X["body_len"][i] <= b[1])
                    for i in range(len(y))])
    if sel.sum() >= 30:
        print(f"  {b[0]:>5}-{b[1]:<10} n={sel.sum():>5} dead={y[sel].mean():.4f}")
print("OK")