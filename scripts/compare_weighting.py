"""Balanced vs unweighted logistic: does calibration cost ranking accuracy?"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from triagelens import experiments as E

records, rows, names, X, y = E.assemble()
n = len(y)
tr = E.subsample(E.time_split(n)[0], 1500, 1)
te = E.subsample(E.time_split(n)[1], 600, 2)

mu = {c: X[c][tr].mean() for c in names}
sd = {c: (X[c][tr].std() or 1.0) for c in names}
A = np.column_stack([(X[c][tr] - mu[c]) / sd[c] for c in names])
B = np.column_stack([(X[c][te] - mu[c]) / sd[c] for c in names])

for label, kw in (("balanced", {"class_weight": "balanced"}), ("unweighted", {})):
    clf = LogisticRegression(max_iter=5000, **kw).fit(A, y[tr])
    p = clf.predict_proba(B)[:, 1]
    e = E.evaluate(y[te], p, label)
    print(f"{label:12} auc={e['roc_auc']:.4f} P@100={e['precision_at_100']:.3f} "
          f"P@10%={e['precision_at_10pct']:.3f} "
          f"score_range=[{p.min():.3f}, {p.max():.3f}] mean={p.mean():.3f} "
          f"base={e['base_rate']:.3f}")

# also: what does calibration do to the deployed window?
print("\nrecent-3000 window base rate:", round(float(y[-3000:].mean()), 4))
print("CMP_DONE")