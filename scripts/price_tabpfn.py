"""Price fit vs predict so the permutation-importance loop can be budgeted."""
import os, time
os.environ.setdefault("TABPFN_MODEL_VERSION", "v2")
os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")
import numpy as np
from tabpfn import TabPFNClassifier

rng = np.random.default_rng(0)
n_train, n_feat = 1500, 46
X = rng.normal(size=(n_train, n_feat))
y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)

clf = TabPFNClassifier(device="cpu", n_estimators=1)
t = time.time(); clf.fit(X, y); t_fit = time.time() - t
print(f"fit({n_train}x{n_feat}) = {t_fit:.1f}s")

for n_test in (300, 600):
    Xt = rng.normal(size=(n_test, n_feat))
    t = time.time(); clf.predict_proba(Xt); t_pred = time.time() - t
    print(f"predict({n_test}) = {t_pred:.1f}s  ({t_pred/n_test*1000:.0f} ms/row)")
    print(f"  => 46 features x 2 reps at this size = "
          f"{46*2*t_pred/60:.1f} min")
print("PRICE_DONE")