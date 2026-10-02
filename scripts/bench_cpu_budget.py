"""Find a TabPFN v2 configuration that actually completes on CPU, in budget."""
import os, time, sys

os.environ.setdefault("TABPFN_MODEL_VERSION", "v2")
os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")

import numpy as np
from tabpfn import TabPFNClassifier

rng = np.random.default_rng(0)
CFGS = [
    (400, 100, 16, 1),
    (800, 150, 16, 1),
    (1000, 200, 24, 1),
]
for n_train, n_test, n_feat, n_est in CFGS:
    Xb = rng.normal(size=(n_train + n_test, n_feat))
    yb = (Xb[:, 0] + 0.6 * Xb[:, 1] - 0.5 * Xb[:, 2] > 0).astype(int)
    t = time.time()
    try:
        c = TabPFNClassifier(device="cpu", n_estimators=n_est)
        c.fit(Xb[:n_train], yb[:n_train])
        c.predict_proba(Xb[n_train:])
        dt = time.time() - t
        print(f"OK  train={n_train:>5} test={n_test:>4} feat={n_feat:>3} est={n_est}: {dt:7.1f}s",
              flush=True)
    except Exception as e:
        print(f"ERR train={n_train} : {type(e).__name__}: {str(e)[:120]}", flush=True)
print("BENCH_DONE")