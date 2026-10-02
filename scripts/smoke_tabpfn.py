"""Smoke-test TabPFN v2 (ungated) and benchmark realistic scale."""
import os, time

os.environ.setdefault("TABPFN_MODEL_VERSION", "v2")
os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")

import numpy as np
import torch
from tabpfn import TabPFNClassifier

print(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()} "
      f"threads={torch.get_num_threads()}")

t0 = time.time()
rng = np.random.default_rng(0)
X = rng.normal(size=(300, 10))
y = (X[:, 0] * 2 + rng.normal(scale=0.3, size=300) > 0).astype(int)

clf = TabPFNClassifier(device="cpu")
clf.fit(X[:240], y[:240])
p = clf.predict_proba(X[240:])[:, 1]
print(f"v2 classifier acc={((p>0.5).astype(int)==y[240:]).mean():.3f}  t={time.time()-t0:.1f}s")

for n_train, n_test, n_feat in [(2000, 400, 24), (5000, 800, 24)]:
    t = time.time()
    Xb = rng.normal(size=(n_train + n_test, n_feat))
    yb = (Xb[:, 0] + 0.6 * Xb[:, 1] - 0.5 * Xb[:, 2] > 0).astype(int)
    c = TabPFNClassifier(device="cpu")
    c.fit(Xb[:n_train], yb[:n_train])
    c.predict_proba(Xb[n_train:])
    print(f"  fit={n_train} test={n_test} feat={n_feat}: {time.time()-t:.1f}s")

print(f"TOTAL {time.time()-t0:.1f}s  TABPFN_V2_OK")