"""Train the triage model and score a backlog.

Two backends, both fully offline:

* ``logistic``  -- a balanced logistic regression. Instant. AUC 0.782.
* ``tabpfn``    -- Prior Labs' TabPFN v2, open weights, CPU-only. Best AUC 0.820,
                   but costs roughly 0.9s per scored row on a laptop CPU, so a
                   1,682-issue backlog takes about 25 minutes. That is fine as an
                   overnight batch job and useless interactively, which is why
                   logistic is the default.
"""

from __future__ import annotations

import time

import numpy as np

from . import data as D
from . import features as F


def _tabpfn_env() -> None:
    import os

    os.environ.setdefault("TABPFN_MODEL_VERSION", "v2")
    os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")


ERA_FEATURES = ("issue_number", "log_issue_number")


class TriageScorer:
    """Fit on historical closed issues, score an arbitrary backlog.

    ``use_era_features`` defaults to **False**, and that default is a direct
    consequence of the analysis in FINDINGS.md. ``issue_number`` is the single
    highest-scoring feature in-corpus (AUC 0.6305) and still worth 0.027 AUC on a
    time split, but an open backlog spans issue numbers from 167 to 29,073 while
    the training window only covers recent numbers. A model that has learned
    "when was this filed" cannot rank a present-day backlog, and it does not fail
    gracefully: it extrapolates and hands a maintainer 96% confidence on issues
    filed in 2022. Kept behind a flag so the effect is reproducible, off by
    default because it is unusable in deployment.
    """

    def __init__(self, backend: str = "logistic", max_train: int = 3000,
                 n_estimators: int = 1, use_era_features: bool = False):
        if backend not in {"logistic", "tabpfn"}:
            raise ValueError(f"unknown backend: {backend}")
        self.backend = backend
        self.max_train = max_train
        self.n_estimators = n_estimators
        self.use_era_features = use_era_features
        self.cols: list[str] = []
        self.vocab: list[str] = []
        self._mu: dict = {}
        self._sd: dict = {}
        self._clf = None
        self._tab = None
        self.train_seconds: float | None = None
        self.train_n: int = 0
        self.train_issue_range: tuple[int, int] | None = None

    # -- training ---------------------------------------------------------- #

    def fit(self, records: list[dict] | None = None) -> "TriageScorer":
        if records is None:
            records = D.build_frame()
            D.add_author_history(records)
        rows, self.cols = F.build_matrix(records)
        self.vocab = F.label_vocabulary(records)
        self.cols = [c for c in rows[0].keys()] if rows else []
        if not self.use_era_features:
            self.cols = [c for c in self.cols if c not in ERA_FEATURES]
        y = np.array([r["dead"] for r in records], dtype=int)

        # Keep the most recent training rows; a maintainer's present-day backlog
        # is best matched by recent history.
        keep = np.arange(max(0, len(y) - self.max_train), len(y))
        self.train_n = int(len(keep))
        nums = [records[i]["number"] for i in keep]
        self.train_issue_range = (min(nums), max(nums))

        self._mu = {c: float(np.mean([rows[i][c] for i in keep])) for c in self.cols}
        self._sd = {c: (float(np.std([rows[i][c] for i in keep])) or 1.0)
                    for c in self.cols}

        A = np.array([[(rows[i][c] - self._mu[c]) / self._sd[c] for c in self.cols]
                      for i in keep])

        t0 = time.time()
        if self.backend == "logistic":
            from sklearn.linear_model import LogisticRegression

            self._clf = LogisticRegression(max_iter=5000)
            self._clf.fit(A, y[keep])
        else:
            _tabpfn_env()
            from tabpfn import TabPFNClassifier

            self._tab = TabPFNClassifier(device="cpu", n_estimators=self.n_estimators)
            self._tab.fit(A, y[keep])
        self.train_seconds = time.time() - t0
        return self

    # -- scoring ----------------------------------------------------------- #

    def score(self, backlog: list[dict], progress_every: int = 25,
              records: list[dict] | None = None) -> np.ndarray:
        """Return P(closed without a fix) for each record in `backlog`."""
        if self._clf is None and self._tab is None:
            raise RuntimeError("call fit() first")
        if not backlog:
            return np.array([])

        rows, cols = F.build_matrix(backlog, vocab=self.vocab)
        # Author-history features must be computed against the full known corpus,
        # not just this batch, or "first issue from this author" is meaningless.
        known = {r["author_login"]: r for r in (records or [])}
        for rec, row in zip(backlog, rows):
            prior = known.get(rec.get("author_login"), {})
            row["author_prior_issues"] = prior.get("author_prior_issues", 0)
            row["author_is_first_issue"] = prior.get("author_is_first_issue", 0)
            row["author_is_frequent"] = prior.get("author_is_frequent", 0)
            row["log_author_prior_issues"] = float(
                np.log1p(max(0, row["author_prior_issues"])))
        A = np.array([[(r.get(c, 0.0) - self._mu.get(c, 0.0)) / self._sd.get(c, 1.0)
                       for c in self.cols] for r in rows])

        out = np.zeros(len(backlog))
        if self.backend == "logistic":
            out = self._clf.predict_proba(A)[:, 1]
        else:
            for i in range(len(backlog)):
                out[i] = self._tab.predict_proba(A[i:i + 1])[0, 1]
                if progress_every and (i + 1) % progress_every == 0:
                    print(f"    scored {i+1}/{len(backlog)}", flush=True)
        self.last_ood_fraction = self.out_of_distribution_fraction(backlog)
        return out

    def out_of_distribution_fraction(self, backlog: list[dict]) -> float:
        """Share of the backlog older than anything the model was trained on.

        Old open issues are exactly the ones a maintainer most wants help with,
        and they are also the ones this model is least entitled to score.
        """
        if not self.train_issue_range or not backlog:
            return 0.0
        lo, _ = self.train_issue_range
        older = sum(1 for r in backlog if r.get("number", 0) < lo)
        return older / len(backlog)


# -- deterministic rules that need no model -------------------------------- #

def empty_body_rule(rec: dict) -> bool:
    """Finding 7: an issue with no body is 4.6x less likely to ever ship.

    Free, model-free, and correct often enough to act on immediately. Returns
    False when the length is unknown, so a missing field is never read as "empty".
    """
    if "body_len" not in rec:
        return False
    return (rec.get("body_len") or 0) == 0


def evidence_for(rec: dict) -> list[str]:
    """Human-readable triage signals for one issue."""
    ev = []
    if empty_body_rule(rec):
        ev.append("empty body (dead rate 3.2% vs 14.9% peak)")
    if rec.get("body_len", 0) and rec["body_len"] < 200:
        ev.append("very short body (<200 chars)")
    labs = {lb.lower() for lb in (rec.get("labels") or [])}
    if "question" in labs:
        ev.append("labelled question")
    if "needs-info" in labs or "needs info" in labs:
        ev.append("awaiting info from reporter")
    if "good first issue" in labs:
        ev.append("scoped as good first issue")
    if rec.get("n_labels", 0) == 0:
        ev.append("no triage labels yet")
    prior = rec.get("author_prior_issues")
    if prior is not None and prior == 0:
        ev.append("first issue from this author")
    return ev