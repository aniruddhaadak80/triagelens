"""Tests for TriageLens.

The leakage-discipline tests are the important ones: they are the assertions that
keep this project honest rather than flattering.
"""

from __future__ import annotations

import numpy as np
import pytest

from triagelens import data as D
from triagelens import experiments as E
from triagelens import features as F
from triagelens import report as R
from triagelens import triage as T

CORPUS = D.load_raw() if (D.DATA_DIR / "issues_raw.json").exists() else None


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #

def test_roc_auc_perfect_and_inverted():
    y = np.array([0, 0, 1, 1])
    assert E.roc_auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    assert E.roc_auc(y, np.array([0.9, 0.8, 0.2, 0.1])) == pytest.approx(0.0)


def test_roc_auc_matches_sklearn():
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 400)
    s = rng.normal(size=400) + y * 0.7
    assert E.roc_auc(y, s) == pytest.approx(sk.roc_auc_score(y, s), abs=1e-9)


def test_roc_auc_handles_ties():
    # All-equal scores must return 0.5, not a division error.
    y = np.array([0, 1, 0, 1])
    assert E.roc_auc(y, np.ones(4)) == pytest.approx(0.5)


def test_roc_auc_single_class_is_nan():
    assert np.isnan(E.roc_auc(np.zeros(5), np.arange(5.0)))


def test_precision_at_k():
    y = np.array([1, 1, 0, 0, 1])
    s = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    assert E.precision_at_k(y, s, 2) == pytest.approx(1.0)
    assert E.precision_at_k(y, s, 5) == pytest.approx(0.6)


# --------------------------------------------------------------------------- #
# splits
# --------------------------------------------------------------------------- #

def test_time_split_is_monotonic_in_time():
    tr, te = E.time_split(100)
    assert tr.max() < te.min(), "train must be strictly older than test"
    assert len(tr) + len(te) == 100


def test_rolling_splits_are_forward_in_time():
    for tr, te in E.rolling_splits(1000, folds=4):
        assert tr.max() < te.min()


def test_random_split_is_actually_random():
    tr, te = E.random_split(1000)
    # A time-ordered corpus must NOT produce an ordered random split.
    assert not (tr.max() < te.min())


# --------------------------------------------------------------------------- #
# features: the leakage boundary
# --------------------------------------------------------------------------- #

def test_text_features_detect_structure():
    body = "```python\nimport os\n```\nTraceback (most recent call last):\nhttp://x.dev"
    f = F.text_features("Bug: crash on Windows 11", body)
    assert f["has_code_block"] == 1
    assert f["has_lang_fence"] == 1
    assert f["has_stacktrace"] == 1
    assert f["has_url"] == 1
    assert f["n_code_fences"] == 2


def test_text_features_on_empty_body():
    f = F.text_features("t", "")
    assert f["body_len"] == 0
    assert f["has_code_block"] == 0
    assert f["n_urls"] == 0


def test_question_detection():
    assert F.text_features("Why does this fail?", "")["title_is_question"] == 1
    assert F.text_features("How do I configure this", "")["title_starts_question"] == 1
    assert F.text_features("Crash on startup", "")["title_is_question"] == 0


def test_leaky_columns_are_quarantined():
    """The default feature set must contain nothing knowable only after the fact."""
    recs = [{
        "title": "t", "body": "", "labels": ["bug"],
        "author_login": "a", "author_prior_issues": 1,
        "author_is_first_issue": 0, "author_is_frequent": 0,
        "created": D._parse("2024-01-02T03:04:05Z"), "number": 42,
        "comments": 17, "reactions_total": 3, "reactions_heart": 0,
        "days_to_close": 5,
    }]
    _, names = F.build_matrix(recs)
    leaked = F.leak_columns(names)
    assert leaked == [], f"default features contain post-hoc columns: {leaked}"
    for banned in ("comments", "reactions_total", "days_to_close"):
        assert banned not in names


def test_resolution_time_labels_are_not_triage_time_labels():
    assert "fixes" in D.RESOLUTION_TIME_LABELS
    assert "accepted" in D.RESOLUTION_TIME_LABELS
    assert "bug" not in D.RESOLUTION_TIME_LABELS
    assert "needs-info" not in D.RESOLUTION_TIME_LABELS


def test_leaky_matrix_really_adds_the_leak():
    base = {"title": "t", "body": "", "labels": [], "author_login": "a",
            "author_prior_issues": 0, "author_is_first_issue": 1,
            "author_is_frequent": 0,
            "created": D._parse("2024-01-02T00:00:00Z"), "number": 1,
            "comments": 9, "reactions_total": 2, "reactions_heart": 1,
            "days_to_close": 3, "labels_extra": None}
    base["labels"] = ["fixes"]
    recs = [dict(base)]
    _, clean = F.build_matrix(recs)
    _, leaky = F.build_matrix(recs, include_leaky=True)
    added = set(leaky) - set(clean)
    assert added, "include_leaky must add columns"
    assert any(c.startswith("reslbl_") for c in added)


# --------------------------------------------------------------------------- #
# data layer
# --------------------------------------------------------------------------- #

def test_author_features_tolerate_missing_keys():
    f = F.author_features({"author_login": "x"})
    assert f["author_prior_issues"] == 0
    assert f["log_author_prior_issues"] == 0.0


def test_author_history_is_monotonic():
    recs = [{"author_login": "a"} for _ in range(3)] + [{"author_login": "a"}]
    out = D.add_author_history(recs)
    priors = [r["author_prior_issues"] for r in out]
    assert priors == sorted(priors)
    assert out[0]["author_is_first_issue"] == 1


def test_empty_body_rule():
    assert T.empty_body_rule({"body_len": 0}) is True
    assert T.empty_body_rule({"body_len": 5}) is False
    assert T.empty_body_rule({}) is False


def test_evidence_flags_under_specified_issues():
    ev = T.evidence_for({"body_len": 0, "labels": ["question"],
                         "author_prior_issues": 0, "n_labels": 0})
    joined = " ".join(ev)
    assert "empty body" in joined
    assert "labelled question" in joined
    assert "first issue from this author" in joined


# --------------------------------------------------------------------------- #
# scorer
# --------------------------------------------------------------------------- #

def _toy_corpus():
    recs = []
    for i in range(120):
        dead = i % 5 == 0
        recs.append({
            "number": 1000 + i,
            "title": "why?" if dead else "fix crash",
            "body": "" if dead else "```python\nx\n```" * 3,
            "labels": ["question"] if dead else ["bug"],
            "n_labels": 0 if dead else 2,
            "author_login": f"u{i}",
            "author_prior_issues": 0 if dead else 9,
            "author_is_first_issue": 1 if dead else 0,
            "author_is_frequent": 0 if dead else 1,
            "created": D._parse("2024-01-01T00:00:00Z"),
            "comments": 0, "reactions_total": 0, "reactions_heart": 0,
            "days_to_close": 1, "dead": int(dead), "slow": 0,
        })
    return recs


def test_scorer_fits_and_scores_within_bounds():
    recs = _toy_corpus()
    s = T.TriageScorer(backend="logistic", max_train=100).fit(recs)
    p = s.score(recs[:20], records=recs)
    assert p.shape == (20,)
    assert ((p >= 0) & (p <= 1)).all()


def test_era_features_are_off_by_default():
    recs = _toy_corpus()
    assert T.TriageScorer().use_era_features is False
    s = T.TriageScorer(backend="logistic", max_train=100).fit(recs)
    assert "issue_number" not in s.cols
    assert "log_issue_number" not in s.cols


def test_era_features_can_be_opted_into():
    recs = _toy_corpus()
    s = T.TriageScorer(backend="logistic", max_train=100,
                       use_era_features=True).fit(recs)
    assert "issue_number" in s.cols


def test_ood_fraction_detects_old_backlog():
    recs = _toy_corpus()
    s = T.TriageScorer(backend="logistic", max_train=50).fit(recs)
    assert s.train_issue_range[0] == 1070
    assert s.out_of_distribution_fraction([{"number": 5}]) == pytest.approx(1.0)
    assert s.out_of_distribution_fraction([{"number": 99999}]) == pytest.approx(0.0)


def test_unknown_backend_rejected():
    with pytest.raises(ValueError):
        T.TriageScorer(backend="gpt")


def test_score_before_fit_raises():
    with pytest.raises(RuntimeError):
        T.TriageScorer().score([{"number": 1}])


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #

def test_report_is_self_contained():
    recs = _toy_corpus()
    s = T.TriageScorer(backend="logistic", max_train=100).fit(recs)
    p = s.score(recs[:5], records=recs)
    html = R.render(recs[:5], p, results={}, backend="logistic", repo="o/r", top=5)
    assert html.startswith("<!doctype html>")
    assert "src=\"http" not in html, "report must not reference remote assets"
    assert "<script" not in html, "report must not need JavaScript"
    assert "o/r" in html


def test_report_escapes_untrusted_titles():
    recs = _toy_corpus()
    recs[0]["title"] = "<img src=x onerror=alert(1)>"
    s = T.TriageScorer(backend="logistic", max_train=100).fit(recs)
    p = s.score(recs[:1], records=recs)
    html = R.render(recs[:1], p, results={}, backend="logistic")
    assert "<img src=x" not in html
    assert "&lt;img" in html


def test_report_handles_empty_backlog():
    html = R.render([], np.array([]), results={}, backend="logistic")
    assert "No open issues found" in html


# --------------------------------------------------------------------------- #
# corpus-backed tests (skipped without the downloaded corpus)
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(CORPUS is None, reason="corpus not downloaded")
def test_corpus_provenance_present():
    assert CORPUS["repo"]
    assert CORPUS["retrieved_at_utc"]
    assert CORPUS["issues"]


@pytest.mark.skipif(CORPUS is None, reason="corpus not downloaded")
def test_prs_are_excluded():
    kinds = {("pull_request" in i) for i in CORPUS["issues"]}
    assert kinds == {False}, "the issues endpoint also returns PRs; they must be filtered"


@pytest.mark.skipif(CORPUS is None, reason="corpus not downloaded")
def test_closed_only_and_labels_are_consistent():
    recs = D.add_author_history(D.build_frame())
    assert all(r["closed_at"] for r in recs)
    for r in recs[:500]:
        expected = 1 if r["state_reason"] in D.DEAD_REASONS else 0
        assert r["dead"] == expected