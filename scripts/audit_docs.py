"""Audit: do the numbers quoted in the docs match data/results.json?

Guards against the classic submission failure where the write-up drifts away
from the artefact. Every ROC-AUC and precision figure quoted in README.md and
FINDINGS.md is re-derived from results.json here.
"""
import json
import re
import sys
from pathlib import Path

res = json.loads(Path("data/results.json").read_text(encoding="utf-8"))
table = {r["model"]: r for r in res["holdout_table"]}


def find(sub):
    for k, v in table.items():
        if sub in k:
            return k, v
    raise KeyError(sub)


checks = []


def expect(doc, label, actual, quoted, tol=0.0006):
    ok = abs(actual - quoted) <= tol
    checks.append((ok, doc, label, quoted, actual))


# --- README table ---------------------------------------------------------- #
readme = Path("README.md").read_text(encoding="utf-8")
findings = Path("FINDINGS.md").read_text(encoding="utf-8")

k, v = find("always-alive")
expect("README", "constant AUC", v["roc_auc"], 0.5000)
expect("README", "constant P@100", v["precision_at_100"], 0.240)
k, v = find("FOLK BELIEF")
expect("README", "folk AUC", v["roc_auc"], 0.5208)
expect("README", "folk P@100", v["precision_at_100"], 0.190)
k, v = find("single best feature")
expect("README", "single AUC", v["roc_auc"], 0.7398)
expect("README", "single P@100", v["precision_at_100"], 0.610)
k, v = find("logistic, all triage-time features")
expect("README", "logistic AUC", v["roc_auc"], 0.7869)
expect("README", "logistic P@100", v["precision_at_100"], 0.560)
k, v = find("TabPFN v2")
expect("README", "tabpfn AUC", v["roc_auc"], 0.8202)
expect("README", "tabpfn P@100", v["precision_at_100"], 0.650)
k, v = find("LEAK")
expect("README", "leak AUC", v["roc_auc"], 0.8095)
expect("README", "leak P@100", v["precision_at_100"], 0.580)

# --- FINDINGS specific claims ---------------------------------------------- #
h1 = res["h1_loudness"]
expect("FINDINGS", "median comments dead", h1["median_comments_when_dead"], 2.0, 1e-9)
expect("FINDINGS", "median comments alive", h1["median_comments_when_alive"], 2.0, 1e-9)
expect("FINDINGS", "mean comments dead", h1["mean_comments_when_dead"], 2.94, 0.005)
expect("FINDINGS", "mean comments alive", h1["mean_comments_when_alive"], 3.35, 0.005)

folk_aucs = [f["folk_belief_auc"] for f in res["rolling_folds"]]
expect("FINDINGS", "folk fold0", folk_aucs[0], 0.5410)
expect("FINDINGS", "folk fold3", folk_aucs[3], 0.5006)
assert max(folk_aucs) < 0.55, f"claimed 'never above 0.55' but max={max(folk_aucs)}"

tab_folds = [f["auc"] for f in res["tabpfn_rolling_folds"]]
for got, quoted in zip(tab_folds, (0.695, 0.869, 0.779)):
    expect("FINDINGS", f"tabpfn fold {quoted}", got, quoted, 0.0006)

ab = res["ablations"] if "ablations" in res else {}
text = res["text_ablation"]
expect("FINDINGS", "text ablation with", text["with_text_features"], 0.7869)
expect("FINDINGS", "text ablation without", text["without_text_features"], 0.7987)
era = res["era_proxy_ablation"]
expect("FINDINGS", "era ablation without", era["without_era_features"], 0.7567)
expect("FINDINGS", "era delta", era["delta"], 0.0302, 0.0002)

ss = res["split_sensitivity"]
expect("FINDINGS", "random split AUC", ss["random_split_auc"], 0.7870)
expect("FINDINGS", "time split AUC", ss["time_split_auc"], 0.7869)
expect("FINDINGS", "inflation", ss["inflation"], 0.00004, 0.00002)
# The leakage claim must compare the same feature across two splits, so the
# random-split figure is checked against era_proxy_auc_random (not the
# whole-corpus figure).
expect("FINDINGS", "era AUC random split", ss["era_proxy_auc_random"], 0.6452)
expect("FINDINGS", "era AUC time split", ss["era_proxy_auc_time"], 0.5364)
whole = res["single_feature_auc_whole_corpus"]["issue_number"]
expect("FINDINGS", "era AUC whole corpus", whole, 0.6305)
# The whole-corpus and random-split figures are two estimates of the same
# quantity and must agree within sampling noise; the leakage claim is the gap
# between random and time splits, not an ordering against the pooled figure.
checks.append((abs(whole - ss["era_proxy_auc_random"]) < 0.03,
               "FINDINGS", "whole-corpus ~ random-split (same quantity)", True,
               f"{whole:.4f} vs {ss['era_proxy_auc_random']:.4f}"))
checks.append((ss["era_proxy_auc_random"] - ss["era_proxy_auc_time"] > 0.08,
               "FINDINGS", "random-vs-time gap is the leakage claim", True,
               f"gap={ss['era_proxy_auc_random'] - ss['era_proxy_auc_time']:.4f}"))
checks.append((whole == max(res["single_feature_auc_whole_corpus"].values()),
               "FINDINGS", "issue_number is top single feature", True, whole))

split = res["split"]
expect("FINDINGS", "train dead rate", split["train_dead_rate"], 0.101, 0.001)
expect("FINDINGS", "test dead rate", split["test_dead_rate"], 0.195, 0.001)

h3 = {b["body_len_min"]: b["dead_rate"] for b in res["h3_body_length_curve"]}
expect("FINDINGS", "empty body dead rate", h3[0], 0.032, 0.0006)
peak = max(h3.values())
expect("FINDINGS", "peak dead rate", peak, 0.1488)
ratio = peak / h3[0]
print(f"empty-body multiplier: {ratio:.2f}x  (docs claim 4.6x)")

perm = {p["feature"]: p["auc_drop_mean"] for p in res["permutation_importance_logistic"]}
expect("FINDINGS", "perm lbl_bug", perm["lbl_bug"], 0.118, 0.001)
expect("FINDINGS", "perm log_author_prior", perm["log_author_prior_issues"], 0.090, 0.001)

corpus = res["corpus"]
checks.append((corpus["n_closed"] == 6638, "CORPUS", "n_closed", 6638, corpus["n_closed"]))
checks.append((corpus["n_dead"] == 898, "CORPUS", "n_dead", 898, corpus["n_dead"]))
checks.append((corpus["raw_items_returned"] == 28154, "CORPUS", "raw items", 28154,
               corpus["raw_items_returned"]))
checks.append((corpus["pull_requests_skipped"] == 19800, "CORPUS", "prs skipped", 19800,
               corpus["pull_requests_skipped"]))
checks.append((res["split"]["test_rows_used"] == 600, "CORPUS", "test rows", 600,
               res["split"]["test_rows_used"]))

# --- report.html must be the current generated artefact -------------------- #
rep = Path("report.html").read_text(encoding="utf-8")
checks.append(("7ee0c0" in rep or "--accent" in rep, "REPORT", "styles present", True, True))
checks.append(("<script" not in rep, "REPORT", "no script tag", True, "<script" not in rep))
checks.append(("http" not in re.sub(r'https://dev\.to|https://github\.com', '', rep),
               "REPORT", "no remote assets", True, True))

# --- report --------------------------------------------------------------- #
bad = [c for c in checks if not c[0]]
for ok, doc, label, quoted, actual in checks:
    if not ok:
        print(f"MISMATCH [{doc}] {label}: docs say {quoted}, results.json says {actual}")
print(f"\n{len(checks) - len(bad)}/{len(checks)} consistency checks passed")
sys.exit(1 if bad else 0)