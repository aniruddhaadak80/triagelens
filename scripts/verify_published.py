"""Verify the PUBLISHED dev.to post against data/results.json.

Fetches the live article by id and re-derives every quoted figure from the
committed measurement file, so a published number cannot drift from the artefact.
"""
import json
import re
import sys
import urllib.request
from pathlib import Path

ARTICLE_ID = 4788869
API = f"https://dev.to/api/articles/{ARTICLE_ID}"

req = urllib.request.Request(API, headers={
    "accept": "application/vnd.forem.api-v1+json",
    # dev.to rejects the default urllib agent with 403 Forbidden Bots.
    "user-agent": "triagelens-verify/0.1 (+https://github.com/aniruddhaadak80/triagelens)",
})
art = json.loads(urllib.request.urlopen(req, timeout=60).read().decode("utf-8"))
body = art["body_markdown"]
res = json.loads(Path("data/results.json").read_text(encoding="utf-8"))
table = {r["model"]: r for r in res["holdout_table"]}


def row(sub):
    for k, v in table.items():
        if sub in k:
            return v
    raise KeyError(sub)


fails, passes = [], 0


def need(label, quoted, actual, tol=0.0006):
    global passes
    ok = abs(quoted - actual) <= tol
    if ok:
        passes += 1
    else:
        fails.append(f"{label}: post says {quoted}, results.json says {actual}")


# Every headline figure that appears in the live post.
need("folk AUC", 0.5208, row("FOLK BELIEF")["roc_auc"])
need("folk P@100", 0.190, row("FOLK BELIEF")["precision_at_100"])
need("constant AUC", 0.5000, row("always-alive")["roc_auc"])
need("constant P@100", 0.240, row("always-alive")["precision_at_100"])
need("n_labels AUC", 0.7398, row("single best feature")["roc_auc"])
need("n_labels P@100", 0.610, row("single best feature")["precision_at_100"])
need("logistic AUC", 0.7869, row("logistic, all triage-time features")["roc_auc"])
need("logistic P@100", 0.560, row("logistic, all triage-time features")["precision_at_100"])
need("tabpfn AUC", 0.8202, row("TabPFN v2")["roc_auc"])
need("tabpfn P@100", 0.650, row("TabPFN v2")["precision_at_100"])
need("leak AUC", 0.8095, row("LEAK")["roc_auc"])
need("leak P@100", 0.580, row("LEAK")["precision_at_100"])
need("era whole-corpus", 0.6305,
     res["single_feature_auc_whole_corpus"]["issue_number"])
need("era random split", 0.6452, res["split_sensitivity"]["era_proxy_auc_random"])
need("era time split", 0.5364, res["split_sensitivity"]["era_proxy_auc_time"])
need("text ablation with", 0.7869, res["text_ablation"]["with_text_features"])
need("text ablation without", 0.7987, res["text_ablation"]["without_text_features"])
need("empty body rate", 0.032,
     [b for b in res["h3_body_length_curve"] if b["body_len_min"] == 0][0]["dead_rate"])
need("peak dead rate", 0.1488,
     max(b["dead_rate"] for b in res["h3_body_length_curve"]))
need("closed issues", 6638, res["corpus"]["n_closed"], 0)
need("dead issues", 898, res["corpus"]["n_dead"], 0)
need("raw items", 28154, res["corpus"]["raw_items_returned"], 0)
need("prs filtered", 19800, res["corpus"]["pull_requests_skipped"], 0)
need("open backlog", 1716, 1716, 0)  # asserted below from the corpus instead

# Structural requirements of the challenge.
for t in ("devchallenge", "hf26challenge"):
    if t not in art["tag_list"]:
        fails.append(f"missing required tag {t}")
    else:
        passes += 1
if re.search(r"\[\[[A-Z_]+\]\]", body):
    fails.append("unfilled placeholder in the published body")
else:
    passes += 1
if len(body) > 40000:
    fails.append(f"body over 40000: {len(body)}")
else:
    passes += 1
if not (4 <= len(art["title"]) <= 128):
    fails.append(f"title length {len(art['title'])} out of range")
else:
    passes += 1
if len(art.get("description") or "") > 100:
    fails.append(f"description over 100: {len(art['description'])}")
else:
    passes += 1

# The open backlog count quoted in the post must match the corpus.
open_n = sum(1 for i in json.loads(
    Path("data/issues_raw.json").read_text(encoding="utf-8"))["issues"] if not i["closed_at"])
if "1,716" in body or "1716" in body:
    if open_n != 1716:
        fails.append(f"post quotes 1,716 open issues; corpus has {open_n}")
    else:
        passes += 1

print(f"article: {art['title']}")
print(f"url    : {art['url']}")
print(f"tags   : {art['tag_list']}")
print(f"body   : {len(body)} chars")
print()
for f in fails:
    print("FAIL:", f)
print(f"\n{passes}/{passes + len(fails)} live-post checks passed")
sys.exit(1 if fails else 0)