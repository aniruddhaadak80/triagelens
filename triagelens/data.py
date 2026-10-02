"""Load a raw issue corpus and derive labels and author-history features.

Everything here is deterministic and depends only on `data/issues_raw.json`,
so every number in the write-up is reproducible from the committed artefact.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Labels a maintainer applies while CLOSING an issue. These are a consequence of
# the outcome, so they must never enter a model that claims to predict it.
RESOLUTION_TIME_LABELS = {"accepted", "fixes", "fixed", "rule", "preview", "ty"}

DEAD_REASONS = {"not_planned", "duplicate"}


def _parse(ts: str | None) -> datetime | None:
    if not ts:
        return None
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def load_raw(path: Path | None = None) -> dict:
    path = path or (DATA_DIR / "issues_raw.json")
    return json.loads(path.read_text(encoding="utf-8"))


def build_frame(path: Path | None = None) -> list[dict]:
    """Return one record per closed issue, sorted by creation time.

    Open issues have no outcome yet, so they are excluded from fitting and
    scoring. They are kept separately later for prediction.
    """
    raw = load_raw(path)
    records = []
    for it in raw["issues"]:
        if not it["closed_at"]:
            continue
        created, closed = _parse(it["created_at"]), _parse(it["closed_at"])
        records.append(
            {
                "number": it["number"],
                "title": it["title"],
                "body": it.get("body", ""),
                "body_len": it["body_len"],
                "title_len": it["title_len"],
                "state_reason": it["state_reason"],
                "labels": it["labels"],
                "n_labels": it["n_labels"],
                "comments": it["comments"],
                "reactions_total": it["reactions_total"],
                "reactions_heart": it.get("reactions_heart", 0),
                "created_at": it["created_at"],
                "closed_at": it["closed_at"],
                "author_login": it.get("author_login"),
                "created": created,
                "days_to_close": (closed - created).days,
                # PRIMARY LABEL: closed without shipping a fix.
                "dead": 1 if it["state_reason"] in DEAD_REASONS else 0,
                # SECONDARY LABEL: shipped, but slowly.
                "slow": 1
                if (
                    it["state_reason"] == "completed"
                    and (closed - created).days > 30
                )
                else 0,
            }
        )
    records.sort(key=lambda r: r["created"])
    return records


def add_author_history(records: list[dict]) -> list[dict]:
    """Attach strictly-at-creation author signals.

    Two features that are knowable the moment an issue is filed:

    * ``author_prior_issues`` -- how many issues this author had already filed
      in this corpus before this one.
    * ``author_is_first_issue`` -- is this the author's first issue here?

    Both are computed from the corpus itself in creation order, so they never
    look at anything that happened after this issue was filed.
    """
    seen: Counter = Counter()
    for r in records:
        login = r["author_login"]
        prior = seen[login]
        r["author_prior_issues"] = prior
        r["author_is_first_issue"] = 1 if prior == 0 else 0
        seen[login] += 1

    # A handful of authors account for a large share of issues. Knowing whether
    # someone is a habitual reporter of this project is triage-relevant.
    totals = Counter(r["author_login"] for r in records)
    ranked = sorted(totals.values(), reverse=True)
    idx = min(len(ranked) - 1, max(0, len(ranked) // 50))
    cutoff = ranked[idx] if ranked else 0
    for r in records:
        r["author_total_issues"] = totals[r["author_login"]]
        r["author_is_frequent"] = 1 if totals[r["author_login"]] >= cutoff else 0
    return records


def label_counts(records: list[dict]) -> dict:
    dead = sum(r["dead"] for r in records)
    return {
        "n_closed": len(records),
        "n_dead": dead,
        "n_alive": len(records) - dead,
        "dead_rate": dead / len(records) if records else 0.0,
        "n_slow": sum(r["slow"] for r in records),
        "span_first": records[0]["created_at"] if records else None,
        "span_last": records[-1]["created_at"] if records else None,
    }


def open_issues(path: Path | None = None) -> list[dict]:
    """Issues with no outcome yet -- the backlog a maintainer wants to triage."""
    raw = load_raw(path)
    out = []
    for it in raw["issues"]:
        if it["closed_at"]:
            continue
        out.append(
            {
                "number": it["number"],
                "title": it["title"],
                "body": it.get("body", ""),
                "body_len": it["body_len"],
                "title_len": it["title_len"],
                "labels": it["labels"],
                "n_labels": it["n_labels"],
                "comments": it["comments"],
                "reactions_total": it["reactions_total"],
                "reactions_heart": it.get("reactions_heart", 0),
                "created_at": it["created_at"],
                "closed_at": None,
                "author_login": it.get("author_login"),
                "created": _parse(it["created_at"]),
                "days_to_close": None,
                "dead": None,
                "slow": None,
            }
        )
    out.sort(key=lambda r: r["created"])
    return out


def resolution_time_labels_used(records: list[dict]) -> Counter:
    c: Counter = Counter()
    for r in records:
        for lb in r["labels"]:
            if lb.lower() in RESOLUTION_TIME_LABELS:
                c[lb] += 1
    return c


def triage_time_label_vocab(records: list[dict]) -> Counter:
    c: Counter = Counter()
    for r in records:
        for lb in r["labels"]:
            if lb.lower() not in RESOLUTION_TIME_LABELS:
                c[lb] += 1
    return c