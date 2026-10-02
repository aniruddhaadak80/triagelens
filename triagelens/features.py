"""Feature extraction for issue triage.

The organising rule: **every feature in `FEATURE_COLUMNS` must be knowable at the
moment the issue is filed.** Nothing that happens after filing may enter the
headline model, because a maintainer deciding what to open next does not have it.

Features that are only knowable after the fact are kept in `LEAKY_COLUMNS` and
are used exclusively to *measure* how much a leak would inflate the score.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from .data import RESOLUTION_TIME_LABELS, triage_time_label_vocab

# --- structural probes on the issue body -------------------------------------

_CODE_FENCE = re.compile(r"```")
_STACKTRACE = re.compile(
    r"(Traceback \(most recent call last\)|error\[E\d+\]|panic:|"
    r"^\s+at [\w.$]+\(|thread 'main' panicked|Stack trace:)",
    re.MULTILINE,
)
_URL = re.compile(r"https?://\S+")
_SEMVER = re.compile(r"\bv?\d+\.\d+(\.\d+)?\b")
_CONFIG_FILE = re.compile(r"\b(pyproject\.toml|ruff\.toml|\.ruff\.toml|setup\.cfg)\b")
_CLI_FLAG = re.compile(r"(^|\s)--[a-z][a-z0-9-]{2,}")
_PY_CODE = re.compile(r"```\s*(python|py|console|text|shell|sh|bash|toml)", re.IGNORECASE)
_CHECKBOX = re.compile(r"^\s*[-*]\s*\[[ xX]\]", re.MULTILINE)
_QUOTE_BLOCK = re.compile(r"^>.*$", re.MULTILINE)

_QUESTION_OPENERS = re.compile(
    r"^\s*(how|why|what|when|where|which|can|could|is|are|does|do|will|would|should)\b",
    re.IGNORECASE,
)


def _log1p(x: float) -> float:
    return math.log1p(max(0.0, float(x)))


def text_features(title: str, body: str) -> dict:
    body = body or ""
    title = title or ""
    fences = _CODE_FENCE.findall(body)
    return {
        "title_len": len(title),
        "body_len": len(body),
        "log_body_len": _log1p(len(body)),
        "n_code_fences": len(fences),
        "has_code_block": 1 if fences else 0,
        "has_stacktrace": 1 if _STACKTRACE.search(body) else 0,
        "has_url": 1 if _URL.search(body) else 0,
        "n_urls": len(_URL.findall(body)),
        "has_version": 1 if _SEMVER.search(body) else 0,
        "has_config_file": 1 if _CONFIG_FILE.search(body) else 0,
        "has_cli_flag": 1 if _CLI_FLAG.search(body) else 0,
        "has_lang_fence": 1 if _PY_CODE.search(body) else 0,
        "has_checkbox": 1 if _CHECKBOX.search(body) else 0,
        "n_quote_lines": len(_QUOTE_BLOCK.findall(body)),
        # Questions are support requests more often than bug reports.
        "title_is_question": 1 if title.strip().endswith("?") else 0,
        "title_starts_question": 1 if _QUESTION_OPENERS.match(title) else 0,
        "title_n_caps": sum(1 for ch in title if ch.isupper()),
    }


def author_features(rec: dict) -> dict:
    # Defaults matter: records being scored (an open backlog) have not been
    # through add_author_history yet. The scorer overwrites these with the values
    # derived from the full corpus; the defaults keep the call total.
    prior = rec.get("author_prior_issues", 0) or 0
    return {
        "author_prior_issues": prior,
        "log_author_prior_issues": _log1p(prior),
        "author_is_first_issue": rec.get("author_is_first_issue", 0),
        "author_is_frequent": rec.get("author_is_frequent", 0),
    }


def time_features(rec: dict) -> dict:
    created = rec.get("created")
    if created is None:
        return {"created_hour": 0, "created_dow": 0, "is_weekend": 0,
                "issue_number": rec.get("number", 0),
                "log_issue_number": _log1p(rec.get("number", 0))}
    return {
        "created_hour": created.hour,
        "created_dow": created.weekday(),
        "is_weekend": 1 if created.weekday() >= 5 else 0,
        "issue_number": rec.get("number", 0),
        "log_issue_number": _log1p(rec.get("number", 0)),
    }


def label_features(rec: dict, vocab: list[str]) -> dict:
    low = {lb.lower() for lb in (rec.get("labels") or [])}
    n = rec.get("n_labels", len(rec.get("labels") or [])) or 0
    out = {"n_labels": n, "log_n_labels": _log1p(n)}
    for lb in vocab:
        out[f"lbl_{lb}"] = 1 if lb in low else 0
    return out


def leaky_features(rec: dict) -> dict:
    """Knowable only after the issue has run its course. Never in the headline."""
    low = {lb.lower() for lb in rec["labels"]}
    out = {
        "comments": rec["comments"],
        "log_comments": _log1p(rec["comments"]),
        "reactions_total": rec["reactions_total"],
        "reactions_heart": rec.get("reactions_heart", 0),
        "days_to_close": rec["days_to_close"],
    }
    for lb in sorted(RESOLUTION_TIME_LABELS):
        out[f"reslbl_{lb}"] = 1 if lb in low else 0
    return out


def build_matrix(records: list[dict], include_leaky: bool = False,
                 label_vocab_size: int = 18,
                 vocab: list[str] | None = None) -> tuple[list[dict], list[str]]:
    """Return (rows, feature_names) for the given records.

    Pass `vocab` explicitly when scoring a batch that was not drawn from the
    training corpus, so the label columns line up with the model instead of being
    re-derived from whatever labels happen to appear in the batch.
    """
    if vocab is None:
        counts = triage_time_label_vocab(records)
        vocab = [lb for lb, _ in counts.most_common(label_vocab_size)]

    rows = []
    for rec in records:
        row = {}
        row.update(text_features(rec["title"], rec.get("body", "")))
        row.update(author_features(rec))
        row.update(time_features(rec))
        row.update(label_features(rec, vocab))
        if include_leaky:
            row.update(leaky_features(rec))
        rows.append(row)

    names = list(rows[0].keys()) if rows else []
    return rows, names


def label_vocabulary(records: list[dict], label_vocab_size: int = 18) -> list[str]:
    return [lb for lb, _ in triage_time_label_vocab(records).most_common(label_vocab_size)]


def leak_columns(names: list[str]) -> list[str]:
    return [n for n in names if n in {"comments", "log_comments",
                                      "reactions_total", "reactions_heart",
                                      "days_to_close"}
            or n.startswith("reslbl_")]