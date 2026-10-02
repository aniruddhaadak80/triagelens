"""Fetch the full issue corpus for a repo from the GitHub REST API.

Stores raw JSON plus provenance so every number in the write-up is traceable.
"""
import json, os, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

REPO = sys.argv[1] if len(sys.argv) > 1 else "astral-sh/ruff"
OUT = Path(__file__).resolve().parent.parent / "data"
OUT.mkdir(parents=True, exist_ok=True)


def gh_json(url):
    """Call the GitHub API via `gh` (token never touches disk)."""
    r = subprocess.run(
        ["gh", "api", "--paginate", url, "-H", "Accept: application/vnd.github+json"],
        capture_output=True, text=True, encoding="utf-8",
    )
    if r.returncode != 0:
        raise RuntimeError(f"gh api failed: {r.stderr[:400]}")
    # --paginate concatenates JSON arrays; merge them.
    data = json.loads(r.stdout)
    if isinstance(data, list) and data and isinstance(data[0], list):
        merged = []
        for chunk in data:
            merged.extend(chunk)
        return merged
    return data


print(f"fetching issues for {REPO} ...", flush=True)
t0 = time.time()
# sort=created&direction=asc gives a stable, resumable order.
raw = gh_json(f"repos/{REPO}/issues?state=all&per_page=100&sort=created&direction=asc")
print(f"  raw items: {len(raw)}  ({time.time()-t0:.1f}s)", flush=True)

issues = []
prs_skipped = 0
for it in raw:
    if "pull_request" in it:          # the issues endpoint also returns PRs
        prs_skipped += 1
        continue
    reactions = it.get("reactions") or {}
    issues.append({
        "number": it["number"],
        "title": it.get("title") or "",
        # Truncated body: enough for structural features (code fences, stack
        # traces, config files) without bloating the committed artefact.
        "body": (it.get("body") or "")[:4000],
        "body_len": len(it.get("body") or ""),
        "title_len": len(it.get("title") or ""),
        "state": it.get("state"),
        "state_reason": it.get("state_reason"),
        "labels": [lb.get("name") for lb in (it.get("labels") or [])],
        "n_labels": len(it.get("labels") or []),
        "comments": it.get("comments") or 0,
        "reactions_total": reactions.get("total_count", 0),
        "reactions_plus1": reactions.get("+1", 0),
        "reactions_heart": reactions.get("heart", 0),
        "created_at": it.get("created_at"),
        "closed_at": it.get("closed_at"),
        "updated_at": it.get("updated_at"),
        "author_login": ((it.get("user") or {}) or {}).get("login"),
        "is_locked": bool(it.get("locked")),
    })

payload = {
    "repo": REPO,
    "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
    "api": "GET /repos/{repo}/issues?state=all&per_page=100&sort=created&direction=asc",
    "raw_items_returned": len(raw),
    "pull_requests_skipped": prs_skipped,
    "issues_kept": len(issues),
    "issues": issues,
}
(OUT / "issues_raw.json").write_text(
    json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8"
)
print(f"  kept {len(issues)} issues, skipped {prs_skipped} PRs", flush=True)
print(f"  wrote {OUT / 'issues_raw.json'}  total {time.time()-t0:.1f}s", flush=True)