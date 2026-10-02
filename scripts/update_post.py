"""Patch the live DEV submission in place.

The body is fetched from DEV, patched with anchored insertions, re-validated against
DEV's field limits, and written back with a PUT. Nothing is retyped, so the published
post cannot drift from POST_DRAFT.md by transcription error.

Usage: python scripts/update_post.py
"""
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

ARTICLE_ID = 4788869
REPO_SLUG = "aniruddhaadak80/triagelens"
RAW = f"https://raw.githubusercontent.com/{REPO_SLUG}/main/assets"
KEY_FILE = Path(os.environ["USERPROFILE"]) / ".devto" / "api-key.txt"

api_key = KEY_FILE.read_text(encoding="utf-8").strip().splitlines()[-1].strip()
HEADERS = {
    "api-key": api_key,
    "Content-Type": "application/json",
    "User-Agent": "Mozilla/5.0",
}

COVER = f"{RAW}/cover.png"
PRECISION = f"{RAW}/precision-at-100.png"
BODY_LEN = f"{RAW}/body-length.png"
CLI = f"{RAW}/cli-output.png"


def request(path, method="GET", payload=None, timeout=90):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        f"https://dev.to{path}", data=data, headers=HEADERS, method=method
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        print(f"HTTP {error.code} on {method} {path}")
        print(error.read().decode("utf-8", "replace")[:600])
        raise SystemExit(1)


# The old title said the rule was checked "with TabPFN" across 6,638 issues. TabPFN was
# fit on 1,500 rows and tested on 600, drawn from that corpus, so the number that is
# actually true is that the rule was tested across the corpus.
TITLE = (
    "Everyone says triage the loudest issues first. I tested it on 6,638 real "
    "issues. It's worse than random."
)

INSERTIONS = [
    (
        "\n## What I Built\n",
        f"\n![Precision@100 of the top 100 issues, by model]({PRECISION})\n\n"
        "*Every bar above is one row of the same held-out comparison: the top 100\n"
        "issues a maintainer would actually look at. Chart generated from\n"
        "`data/results.json`.*\n\n## What I Built\n",
    ),
    (
        "\n{% embed https://triagelens-report-astral-sh-ruff-real-generated-3ykaxpxpc6.openbot.site %}\n",
        f"\n![Terminal output of a real triage run]({CLI})\n\n"
        "*The same run as above, rendered. `report.html` is generated locally and never\n"
        "uploaded anywhere.*\n\n"
        "{% embed https://triagelens-report-astral-sh-ruff-real-generated-3ykaxpxpc6.openbot.site %}\n",
    ),
    (
        "\nAn issue with an empty body is **4.6x less likely to ever ship**.",
        f"\n![Dead rate by issue body length]({BODY_LEN})\n\n"
        "An issue with an empty body is **4.6x less likely to ever ship**.",
    ),
    (
        "\nMIT licensed. 30 tests.",
        "\n{% embed https://github.com/" + REPO_SLUG + " %}\n\nMIT licensed. 30 tests.",
    ),
]


def main():
    article = request(f"/api/articles/{ARTICLE_ID}")
    body = article["body_markdown"]
    before = len(body)

    for anchor, replacement in INSERTIONS:
        if replacement.strip() in body:
            print(f"skip (already present): {anchor.strip()[:46]}")
            continue
        count = body.count(anchor)
        if count != 1:
            print(f"ABORT: anchor matched {count} times, expected 1: {anchor.strip()[:60]}")
            sys.exit(1)
        body = body.replace(anchor, replacement)
        print(f"inserted after: {anchor.strip()[:46]}")

    if not 4 <= len(TITLE) <= 128:
        print(f"ABORT: title length {len(TITLE)} outside 4..128")
        sys.exit(1)
    if len(body) > 40000:
        print(f"ABORT: body length {len(body)} > 40000")
        sys.exit(1)
    if re.search(r"\[\[[A-Z_]+\]\]", body):
        print("ABORT: unfilled placeholder in body")
        sys.exit(1)

    print(f"body {before} -> {len(body)} chars")
    print(f"title {len(TITLE)} chars")

    updated = request(
        f"/api/articles/{ARTICLE_ID}",
        method="PUT",
        payload={
            "article": {
                "id": ARTICLE_ID,
                "title": TITLE,
                "body_markdown": body,
                "published": True,
                "tags": article["tag_list"],
                "description": article.get("description"),
                "cover_image": COVER,
                "main_image": COVER,
                "canonical_url": article.get("canonical_url"),
                "ai_disclosure_level": "some_ai",
            }
        },
    )
    print()
    print("cover_image ", updated.get("cover_image"))
    print("published_at", updated.get("published_at"))
    print("url         ", updated.get("url"))


if __name__ == "__main__":
    main()