"""Publish the staged DEV submission straight from POST_DRAFT.md.

The body is read from disk and serialised in Python so the ~12 KB markdown is never
retyped or round-tripped through a shell JSON encoder. DEV answers a malformed
payload with a bare HTTP 500, so the payload is validated locally first.

Usage: python scripts/publish_submission.py
"""
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DRAFT = REPO / "POST_DRAFT.md"
KEY_FILE = Path(os.environ["USERPROFILE"]) / ".devto" / "api-key.txt"

api_key = KEY_FILE.read_text(encoding="utf-8").strip().splitlines()[-1].strip()

src = DRAFT.read_text(encoding="utf-8")
frontmatter = re.search(r"```yaml\n(.*?)```", src, re.S).group(1)
title = re.search(r'title:\s*"(.*?)"', frontmatter).group(1)
description = re.search(r'description:\s*"(.*?)"', frontmatter).group(1)
tags = [t.strip() for t in re.search(r"tags:\s*(.*)", frontmatter).group(1).split(",")]

# The body sits between the ```markdown fence and the fence that closes it, which
# precedes the publish checklist. Nested fences mean a non-greedy match is wrong.
start = src.index("```markdown\n") + len("```markdown\n")
end = src.rindex("```", start, src.index("## Publish checklist"))
body = src[start:end].rstrip()

payload = json.dumps(
    {
        "article": {
            "title": title,
            "description": description,
            "body_markdown": body,
            "tags": tags,
            "published": True,
        }
    }
)

# Guard against the failure modes that surface as a bare 500 on DEV.
roundtrip = json.loads(payload)["article"]
problems = []
if not isinstance(roundtrip["body_markdown"], str):
    problems.append("body_markdown did not survive as a string")
if not 4 <= len(title) <= 128:
    problems.append(f"title length {len(title)} outside 4..128")
if len(description) > 100:
    problems.append(f"description length {len(description)} > 100")
if len(roundtrip["body_markdown"]) > 40000:
    problems.append(f"body length {len(roundtrip['body_markdown'])} > 40000")
if len(tags) > 4:
    problems.append(f"{len(tags)} tags > 4")
for tag in tags:
    if not re.fullmatch(r"[a-z0-9]+", tag):
        problems.append(f"tag {tag!r} is not lowercase alphanumeric")
for required in ("devchallenge", "hf26challenge"):
    if required not in tags:
        problems.append(f"missing required tag {required}")
if re.search(r"\[\[[A-Z_]+\]\]", roundtrip["body_markdown"]):
    problems.append("unfilled [[PLACEHOLDER]] in body")
if len(payload) >= 40000:
    problems.append(f"payload {len(payload)} bytes too large")

if problems:
    print("REFUSING TO POST:")
    for problem in problems:
        print("  -", problem)
    sys.exit(1)

print(f"title       {len(title)} chars")
print(f"description {len(description)} chars")
print(f"body        {len(body)} chars")
print(f"tags        {tags}")
print(f"payload     {len(payload)} bytes")
print("posting...")

request = urllib.request.Request(
    "https://dev.to/api/articles",
    data=payload.encode("utf-8"),
    headers={
        "api-key": api_key,
        "Content-Type": "application/json",
        # DEV answers a write without a User-Agent with a bodiless 403, which is
        # indistinguishable from a bad key. Verified: 403 without, 201 with.
        "User-Agent": "Mozilla/5.0",
    },
    method="POST",
)

try:
    with urllib.request.urlopen(request, timeout=90) as response:
        article = json.loads(response.read().decode("utf-8"))
except urllib.error.HTTPError as error:
    print(f"HTTP {error.code}")
    print(error.read().decode("utf-8", "replace")[:2000])
    sys.exit(1)

print()
print("id        ", article["id"])
print("published ", article["published"])
print("url       ", article["url"])
print("tags      ", article["tag_list"])