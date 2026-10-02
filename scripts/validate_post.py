"""Validate the submission post against DEV's field limits before staging.

DEV rejects violations with a bare HTTP 500 and no error body, so these are
checked locally first.
"""
import re
import sys
from pathlib import Path

src = Path("POST_DRAFT.md").read_text(encoding="utf-8")

fm = re.search(r"```yaml\n(.*?)```", src, re.S).group(1)
title = re.search(r'title:\s*"(.*?)"', fm).group(1)
desc = re.search(r'description:\s*"(.*?)"', fm).group(1)
tags = [t.strip() for t in re.search(r"tags:\s*(.*)", fm).group(1).split(",")]

# The body is everything between the ```markdown fence and the closing fence that
# precedes the publish checklist. The body itself contains nested ``` fences, so a
# non-greedy regex would stop at the first one.
start = src.index("```markdown\n") + len("```markdown\n")
end = src.index("## Publish checklist")
end = src.rindex("```", start, end)
body = src[start:end].rstrip()

print(f"title       {len(title):>6} chars   (min 4, max 128)")
print(f"description {len(desc):>6} chars   (max 100)")
print(f"body        {len(body):>6} chars   (max 40000)")
print(f"tags        {len(tags):>6} count   (max 4)  -> {tags}")
print()

fails = []
if not 4 <= len(title) <= 128:
    fails.append(f"title length {len(title)} outside 4..128")
if len(desc) > 100:
    fails.append(f"description length {len(desc)} > 100")
if len(body) > 40000:
    fails.append(f"body length {len(body)} > 40000")
if len(tags) > 4:
    fails.append(f"{len(tags)} tags > 4")
for t in tags:
    if not re.fullmatch(r"[a-z0-9]+", t):
        fails.append(f"tag {t!r} is not lowercase alphanumeric")
for required in ("devchallenge", "hf26challenge"):
    if required not in tags:
        fails.append(f"missing required tag {required}")

# Placeholders must not survive into a published post.
left = re.findall(r"\[\[[A-Z_]+\]\]", body)
if left:
    fails.append(f"unfilled placeholders in body: {sorted(set(left))}")

print("title:      ", title)
print("description:", desc)
print("tags:       ", tags)
print()
if fails:
    print("FAIL:")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("ALL FIELD CHECKS PASS - safe to stage")