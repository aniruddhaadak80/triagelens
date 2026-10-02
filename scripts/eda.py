"""Quick corpus EDA: label balance, feature availability, time span."""
import json, re
from collections import Counter
from datetime import datetime
from pathlib import Path

d = json.loads((Path(__file__).resolve().parent.parent / "data" / "issues_raw.json")
               .read_text(encoding="utf-8"))
iss = d["issues"]
print(f"repo={d['repo']} kept={d['issues_kept']} retrieved={d['retrieved_at_utc']}")

print("\n--- state / state_reason ---")
for k, v in Counter((i["state"], i["state_reason"]) for i in iss).most_common():
    print(f"  {k}: {v}")

closed = [i for i in iss if i["closed_at"]]
print(f"\nclosed issues: {len(closed)}")

def days(i):
    a = datetime.fromisoformat(i["created_at"].replace("Z", "+00:00"))
    b = datetime.fromisoformat(i["closed_at"].replace("Z", "+00:00"))
    return (b - a).days

d30 = [days(i) for i in closed]
d30.sort()
print(f"days-to-close: min={d30[0]} p25={d30[len(d30)//4]} median={d30[len(d30)//2]} "
      f"p75={d30[3*len(d30)//4]} max={d30[-1]}")

fast = sum(1 for i in closed
           if i["state_reason"] == "completed" and days(i) <= 30)
comp = sum(1 for i in closed if i["state_reason"] == "completed")
print(f"\nclosed & completed: {comp} ({comp/len(closed)*100:.1f}%)")
print(f"fixed_fast (completed & <=30d): {fast} ({fast/len(closed)*100:.1f}% of closed)")

print("\n--- top labels ---")
c = Counter(l for i in iss for l in i["labels"])
for k, v in c.most_common(25):
    print(f"  {k}: {v}")

print("\n--- author_association ---")
for k, v in Counter(i["author_association"] for i in iss).most_common():
    print(f"  {k}: {v}")

print("\n--- time span ---")
created = sorted(i["created_at"] for i in iss)
print(f"  first={created[0]}  last={created[-1]}")

print("\n--- body/title length ---")
bl = sorted(i["body_len"] for i in iss)
print(f"  body_len: min={bl[0]} p25={bl[len(bl)//4]} median={bl[len(bl)//2]} "
      f"p75={bl[3*len(bl)//4]} max={bl[-1]}  zero={sum(1 for x in bl if x==0)}")