"""Upload the generated images to DEV and print their hosted URLs.

DEV takes a base64 payload on POST /api/images and returns a hosted URL. Like every
DEV write, it needs a User-Agent header or it answers a bodiless 403.

Usage: python scripts/upload_images.py
"""
import base64
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
KEY_FILE = Path(os.environ["USERPROFILE"]) / ".devto" / "api-key.txt"

api_key = KEY_FILE.read_text(encoding="utf-8").strip().splitlines()[-1].strip()
HEADERS = {
    "api-key": api_key,
    "Content-Type": "application/json",
    "User-Agent": "Mozilla/5.0",
}

uploaded = {}
for path in sorted((REPO / "assets").glob("*.png")):
    name = path.name
    payload = json.dumps(
        {"image": base64.b64encode(path.read_bytes()).decode("ascii")}
    ).encode("utf-8")
    request = urllib.request.Request(
        "https://dev.to/api/images", data=payload, headers=HEADERS, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            body = json.loads(response.read().decode("utf-8"))
        uploaded[name] = body.get("url") or body.get("image_of")
        print(f"{name:22} -> {uploaded[name]}")
    except urllib.error.HTTPError as error:
        print(f"{name:22} -> HTTP {error.code} {error.read().decode('utf-8','replace')[:200]}")

if not uploaded:
    sys.exit(1)

# A machine-readable map the publish step can read without a human copying URLs.
(REPO / "assets" / "devto_urls.json").write_text(
    json.dumps(uploaded, indent=2) + "\n", encoding="utf-8"
)
print()
print("wrote assets/devto_urls.json")