"""Build the static demo site deployed to Vercel.

Runs the tool for real, then assembles site/ from its actual output rather than a
hand-maintained copy, so the deployed demo cannot drift from the code.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site"


def main() -> int:
    SITE.mkdir(exist_ok=True)

    print("regenerating the report from a clean run ...")
    r = subprocess.run(
        [sys.executable, "-m", "triagelens.cli", "triage",
         "--model", "logistic", "--top", "40", "--out", "report.html"],
        cwd=ROOT, capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-2000:])
        return 1
    print(r.stdout.strip().splitlines()[-1])

    shutil.copyfile(ROOT / "report.html", SITE / "index.html")
    print(f"wrote {SITE / 'index.html'} "
          f"({(SITE / 'index.html').stat().st_size // 1024} KB)")

    # Static-only: no build step, no functions, no framework.
    (SITE / "vercel.json").write_text(json.dumps({
        "$schema": "https://openapi.vercel.sh/vercel.json",
        "cleanUrls": True,
        "trailingSlash": False,
        "headers": [{
            "source": "/(.*)",
            "headers": [
                {"key": "X-Content-Type-Options", "value": "nosniff"},
                {"key": "Referrer-Policy", "value": "strict-origin-when-cross-origin"},
            ],
        }],
    }, indent=2) + "\n", encoding="utf-8")

    # Keep the corpus out of the deployed bundle; it lives in the GitHub repo.
    total = sum(p.stat().st_size for p in SITE.rglob("*") if p.is_file())
    print(f"site bundle: {total // 1024} KB across "
          f"{sum(1 for p in SITE.rglob('*') if p.is_file())} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())