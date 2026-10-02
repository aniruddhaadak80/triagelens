"""Command line interface.

    python -m triagelens.cli fetch astral-sh/ruff
    python -m triagelens.cli triage --model logistic --top 40
    python -m triagelens.cli triage --model tabpfn --top 40
    python -m triagelens.cli report
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from triagelens import data as D          # noqa: E402
from triagelens import report as R        # noqa: E402
from triagelens import triage as T        # noqa: E402


def cmd_fetch(args: argparse.Namespace) -> int:
    script = ROOT / "scripts" / "fetch_issues.py"
    cmd = [sys.executable, str(script), args.repo]
    print(f"fetching {args.repo} via the GitHub API (gh must be authenticated)...")
    return subprocess.call(cmd)


def cmd_triage(args: argparse.Namespace) -> int:
    raw = D.load_raw()
    closed = D.build_frame()
    D.add_author_history(closed)
    backlog = D.open_issues()
    if args.limit:
        backlog = backlog[: args.limit]

    print(f"training on {len(closed)} closed issues ({D.label_counts(closed)['n_dead']} "
          f"closed without a fix), backend={args.model}")
    scorer = T.TriageScorer(backend=args.model, max_train=args.max_train)
    scorer.fit(closed)
    if scorer.train_seconds is not None:
        print(f"  fitted in {scorer.train_seconds:.1f}s on "
              f"{scorer.train_n} rows / {len(scorer.cols)} features")

    print(f"scoring {len(backlog)} open issues...")
    scores = scorer.score(backlog, records=closed)
    ood = getattr(scorer, "last_ood_fraction", 0.0)
    if ood > 0:
        print(f"  warning: {ood*100:.0f}% of the backlog predates the training "
              f"window (issue numbers below {scorer.train_issue_range[0]}). "
              f"Their scores are extrapolation and should not be trusted.")
    out = R.write(ROOT / args.out, backlog, scores,
                  results=R.load_results(), backend=args.model,
                  repo=raw["repo"], top=args.top,
                  train_seconds=scorer.train_seconds)
    print(f"wrote {out}")

    print(f"\nTop {min(10, len(backlog))} most likely to be closed without a fix:")
    for i in sorted(range(len(backlog)), key=lambda k: -scores[k])[:10]:
        ev = T.evidence_for(backlog[i])
        print(f"  {scores[i]*100:5.1f}%  #{backlog[i]['number']} "
              f"{backlog[i]['title'][:64]}")
        if ev:
            print(f"          {'; '.join(ev)}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    closed = D.build_frame()
    D.add_author_history(closed)
    backlog = D.open_issues()[: args.limit]
    scorer = T.TriageScorer(backend=args.model, max_train=args.max_train)
    scorer.fit(closed)
    scores = scorer.score(backlog, records=closed)
    out = R.write(ROOT / args.out, backlog, scores, results=R.load_results(),
                  backend=args.model, repo=D.load_raw()["repo"], top=args.top,
                  train_seconds=scorer.train_seconds)
    print(f"wrote {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="triagelens",
        description="Offline issue triage for a solo maintainer.")
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="download an issue corpus via the GitHub API")
    f.add_argument("repo", help="e.g. astral-sh/ruff")
    f.set_defaults(func=cmd_fetch)

    for name, fn, helptext in (("triage", cmd_triage, "score the backlog and write a report"),
                               ("report", cmd_report, "alias for triage")):
        s = sub.add_parser(name, help=helptext)
        s.add_argument("--model", default="logistic", choices=["logistic", "tabpfn"],
                       help="logistic is instant; tabpfn is more accurate but slow on CPU")
        s.add_argument("--top", type=int, default=40, help="rows in the HTML report")
        s.add_argument("--limit", type=int, default=0,
                       help="score only the N oldest open issues (0 = all)")
        s.add_argument("--max-train", type=int, default=3000,
                       help="most recent closed issues to train on")
        s.add_argument("--out", default="report.html")
        s.set_defaults(func=fn)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())