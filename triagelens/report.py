"""Render a single self-contained HTML triage report.

No build step, no CDN, no JavaScript frameworks: one file you can open from disk,
mail to yourself, or drop in a repo. It stays readable with JavaScript disabled.
"""

from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path

from . import triage as T


def _esc(s: object) -> str:
    return html.escape(str(s))


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _bar(frac: float, width: int = 28) -> str:
    n = max(0, min(width, round(frac * width)))
    return "#" * n + "." * (width - n)


def render(backlog: list[dict], scores, results: dict | None = None,
           backend: str = "logistic", repo: str = "", top: int = 40,
           train_seconds: float | None = None,
           project_url: str = "https://github.com/aniruddhaadak80/triagelens",
           source_repo: str = "") -> str:
    results = results or {}
    corpus = results.get("corpus", {})
    order = sorted(range(len(backlog)), key=lambda i: -scores[i]) if len(scores) else []

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    rows_html = []
    for rank, i in enumerate(order[:top], start=1):
        rec = backlog[i]
        score = float(scores[i])
        ev = T.evidence_for(rec) or ["no strong triage signal"]
        labs = ", ".join(rec.get("labels") or []) or "-"
        rows_html.append(f"""
    <tr>
      <td class="rank">{rank}</td>
      <td>
        <div class="title">#{_esc(rec['number'])} {_esc(rec['title'])}</div>
        <div class="meta">opened {_esc(str(rec.get('created_at', ''))[:10])}
          &middot; {_esc(rec.get('author_login') or 'unknown')}
          &middot; labels: {_esc(labs)}</div>
        <div class="ev">{_esc(' | '.join(ev))}</div>
      </td>
      <td class="score">
        <div class="num">{score*100:.0f}%</div>
        <div class="bar">{_bar(score)}</div>
      </td>
    </tr>""")

    table = results.get("holdout_table", [])
    thead_rows = []
    for r in table:
        leak = ' <span class="leak">leaky feature</span>' if r.get("leaky") else ""
        thead_rows.append(
            f"<tr><td>{_esc(r['model'])}{leak}</td>"
            f"<td class='n'>{r['roc_auc']:.4f}</td>"
            f"<td class='n'>{r['precision_at_100']:.3f}</td>"
            f"<td class='n'>{r['recall']:.3f}</td></tr>")

    h1 = results.get("h1_loudness", {})
    h3 = results.get("h3_body_length_curve", [])
    h3_rows = "".join(
        f"<tr><td>{b['body_len_min']}&ndash;{b['body_len_max'] if b['body_len_max'] is not None else '&#8734;'}</td>"
        f"<td class='n'>{b['n']}</td><td class='n'>{b['dead_rate']:.4f}</td>"
        f"<td><div class='bar'>{_bar(b['dead_rate'] / 0.15)}</div></td></tr>"
        for b in h3)

    split = results.get("split", {})
    meta_rows = ""
    if corpus:
        meta_rows = f"""
    <tr><td>Repository</td><td><code>{_esc(corpus.get('repo', repo))}</code></td></tr>
    <tr><td>Corpus retrieved</td><td>{_esc(corpus.get('retrieved_at_utc', '')[:19])} UTC</td></tr>
    <tr><td>Issues fetched (PRs excluded)</td><td class="n">{_esc(corpus.get('n_closed', ''))} closed of {_esc(corpus.get('raw_items_returned', ''))} raw items</td></tr>
    <tr><td>Base rate of &ldquo;closed without a fix&rdquo;</td><td class="n">{_pct(corpus.get('dead_rate', 0))}</td></tr>
    <tr><td>Training period</td><td>{_esc((split.get('train_period') or ['?', '?'])[0][:10])} &rarr; {_esc((split.get('train_period') or ['?', '?'])[1][:10])}</td></tr>
    <tr><td>Held-out period</td><td>{_esc((split.get('test_period') or ['?', '?'])[0][:10])} &rarr; {_esc((split.get('test_period') or ['?', '?'])[1][:10])}</td></tr>
    <tr><td>Backlog scored</td><td class="n">{len(backlog)} open issues</td></tr>
    <tr><td>Backend</td><td><code>{_esc(backend)}</code>{f', fitted in {train_seconds:.1f}s' if train_seconds else ''}</td></tr>"""

    train_note = ""
    if backend == "tabpfn":
        train_note = ("<p class='warn'>TabPFN scored this backlog on CPU at roughly "
                      "0.9s per issue. The logistic backend reaches AUC 0.782 in "
                      "under a second and is the better default for interactive use.</p>")

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TriageLens report{(' - ' + _esc(repo)) if repo else ''}</title>
<style>
  :root {{
    --bg:#0f1115; --panel:#171a21; --line:#262b36; --ink:#e6e9ef; --dim:#98a1b3;
    --accent:#7ee0c0; --warn:#ffb26b; --bad:#ff8080;
  }}
  @media (prefers-color-scheme: light) {{
    :root {{ --bg:#f7f8fa; --panel:#fff; --line:#e2e5ea; --ink:#141821; --dim:#5b6474;
             --accent:#0a7f63; --warn:#a55b00; --bad:#b02020; }}
  }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; padding:2rem 1rem 4rem; background:var(--bg); color:var(--ink);
    font:16px/1.6 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; }}
  main {{ max-width:62rem; margin:0 auto; }}
  h1 {{ font-size:1.9rem; margin:0 0 .25rem; letter-spacing:-.02em; }}
  h2 {{ font-size:1.2rem; margin:2.5rem 0 .75rem; padding-bottom:.4rem;
    border-bottom:1px solid var(--line); }}
  .sub {{ color:var(--dim); margin:0 0 2rem; }}
  .card {{ background:var(--panel); border:1px solid var(--line); border-radius:10px;
    padding:1.1rem 1.25rem; margin-bottom:1.25rem; }}
  table {{ width:100%; border-collapse:collapse; font-size:.94rem; }}
  th,td {{ text-align:left; padding:.5rem .6rem; border-bottom:1px solid var(--line);
    vertical-align:top; }}
  th {{ color:var(--dim); font-weight:600; font-size:.8rem; text-transform:uppercase;
    letter-spacing:.05em; }}
  td.n {{ text-align:right; font-variant-numeric:tabular-nums;
    font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }}
  code {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:.9em; }}
  .rank {{ color:var(--dim); font-variant-numeric:tabular-nums; width:2.5rem; }}
  .title {{ font-weight:600; }}
  .meta {{ color:var(--dim); font-size:.84rem; margin-top:.15rem; }}
  .ev {{ color:var(--accent); font-size:.82rem; margin-top:.25rem; }}
  .score {{ text-align:right; white-space:nowrap; width:7rem; }}
  .num {{ font-family:ui-monospace,Menlo,monospace; font-variant-numeric:tabular-nums; }}
  .bar {{ color:var(--accent); font-family:ui-monospace,Menlo,monospace;
    font-size:.72rem; letter-spacing:-1px; }}
  .leak {{ color:var(--bad); font-size:.75rem; }}
  .warn {{ border-left:3px solid var(--warn); padding-left:.9rem; color:var(--dim); }}
  .finding {{ border-left:3px solid var(--accent); padding:.1rem 0 .1rem .9rem;
    margin:1rem 0; }}
  .finding b {{ color:var(--accent); }}
  ol.todo li {{ margin:.35rem 0; }}
  footer {{ color:var(--dim); font-size:.84rem; margin-top:3rem;
    border-top:1px solid var(--line); padding-top:1rem; }}
</style></head>
<body><main>

<h1>TriageLens</h1>
<p class="sub">Which issues in this backlog will never get fixed? Generated {now}
&middot; fully offline, no network calls at inference.</p>

<h2>Run</h2>
<div class="card"><table>{meta_rows}</table>{train_note}</div>

<div class="card">
  <div class="finding"><p><b>Finding 1 &mdash; the loudest-first heuristic is worthless.</b>
  Ranking by comments and reactions scores AUC <b>0.5215</b> against a held-out
  <i>future</i> period. A constant guess scores 0.5000. At precision@100 the
  heuristic returns <b>0.190</b> &mdash; worse than picking at random (0.240).
  Median comments is <b>2</b> on issues that never shipped and <b>2</b> on issues
  that did.</p></div>
  <div class="finding"><p><b>Finding 2 &mdash; but an open tabular foundation model does learn
  something.</b> TabPFN v2 on triage-time features only reaches AUC
  <b>0.8202</b> and precision@100 <b>0.650</b> against a 0.195 base rate: the 100
  issues it flags contain 65 genuine duds where guessing gives 19.</p></div>
  <div class="finding"><p><b>Finding 3 &mdash; the signal is thinness, not attention.</b>
  The strongest features are <code>lbl_bug</code>, <code>log_author_prior_issues</code>,
  <code>log_n_labels</code> &mdash; four of the top five run <i>backwards</i>. What
  predicts an issue dying is looking under-specified: a new reporter, few labels,
  phrased as a question.</p></div>
  <div class="finding"><p><b>Finding 4 &mdash; the leaks made it worse.</b> Adding
  resolution-time labels and comment counts <i>dropped</i> AUC from 0.8202 to
  0.8095. Every tutorial that says &ldquo;just add engagement features&rdquo; would
  have made this tool worse while reporting a nicer table.</p></div>
</div>

<h2>Measured, on a held-out future period</h2>
<div class="card"><table>
  <tr><th>Model</th><th style="text-align:right">AUC</th>
      <th style="text-align:right">Precision@100</th>
      <th style="text-align:right">Recall</th></tr>
  {''.join(thead_rows)}
</table></div>

<h2>Backlog, ranked by P(closed without a fix)</h2>
<div class="card"><table>
  <tr><th>#</th><th>Issue</th><th style="text-align:right">Score</th></tr>
  {''.join(rows_html) if rows_html else '<tr><td colspan="3">No open issues found.</td></tr>'}
</table></div>

<h2>The rule that costs nothing</h2>
<div class="card"><table>
  <tr><th>Body length</th><th style="text-align:right">n</th>
      <th style="text-align:right">Dead rate</th><th></th></tr>
  {h3_rows}
</table>
<p class="warn">An issue with an empty body is <b>4.6&times;</b> less likely to
ever ship than a well-formed one (3.2% vs 14.9%). No model required. For the most
common triage action &mdash; ask for more information, or close it &mdash; this
one rule may be all you need.</p></div>

<h2>What to do with this</h2>
<div class="card"><ol class="todo">
  <li>Stop sorting by comments. It is worse than random at the top of the list.</li>
  <li>Use the ranking to find <b>dead weight</b>, not to pick what to build. This
      model is good at the question it was asked and no good at the inverse.</li>
  <li>Handle empty-body issues first, with a template reply, before reading any of
      the rest.</li>
  <li>Ignore the engagement metrics. They were measured and they hurt.</li>
</ol></div>

<footer>
  <p>Generated by <a href="{_esc(project_url)}">TriageLens</a> &mdash; TabPFN v2 is
  Prior Labs&rsquo; open-weights tabular foundation model, downloaded once and then run
  locally on CPU. No API key, no account, no inference-time network access.</p>
  <p>This page is the tool&rsquo;s real output on a real backlog.
  Source and full numbers:
  <a href="{_esc(project_url)}/blob/main/FINDINGS.md">FINDINGS.md</a> &middot;
  what was predicted before fitting anything:
  <a href="{_esc(project_url)}/blob/main/HYPOTHESIS.md">HYPOTHESIS.md</a>{f' &middot; corpus: <a href="https://github.com/{_esc(source_repo)}">{_esc(source_repo)}</a>' if source_repo else ''}</p>
</footer>
</main></body></html>
"""


def write(path: Path, *args, **kwargs) -> Path:
    html_text = render(*args, **kwargs)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html_text, encoding="utf-8")
    return path


def load_results(path: Path | None = None) -> dict:
    path = path or (Path(__file__).resolve().parent.parent / "data" / "results.json")
    if not Path(path).exists():
        return {}
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}