# Hypothesis — written before any model was fitted

**Date:** 2026-10-02
**Corpus:** `astral-sh/ruff`, 8,354 real issues, retrieved 2026-10-02 via
`GET /repos/astral-sh/ruff/issues?state=all&per_page=100&sort=created&direction=asc`
(19,800 pull requests filtered out of 28,154 raw items).

## The folk belief this test targets

Every issue tracker tells maintainers the same thing: **triage the loudest issues
first.** "Loudest" gets operationalised as *most comments*, *most reactions*, or
*most 👍*. Maintainers sort by comment count and work down the list.

## The target

For a closed issue, define:

```
fixed_fast = 1  if  state_reason == "completed"
                and (closed_at - created_at) <= 30 days
             0  otherwise   (not_planned / duplicate / closed slower than 30d)
```

`not_planned` is GitHub's own bucket for "closed without a fix" (rejected,
duplicate, question, stale). So the model answers a question a maintainer
actually cares about: **of the things I could work on this weekend, which ones
will actually land?**

## What I predict, before fitting

**H1 — The loudness heuristic is not measurable at triage time.**
Comment count and reaction count at the moment an issue is *filed* are zero.
The heuristic that governs real triage either (a) requires waiting to see which
issues become loud, in which case it is a popularity contest that has already
happened by the time you act, or (b) is applied to final-state numbers that were
themselves shaped by the outcome. Either way it is not available to the person
deciding what to open next.

**H2 — The strongest legitimate signal is the shape of the report, not its volume.**
I expect body structure to dominate: presence of a fenced code block, presence
of a stack trace or version string, and presence of a URL. An issue that carries
its own reproduction should be far more likely to reach a fix than one that says
"it broke".

**H3 — Body length will be non-monotonic (inverted U), not "longer is better".**
Long bodies mean either a well-specified bug or a frustrated vent. I expect a
mid-range peak: very short issues are under-specified, very long issues are
often duplicate/supersession arguments, and the sweet spot is a compact report.

**H4 — The best single feature will beat the full TabPFN ensemble, or come close.**
A prior-data-fitted network over ~20 mostly-weak tabular features may add little
over one good heuristic. **I am explicitly willing to report that TabPFN loses to
a no-model baseline**, because a measured negative result is a more useful thing
to hand a maintainer than a flattering number.

## Leakage rules I am holding myself to

1. **Time split, never random split.** Train on older issues, test on newer.
   Random splits leak project-era information and inflate scores.
2. **Strictly-at-creation features only** for the headline model. Comment counts,
   reaction totals and close timestamps are *consequences* of the outcome, so
   they are excluded from the headline feature set. I will separately measure
   what they would have added, and label that number explicitly as leakage.
3. **Only closed issues carry a label.** Open issues have no outcome yet and are
   never used for fitting or scoring.
4. Every number in the write-up comes from a script in `scripts/`, re-runnable
   from the committed raw JSON.

## Falsification

If the time-split TabPFN does not beat (a) always-predict-no, (b) the median
class, or (c) the single best feature, I will say so in the README and in the
submission post, and explain what that implies for a maintainer deciding whether
this tool is worth installing.

---

# Addendum — target reframed, recorded before fitting

**Date:** 2026-10-02, after EDA, before any model was fitted.

The original target above was `fixed_fast = completed AND closed within 30 days`.
EDA over the 6,638 closed issues gives:

| quantity | value |
| --- | --- |
| closed | 6,638 |
| `completed` | 5,740 (86.5%) |
| `completed` within 30 days | 4,615 (**69.5% of all closed**) |
| `not_planned` | 660 (9.9%) |
| `duplicate` | 238 (3.6%) |

**A 69.5% base rate makes accuracy useless as a metric.** A model that always
answers "will be fixed" scores 69.5% and looks respectable while being useless.
I am not going to report that number as a result.

The deployable question a maintainer actually has is the inverse one:

> **Which issues in my backlog will never get fixed, so I can stop pretending
> they will?**

So the primary target becomes:

```
dead = 1  if  state_reason in {"not_planned", "duplicate"}
       0  if  state_reason == "completed"
```

898 positive / 5,740 negative = **13.5% base rate**. Because the classes are
imbalanced I will report **ROC-AUC and precision@k**, not accuracy, and
"always predict alive" (86.5% accuracy, 0.0 recall) is the trivial baseline the
model must beat to be worth anything.

This reframing does not change H1-H4; it sharpens them. H4 is now harsher and
more meaningful: beating a 13.5%-base-rate task with ~20 weak tabular features is
a genuinely hard bar.

## Second leakage hazard found in EDA: labels applied at resolution time

Ruff's maintainers add labels when they close a bug, not only at triage time.
Two groups exist and must not be mixed:

- **Triage-time labels** (applied when the issue arrives):
  `bug`, `question`, `needs-info`, `help wanted`, `good first issue`,
  `documentation`, `cli`, `server`, `plugin`, `fuzzer`, `internal`,
  `configuration`, `type-inference`, `docstring`, `isort`, `suppression`, `style`
- **Resolution-time labels** (applied by a maintainer while closing):
  `accepted`, `fixes`, `rule`, `preview`, `ty`

Resolution-time labels are a *consequence* of the outcome, so the headline model
uses **triage-time labels only**. I will separately measure what the
resolution-time labels would have added and report that number explicitly
labelled as leakage. My expectation is that they add a lot, and that showing the
size of that gap is the most useful thing in this write-up: it is a concrete
demonstration of how easy it is to build an issue-triage model that looks great
and is worthless in production.