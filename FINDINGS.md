# Findings — what the numbers actually say

**Corpus:** `astral-sh/ruff`, 8,354 issues retrieved 2026-10-02 (19,800 PRs
filtered out of 28,154 raw items). 6,638 closed issues used for fitting and
scoring, spanning 2022-08-19 to 2026-10-01.
**Engine:** TabPFN v2 — Prior Labs' open-weights tabular foundation model, run
offline on CPU.
**Evaluation:** train on issues filed up to 2025-03-21, score issues filed
2025-03-22 to 2026-10-01. Always a time split, never a random one.

Target: `dead = closed as not_planned or duplicate` — closed without shipping a fix.
Base rate **13.5%** overall, **19.5%** in the held-out future period.

Every figure below is in `data/results.json`, produced by
`python -m triagelens.experiments`. Expectations were registered in
`HYPOTHESIS.md` before any model was fitted. **Three of my four hypotheses were
wrong**, and the sections where that happened are marked.

---

## Finding 1 — "triage the loudest issues first" is worse than useless

This was the folk belief I set out to test, so it goes first.

| | AUC | precision@100 | precision@top 10% |
| --- | --- | --- | --- |
| Loudest-first (comments + reactions) | **0.5208** | **0.190** | **0.167** |
| Constant "everything is alive" | 0.5000 | 0.240 | 0.217 |

Two things make this worse than "uninformative":

1. **0.5208 is a coin flip.** Across four independent rolling time windows it
   scored 0.5410 / 0.5239 / 0.5034 / 0.5006 — it never once rose above 0.55.
2. **Sorting by loudness puts *fewer* duds in your top 100 than picking at
   random** — 0.190 against 0.240, and 0.167 against 0.217 in the top decile. It
   is not merely uninformative, it is mildly *anti*-informative exactly where a
   maintainer would act on it.

The reason is blunt. **Median comments is 2 on issues that never shipped and 2 on
issues that did.** Mean is 2.94 against 3.35. Popularity is not a signal.

## Finding 2 — an offline tabular foundation model does learn something real

| model | AUC | P@100 | P@10% | recall |
| --- | --- | --- | --- | --- |
| Constant "alive" | 0.5000 | 0.240 | 0.217 | 0.000 |
| Single best feature (`n_labels`) | 0.7398 | **0.610** | 0.667 | 0.778 |
| Logistic, all 46 triage-time features | 0.7869 | 0.560 | 0.650 | 0.778 |
| **TabPFN v2, triage-time only** | **0.8202** | **0.650** | **0.700** | 0.838 |
| TabPFN + resolution labels & comments (**leak**) | 0.8095 | 0.580 | 0.733 | 0.855 |

**My H4 was wrong.** I expected the prior-data-fitted network to lose to a trivial
baseline and pre-registered that I would publish it if so. It did not — TabPFN is
the best model here on both metrics, beating 46 logistic features by 0.033 AUC.

The number I would actually stake a maintainer's weekend on is
**precision@100 = 0.650 against a 0.195 base rate**: take the 100 issues
TriageLens flags as most likely dead and **65 of them really are**, where
guessing gives 19. A 3.3x lift.

**But do not trust 0.8202 as a stable figure.** Across rolling folds TabPFN scored
**0.6954 / 0.8690 / 0.7794** — a 0.17 spread on 150 rows each. That is one draw
from a noisy distribution. Honest reporting means the point estimate is 0.82 with
an uncertainty you would not want to quote in a README without the spread next to
it.

## Finding 3 — at the precision that matters, one feature beats 46

The most useful thing in the table above is the row nobody expects:

> `n_labels` alone: AUC 0.7398, **P@100 = 0.610**.
> Full 46-feature logistic: AUC 0.7869, **P@100 = 0.560**.

The single-feature rule has the *worse* overall ranking and the *better* answer to
the only question that matters operationally — *what is in my top 100?* A
maintainer does not sweep a whole backlog; they take a shortlist. On a shortlist,
**"count the triage labels and take the ones with the fewest"** beats a
44-feature model.

I have kept both rows in the README rather than quoting only the flattering one,
because "install this 46-feature model" is the wrong recommendation when the
one-line rule wins on the metric you care about.

## Finding 4 — the signal is thinness, not attention

Permutation importance on the logistic model (10 repeats, held-out AUC drop):

| feature | AUC drop | in-corpus AUC | reading |
| --- | --- | --- | --- |
| `lbl_bug` | +0.118 | 0.369 | bug-labelled issues get **fixed** |
| `log_author_prior_issues` | +0.090 | 0.393 | **returning reporters are worth listening to** |
| `log_body_len` | +0.035 | 0.506 | weak, non-monotonic |
| `log_n_labels` | +0.032 | 0.361 | more labels ⇒ more likely alive |
| `lbl_question` | +0.030 | 0.511 | questions **do** die |
| `log_issue_number` | +0.017 | 0.631 | era proxy, decays out-of-time |
| `has_code_block` | +0.015 | — | small |
| `title_starts_question` | +0.010 | 0.506 | |

The two strongest features are both **inversely** related to the label. What
separates an issue that never ships is not how loudly it is demanding attention —
it is whether it looks **thin**: a reporter new to the project, few labels,
question-shaped phrasing.

`lbl_question` is the only top feature pointing the other way, and it is the case
a maintainer already suspects: a question is frequently closed as a support
request rather than fixed.

## Finding 5 — every handcrafted text feature I wrote made the model worse

| ablation | AUC | delta |
| --- | --- | --- |
| Full 46-feature set | 0.7869 | — |
| Remove all 17 text/structure features | **0.7987** | **+0.0118** |
| Remove all author-history features | 0.7800 | −0.0070 |
| Remove `issue_number` / `log_issue_number` | 0.7567 | −0.0302 |

**Deleting all seventeen handcrafted body probes made the model measurably
better.** H2 predicted body structure would dominate. It is worse than useless.

The author ablation is a clean lesson in reading importance: `log_author_prior_issues`
is the second most important feature, yet removing every author feature costs
0.007 AUC — the information is redundant with `lbl_bug` and `log_n_labels`.
Permutation importance ranks features; it does not establish that a feature is
carrying anything.

## Finding 6 — the era proxy is the strongest feature and the worst one to ship

`issue_number` is the highest-scoring single feature in the whole corpus
(**0.6305** over all 6,638 issues). Scored on two different splits, it behaves
completely differently:

| | AUC |
| --- | --- |
| `issue_number`, random split | **0.6452** |
| `issue_number`, over the whole corpus | 0.6305 |
| `issue_number`, time split | **0.5364** |

The top two rows are the same quantity estimated two ways and agree within
sampling noise. The bottom row is the same feature on the same rows, and it
collapses to barely better than a coin flip. That is the leakage argument in one
line: change only the split and a solid ~0.64 becomes ~0.54. What the
whole-corpus figure was really measuring is project age, not issue content.

Removing the feature costs 0.030 AUC even on the time split, because the dead rate
genuinely moved: **10.1%** of the training period closed dead versus **19.5%** of
the test period.

This one changed the product. Scoring a real backlog with the era features
included produced this:

```
96.4%  #14638   ...
96.3%  #4368    ...
96.2%  #29073   ...
95.9%  #283     Meta issue: plugin system
95.7%  #167     Enforce import from `__all__`
```

A 2022 issue outranking a 2026 one at 96% confidence, because a linear model had
learned "when was this filed" and was extrapolating far outside its training range.
**A maintainer cannot use "when it was filed" to rank today's backlog, where every
issue has roughly the same number.** The deployed scorer therefore drops
`issue_number` and `log_issue_number` by default, and warns that **31% of this
backlog predates the training window** and is being extrapolated.

## Finding 7 — the leaky features made the model worse

I pre-registered an expectation that adding resolution-time labels (`accepted`,
`fixes`, `rule`) plus comment counts would inflate the score substantially.
**It cost 0.011 AUC** (0.8095 vs 0.8202) and dropped precision@100 from 0.650 to
0.580.

Under a time split the leak is not merely useless, it is actively harmful, because
comment volume drifts enough between periods that the model latches onto a
distribution that no longer holds. Every tutorial that says "just add engagement
features" would have made this tool worse while reporting a nicer table.

## Finding 8 — the standard fix for class imbalance also made it worse

With a 13.5% positive rate the reflex is `class_weight="balanced"`. Measured on
this corpus it made **everything** worse:

| | AUC | P@100 | P@10% | mean predicted score |
| --- | --- | --- | --- | --- |
| `class_weight="balanced"` | 0.7819 | 0.550 | 0.583 | 0.508 |
| unweighted | **0.7869** | **0.560** | **0.650** | **0.148** |
| actual base rate | — | — | — | 0.176 |

Unweighted wins on every metric *and* calibrates: mean predicted 0.148 against a
0.176 base rate, with the balanced model predicting 0.508 for a 0.176 event rate.
Ranking metrics are threshold-free, so re-weighting bought nothing and destroyed
the probability a maintainer would actually act on. The shipped code is
unweighted, and the comment in `experiments.py` says why.

## Finding 9 — random splits did *not* inflate anything here

I expected a random split to flatter the model via the era proxy. It did not:

```
random-split AUC 0.7870   time-split AUC 0.7869   inflation +0.00004
```

Essentially zero. My hypothesis about split sensitivity was wrong in magnitude
for this dataset, and I am recording it because "always use a time split" is advice
I gave myself and then found unsupported by this particular corpus. The time split
is still the right default — it is the only split that matches deployment, where a
model scores issues filed *after* it was trained — but on this data it did not
change the headline.

## Finding 10 — the cheapest rule in the whole project

Dead rate by issue body length:

| body length | n | dead rate |
| --- | --- | --- |
| 0 (empty) | 155 | **3.2%** |
| 1–200 | 608 | 11.4% |
| 201–500 | 1,552 | **14.9%** (peak) |
| 501–1000 | 2,092 | 14.4% |
| 1001–2000 | 1,541 | 13.2% |
| 2001–4000 | 690 | 12.8% |

H3 predicted an inverted U. What is there is a **rise then plateau** — the only
transition that matters is 0 → 200 characters. An issue with an empty body is
**4.6x less likely to ever ship** (3.2% vs 14.9%).

A deterministic rule costing no compute and no model, and for the most common
triage action — *ask for more information, or close it* — it is probably all a
maintainer needs. The shipped tool applies it as a separate, model-free flag.

---

## What I would actually tell a maintainer

1. **Stop sorting by comments.** Measured across four time windows, never better
   than 0.55 AUC, and worse than random in the top 100.
2. **Start with a one-line rule, not a model.** `n_labels` alone gets
   precision@100 = 0.610 and beats a 46-feature logistic at 0.560.
3. **Use TabPFN only for a shortlist.** It is the best model here (0.8202 AUC,
   0.650 precision@100) but costs ~0.9s per issue on a laptop CPU — about 25
   minutes for a 1,700-issue backlog. Fine overnight, useless interactively.
4. **Handle empty-body issues first**, with a template reply.
5. **Do not add engagement features** (measured: they cost 0.011 AUC) and **do not
   use `class_weight="balanced"`** (measured: it costs accuracy and calibration).
6. **Distrust any issue-triage score for old issues.** 31% of a real backlog
   predates the training window and is extrapolation.

## Known limitations, stated plainly

- **One repository.** `astral-sh/ruff` is a well-maintained, heavily-labelled
  project. A repo with sparse triage labels would shift every number.
- **Labels are the target's weakest link.** `dead` is inferred from GitHub's
  `state_reason`, which is maintainer-assigned and inconsistently applied.
- **Text is truncated at 4,000 characters** in the committed corpus.
- **TabPFN v2, not 3.5.** The 3.5 checkpoint is licence-gated behind an
  interactive browser login; v2 has an ungated public mirror and runs unattended.
  v3.5 would likely score higher.
- **AUROC on 600 rows has a standard error around 0.02**, and the rolling-fold
  spread for TabPFN is 0.17. Treat 0.8202 as "about 0.8", not "0.82".
- **`dead` is not "bad".** A question closed as a support answer is a good outcome.
  This model finds issues that will not ship a *code change*, which is a different
  question from whether the maintainer handled them well.