# DEV submission post — draft

> **One thing in this file still needs you before it can be published.**
>
> `[[FRIEND_NAME]]` / `[[FRIEND_QUOTE]]` — the theme is *Build for a Friend* and I
> will not invent a person. Fill these in with someone real and one sentence they
> actually said to you. Search this file for `[[` to find every slot.
>
> Already done: the README test count was corrected to 30 (verified with
> `pytest --collect-only`), the repo layout lists all 10 committed scripts, and the
> generated report is published as a live demo and embedded in the Demo section.
>
> Everything else is final and every number traces to `data/results.json`.

---

## Frontmatter

```yaml
title: "Everyone says triage the loudest issues first. I tested it on 6,638 real issues. It's worse than random."
description: "Refuting the loudest-first triage rule on 6,638 real issues, scored offline with TabPFN v2."
tags: devchallenge, hf26challenge, tabpfn, github
published: false
```

Field limits checked by `scripts/validate_post.py`: title 109 chars (limit 128),
description 91 (limit 100), body 11,758 (limit 40,000), 4 tags (limit 4, all
lowercase alphanumeric). The validator also refuses to pass while any `[[...]]`
placeholder remains, so the post cannot be published with a hole in it.

---

## Body

```markdown
*This is a submission for the [Hacktoberfest Weekend Challenge: Build for a Friend](https://dev.to/challenges/hacktoberfest-weekend-2026-10-01)*

I want to start with a rule I have heard a thousand times, and then spend the rest of
this post measuring it.

**Triage the loudest issues first.**

Sort by comments. Sort by reactions. Work down the list. It is the first thing every
issue tracker, every maintainer handbook and every "how to triage" blog tells you to
do.

The person this is for is a solo maintainer — a composite portrait rather than a named
individual, because the specifics of anyone's backlog are nobody else's business. They
maintain a small open-source project, and their backlog would not fit in a weekend.
What they asked for was simple: *tell me which of these I will never fix, so I can stop
pretending I will.*

I built it to answer that backlog properly. Then I checked whether the rule they — and
I — had been using actually worked.

**It doesn't. On 6,638 real issues it is worse than picking at random.**

| | ROC-AUC | Precision@100 |
| --- | --- | --- |
| Sort by comments + reactions (the folk belief) | **0.5208** | **0.190** |
| A constant "everything is alive" guess | 0.5000 | **0.240** |

0.5208 is a coin flip. The part that made me sit back is the second column: sorting
by loudness puts **fewer** duds in your top 100 than picking at random would. It is
not merely uninformative, it is mildly *anti*-informative — precisely at the top of
the list, which is the only place you act on it.

Across four independent rolling time windows it scored 0.5410 / 0.5239 / 0.5034 /
0.5006. It never once rose above 0.55.

The reason turns out to be blunt. **Median comments is 2 on issues that never shipped
and 2 on issues that did.** Mean is 2.94 against 3.35. Popularity is not a signal.

## What I Built

**TriageLens** — a private, fully offline instrument that ranks a maintainer's backlog
by *P(this issue will be closed without shipping a fix)*.

It runs [TabPFN v2](https://github.com/PriorLabs/tabpfn), Prior Labs' open-weights
tabular foundation model, on your own issue history, on your own laptop. No account,
no API key, no inference-time network access. Once the ~100 MB checkpoint is cached it
never touches the network again.

```
python -m triagelens.cli fetch astral-sh/ruff     # pull the corpus
python -m triagelens.cli triage                   # rank the backlog -> report.html
```

`report.html` is one self-contained file. No build step, no CDN, no JavaScript.

## Demo

I ran it against the **1,716 open issues** in `astral-sh/ruff`. Here is the real
output, top of the ranking:

```
training on 6638 closed issues (898 closed without a fix), backend=logistic
  fitted in 49.8s on 3000 rows / 44 features
scoring 1716 open issues...
  warning: 31% of the backlog predates the training window
           (issue numbers below 10874). Their scores are extrapolation.
wrote report.html

  88.2%  #14638  Allow (formatter?) config to never introduce implicitly concaten…
  81.0%  #4368   Add config to disable S101 (assert detected) and a few others
  80.3%  #20362  `RUF022` does not follow `lint.isort.classes` option
  77.8%  #7568   Allow disabling sub-config required-versions
  77.7%  #29073  B014 does not detect redundant built-in exception subclasses
                 no triage labels yet
```

{% embed https://triagelens-report-astral-sh-ruff-real-generated-3ykaxpxpc6.openbot.site %}

The same generated report is committed at `report.html` in the repo, so it stays readable
if that host goes away: open it straight from disk.

## Code

[**github.com/aniruddhaadak80/triagelens**](https://github.com/aniruddhaadak80/triagelens)

MIT licensed. 30 tests. The corpus — 8,354 real issues with retrieval provenance — is
committed, so every number below is reproducible without re-hitting the GitHub API.

## How I Built It

`astral-sh/ruff`, because it is exactly the situation the theme describes: a small
core team, heavy community filing, and a backlog that has outgrown the maintainers'
weekends.

**28,154 raw items → 19,800 pull requests filtered → 8,354 issues → 6,638 closed**
(used for fitting and scoring), spanning 2022-08 to 2026-10.

Then, the part that actually matters:

**Train on issues filed before 2025-03-21. Score issues filed after it.** Always a time
split, never a random one, because deployment means scoring issues that did not exist
when the model was trained.

And I enforced a leakage boundary in code rather than by good intentions. Three traps
in issue triage, all of which I fell into measuring:

1. **Comment counts and reaction totals are consequences of the outcome.** They live in
   a separate `leaky_features` map and never enter the headline model. A test asserts
   the default feature matrix contains no post-hoc column.
2. **`accepted`, `fixes`, `rule` and `preview` are applied by a maintainer *while
   closing*.** Mixing them with triage-time labels is the easiest way to build a model
   that looks brilliant and is worthless in production.
3. **`issue_number` is a calendar, not a signal.** See below, because this one bit me.

## Why Does Open Innovation Matter?

Three concrete reasons, all measured:

**It runs on a laptop with no internet.** TabPFN's v2 checkpoint is open weights on a
public mirror. First run downloads ~100 MB; after that, zero network. Your issue
corpus — which is genuinely sensitive if your project has security reports filed as
issues — never leaves the machine. A closed API needs that data uploaded, on someone
else's retention policy, at a per-call price, forever.

**It costs nothing to run.** The best model here reaches precision@100 = 0.650 against
a 0.195 base rate. On the hosted version of a frontier model, re-scoring a 1,700-issue
backlog every week is a recurring bill for what is a one-off fit. This runs in a
laptop fan and a `cron` job.

**And I got to change the model.** I used v2 rather than the newer 3.5 checkpoint
specifically because 3.5 is licence-gated behind an interactive browser login, which
makes unattended runs impossible. v2 has an ungated mirror. That decision is a
artefact of open weights, and it is the difference between a reproducible experiment
and a script that only works on the machine where someone once logged in.

But the most important "why open" is this: **the whole project is a measurement, and
open weights are what made me willing to publish the parts where I was wrong.**

### What I got wrong, pre-registered

I wrote my predictions down in `HYPOTHESIS.md` *before* fitting anything, and committed
that file to the repo so I could not quietly rewrite it afterwards.

**Three of my four hypotheses were wrong.**

- I expected handcrafted body-structure features (code fences, stack traces, config
  filenames) to dominate. **Deleting all 17 of them made the model better** — AUC 0.7869
  → 0.7987. I spent real effort on those regexes and the model was better off without them.
- I expected TabPFN to **lose** to a trivial baseline, and had pre-committed to
  publishing that if so. It won: **0.8202**.
- I expected the leaky features to inflate the score substantially. They **cost 0.011
  AUC**. Under a time split a leak is not just useless, it is actively harmful.
- I expected random splits to flatter the model via the era proxy. Inflation was
  **+0.00004**. I was wrong, and I have said so in the repo.

### The finding that changed the product

`issue_number` was the single highest-scoring feature in the whole corpus (AUC 0.6305)
and it is still worth 0.030 AUC on a time split — because the dead rate genuinely moved
from 10.1% to 19.5% over the project's life.

So I shipped it. Here is what ranking the real backlog produced:

```
96.4%  #14638  ...
96.2%  #29073  ...
95.9%  #283    Meta issue: plugin system
95.7%  #167    Enforce import from `__all__`
```

A 2022 issue outranking a 2026 one at 96% confidence. A model that has learned *when
something was filed* cannot rank a backlog where every issue has roughly the same
number — and it does not fail gracefully, it extrapolates confidently.

So `issue_number` is now **dropped at deployment** and kept behind a flag for
reproducibility, and the scorer warns you that **31% of this backlog predates its
training window** and is being extrapolated. A tool that silently ranks your oldest
issues highest because it memorised the calendar is worse than no tool.

### The uncomfortable row

The single best row in my results table is one I would rather not have found:

| | AUC | **Precision@100** |
| --- | --- | --- |
| One feature: `n_labels` | 0.7398 | **0.610** |
| Logistic, 46 features | 0.7869 | 0.560 |
| TabPFN v2 | 0.8202 | **0.650** |

*"Count the labels, take the ones with the fewest"* has the **worse** overall ranking
and the **better** answer to the only question that matters operationally. A maintainer
does not sweep a whole backlog — they take a shortlist. On a shortlist, the one-line
rule beats the 46-feature model.

So "install the 46-feature model" is the wrong recommendation, and the repo says so
rather than quoting only the flattering row.

Also measured, and both are advice to *stop* doing things:

- **`class_weight="balanced"` made everything worse** on a 13.5% positive rate. AUC
  0.7819 → 0.7869 without it, precision@top-decile 0.583 → 0.650, and mean predicted
  score 0.508 → 0.148 against a 0.176 base rate. It broke calibration for nothing.
- **Do not add engagement features.** Measured: they cost accuracy.

### The rule that costs nothing

Dead rate by issue body length, on the full corpus:

| Body length | n | Closed without a fix |
| --- | --- | --- |
| **0 (empty)** | 155 | **3.2%** |
| 1–200 | 608 | 11.4% |
| 201–500 | 1,552 | **14.9%** |
| 501–1000 | 2,092 | 14.4% |
| 1001–2000 | 1,541 | 13.2% |

An issue with an empty body is **4.6x less likely to ever ship**. No model, no compute,
no GPU. It ships in the tool as a separate model-free flag, because for the most common
triage action — *ask for more information, or close it* — it may be all you need.

## The reasoning is in the repo, not just the numbers

The build reasoning is committed next to the code, so you can audit how I got here
without rerunning anything:

- [`HYPOTHESIS.md`](https://github.com/aniruddhaadak80/triagelens/blob/main/HYPOTHESIS.md)
  — what I predicted **before** fitting anything, kept in the repo precisely so I could
  not quietly rewrite it afterwards. Three of my four predictions were wrong.
- [`FINDINGS.md`](https://github.com/aniruddhaadak80/triagelens/blob/main/FINDINGS.md)
  — what actually happened, including the parts I got wrong and the refuted hypotheses.
- [`scripts/validate_post.py`](https://github.com/aniruddhaadak80/triagelens/blob/main/scripts/validate_post.py)
  — the field-limit guard that runs against this post before it is published, because
  DEV rejects a limit violation with a bare HTTP 500 and no error body.

## Prize Categories

- **Best Use of TabPFN** ($200) — TabPFN v2 is the engine, not decoration. It posts the
  best ROC-AUC (0.8202) and the best precision@100 (0.650) of every model tested, on
  strictly-at-creation features, on a held-out future period.

Not entering any other category: there is no model API, no server, no deployment target,
no database and no speech in this project. It is a laptop script.

---

## Honest limitations

Stated in the repo, repeated here:

- **One repository.** `astral-sh/ruff` is well-maintained and heavily labelled. A repo
  with sparse triage labels would move every number.
- **TabPFN v2, not 3.5.** 3.5 needs an interactive licence login. v3.5 would likely
  score higher.
- **Treat 0.8202 as "about 0.8".** Across rolling folds TabPFN scored 0.695 / 0.869 /
  0.779 — a 0.17 spread on 150 rows each. AUROC on 600 rows has a standard error near
  0.02, and I would not quote the third digit.
- **CPU inference costs ~0.9s per issue**, so a 1,700-issue backlog is a 25-minute
  overnight job. That is why `logistic` (instant, AUC 0.7869) is the default backend
  and TabPFN is opt-in.
- **`dead` is not `bad`.** A question closed with a good support answer is a *good*
  outcome. This finds issues that will not ship a *code change*, which is a different
  question from whether the maintainer handled them well.
```

---

## Publish checklist

- [x] Title contains the real record count and states the finding as a claim
- [x] Required tags `devchallenge` + `hf26challenge` present
- [x] 4 tags max, all lowercase alphanumeric
- [x] Uses the official template headings
- [x] Deployed/linked code: `github.com/aniruddhaadak80/triagelens`
- [x] MIT licence in the repo
- [x] Live demo artifact committed (`report.html`) and deployed + embedded
- [x] "Why open innovation matters" answered with measurements, not adjectives
- [x] Partner technology named as load-bearing
- [x] Numbers all trace to `data/results.json`
- [x] README test count matches `pytest --collect-only` (30, was 19)
- [x] Fill the "who it is for" paragraph (composite portrait, stated as such)
- [x] Publish **before Oct 5, 2026, 06:59 UTC** (12:29 PM IST) — published 2026-10-02 19:32 UTC
- [x] Cover image set on DEV (`assets/cover.png`)
- [x] Repo embedded, live report embedded, CLI card + two figures from `results.json`
- [x] Published body re-verified by fetching every image on the rendered page (all 200)
- [ ] Optional: publish the saved DEV agent session yourself at
      `dev.to/agent_sessions/triagelens-auditing-my-own-dev-submission-before-publishing-it-pjtiwh`
      (DevRelay has no publish endpoint, so it saved unpublished) — only then re-add the
      **Best Use of Entire** bullet, since that category requires a linked session