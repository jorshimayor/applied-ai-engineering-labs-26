# Memo: How to Evaluate a Claim Rigorously

**From:** Joshua Obafemi · **Date:** 2026-10-15 · **Phase 1 deliverable** (Weeks 1 to 4)

I audited 15 claims from my own Week 2 and 3 notes. Eight held up; seven could not be supported by the data I had. None were wrong in direction, but nearly half were stated with more confidence than the evidence allowed. These are the rules I'm adopting as a result, each tied to the mistake that taught it.

## Before running anything

**1. Write the claim so it can fail.** Name the population, the conditions, the metric, the size of effect that matters and the cost you'll accept. "ReAct is better" can't fail. "ReAct solves ≥10 points more held-out problems than Act-only at ≤1.5× the tokens" can.

**2. Decide what would change your mind, in writing.** Success, failure and "inconclusive" each need a rule set before the data exists. Otherwise every result can be read as support.

**3. Size the experiment.** Run a power analysis from a pilot. My Week 3 comparison (30 problems) had **10% power** to detect a 10-point difference. It could not have found the effect it was looking for, so its "ReAct ≈ Act" result says almost nothing.

**4. Hold something out.** Any choice made while looking at test results (best prompt, best ordering, a parser fix) leaks into the score. In Week 2, the best-vs-worst ordering gap shrank from 8.0 to 4.8 points when the "best" was chosen on half the data and measured on the other half. Match the holdout to the claim's generality (Kapoor et al.): a task-level claim needs problem shapes the prompt never showed.

**5. List the confounds and kill the big ones.** Week 3's worst confound was invisible until I looked per template: almost every success came from the one problem type shaped like a prompt example (Fisher p = 0.015).

**5b. Pilot the task before you freeze the protocol.** A pilot measures headroom and variance, not the effect. My first protocol assumed the Week 3 problems were a fair test. A quick run at 1.5B and 3B showed Act-only already scoring 97% to 100%, so no gain could ever appear, and the whole 1.5B difference came from two templates. Both problems were fixable on paper, before any test data existed.

## When analysing

**6. Pair your comparisons.** Run both conditions on the same items and compare item by item (McNemar for 0/1, paired bootstrap or sign-flip otherwise). Pairing removes item difficulty, which is most of the noise.

**7. Report intervals, not just points.** 6/30 is "20%" but the 95% interval runs from 9.5% to 37%. Use Wilson intervals for proportions; the textbook formula gives nonsense near 0% and 100%.

**8. Absence of evidence isn't evidence of absence.** "No difference" needs its own test: the whole confidence interval must sit inside a margin you set in advance. My "random labels barely matter" claim passed this (CI −0.3 to +2.7 points). My "ReAct ≈ Act" claim did not (CI −16.7 to +10.0).

**9. Correct for the number of claims.** Testing 11 directional claims at α = 0.05 makes a false positive likely. After Holm correction, "the calculator helps" (raw p = 0.016) was no longer significant on 30 problems, even though a 0-vs-7 split looks convincing.

**10. Report cost next to accuracy.** Plot accuracy against tokens or dollars and keep only Pareto-optimal designs. Kapoor et al. found simple retry baselines matching complex agents at a fraction of the cost. In my Week 3 data, ReAct used 26% more tokens than Act-only for no measurable gain.

## When writing up

**11. Separate confirmatory from exploratory.** Only pre-registered hypotheses get significance claims. Everything else is "observed, worth testing".

**12. Say what you'd need to be wrong.** End with the result that would overturn your conclusion and how big a study would detect it.

## The checklist
- [ ] Falsifiable claim with effect size and cost bound
- [ ] Success, failure and inconclusive rules written first
- [ ] Pilot run: check for ceiling, floor and clustering
- [ ] Power analysis from the pilot; N fixed in advance
- [ ] Holdout matched to the claim's generality; no tuning on test
- [ ] Confounds listed; the main ones designed out
- [ ] Paired design and paired tests
- [ ] 95% CIs on every number
- [ ] Equivalence margin for "no difference" claims
- [ ] Multiple-comparison correction
- [ ] Cost-accuracy plot
- [ ] Confirmatory vs exploratory labelled
