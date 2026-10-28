# Eval Protocol: Does ReAct Beat Act-Only Once the Model Can Plan?

| | |
|---|---|
| **Status** | Designed, not yet run. Pre-registration draft, **version 1.3**: calibrated, levels chosen, ready to freeze |
| **Author** | Joshua Obafemi |
| **Dates** | v1.0: 15 October 2026 · v1.1 to v1.3: 16 October 2026 |
| **Origin** | Week 3 left this open. At 0.5B, ReAct scored 20% and Act-only 23% on 30 problems, which the Week 4 audit showed is uninformative (95% CI −16.7 to +10.0 points, 10% power) |
| **Pilot** | Week 3 rerun at 1.5B and 3B on a Colab T4 (results in `results/week3_gpu/`). Results summarised in section 1.1; changes listed in section 11 |
| **Supporting analysis** | [`jupyter/week4_rigour_audit.ipynb`](../jupyter/week4_rigour_audit.ipynb) (power analysis) and `results/week3_gpu/` (pilot data) |

---

## 1. The claim

The ReAct paper (Yao et al., 2022) reports that adding reasoning traces to tool-using prompts improves task success, and that prompted ReAct works mainly with large models.

**Claim under test (C):**
> For instruction-tuned Qwen2.5 models at 1.5B and 3B parameters, ReAct prompting solves at least **10 percentage points more** held-out multi-step calculator word problems than Act-only prompting, at no more than **1.5× the token cost**.

Why these numbers:
- **10 points** is the smallest gain worth the extra tokens. In the pilot, ReAct cost 1.26× (0.5B) to 1.89× (3B) the tokens of Act-only.
- **1.5×** caps the cost side, following Kapoor et al.: an accuracy gain only counts if it doesn't come with an unreasonable cost increase.

### 1.1 What the pilot showed (why the claim is still worth testing)

| Model | Act-only | ReAct | ReAct − Act (95% CI) | McNemar p | Discordance | Tokens ReAct / Act |
|---|---|---|---|---|---|---|
| 0.5B | 23% | 20% | −3.3 (−16.7 to +10.0) | 1.000 | 17% | 1.26× |
| 1.5B | 97% | 67% | **−30.0 (−50.0 to −13.3)** | **0.012** | 37% | 1.43× |
| 3B | 100% | 100% | 0 | 1.000 | 0% | 1.89× |

The pilot points *against* the claim at 1.5B and can't test it at 3B. It can't settle the question, for three reasons:
1. **Ceiling:** Act-only scored 97% to 100%, leaving no room to show a 10-point gain.
2. **Template clustering:** the whole 1.5B gap came from 2 of 6 templates (5 of 5 failures each). With 6 templates, the effective sample size is closer to 6 than to 30.
3. **No holdout:** the pilot templates overlap with the prompt exemplars.

So the claim stays the same, the question becomes explicitly two-sided (benefit *or* harm), and the task is redesigned.

## 2. Hypotheses

| | Statement | Test |
|---|---|---|
| **H1 (primary)** | For each model size, accuracy(ReAct) − accuracy(Act) ≥ +10 points | Exact McNemar, two-sided, on paired outcomes. Holm correction across the 2 model sizes |
| **H0** | accuracy(ReAct) = accuracy(Act) | |
| **H2 (cost)** | mean tokens(ReAct) / mean tokens(Act) ≤ 1.5 | Point estimate with 95% bootstrap CI |
| **H3 (secondary, exploratory)** | The ReAct − Act gap changes with model size (0.5B, 1.5B, 3B) | Reported descriptively. Not a confirmatory test |

The test is two-sided, so a significant result in either direction is reported. Harm is a named outcome (section 6.2), not a footnote.

## 3. Design

### 3.1 Conditions
| Factor | Levels |
|---|---|
| Prompting mode (**independent variable**) | Act-only, ReAct |
| Model (blocking factor) | Qwen2.5-1.5B-Instruct, Qwen2.5-3B-Instruct. Qwen2.5-0.5B-Instruct as a replication anchor |
| Reference baselines | Direct answer, CoT (no tool), run once to confirm the task needs a tool |

**Within-problem (paired) design:** every problem is solved under every mode, so problem difficulty cancels out of the comparison.

### 3.2 Task set: harder, and held out

The pilot's two-step problems were too easy once the model could read them. The new generator produces problems that stress planning, which is where reasoning traces are supposed to help:

| Feature | Pilot (Week 3) | v1.1 |
|---|---|---|
| Steps | 2 | **3 to 4** |
| Numbers | All needed | **1 to 2 distractor quantities** that must be ignored (e.g. "the warehouse also has 12 staff") |
| Operations | Fixed order in the text | Order in the text ≠ order of operations (e.g. the subtraction is mentioned first) |
| Units | One unit | **A unit conversion** in some templates (minutes → hours, grams → kilograms) |

Kapoor et al.'s framework applies: the claim is **task-specific**, so the holdout must contain **problem shapes the prompt never showed**.

| Split | Templates | Problems | Purpose |
|---|---|---|---|
| Exemplars | 2 (shop/pens, factory/bolts, as in Week 3) | 2 | Shown in every prompt |
| Dev | 2 new templates | 40 | Prompt debugging, parser fixes and **difficulty calibration** (6.1, gate 5). Can be looked at freely |
| **Test (held out)** | **8 further templates**, none structurally identical to an exemplar | **480** (60 per template) | Run once, after the protocol is frozen, at each model's frozen level (3.2.3) |

Test templates are checked by hand for exemplar overlap, and by script for operation-sequence overlap.

**Implementation:** [`src/react_tasks_v2.py`](../src/react_tasks_v2.py). Dev templates: bakery, parking. Test templates: orchard, printing, water tank, school trip, factory, fundraiser, shipping, cinema. Test seed fixed at 4200 + level. Every problem carries a solution expression that is checked against its answer with the agent's own calculator.

### 3.2.1 Difficulty levels and how one is chosen (v1.2, fixed before calibration v2)

| Level | Steps | Distractors | Fact order |
|---|---|---|---|
| 1 | 3 | 1 | Natural |
| 2 | 3 | 1 | Scene sentence first, other facts shuffled |
| 3 | 3 | 2 | Scene sentence first, other facts shuffled |
| 4 | 4 (an extra operation or a unit conversion) | 2 | Scene sentence first, other facts shuffled |

**Calibration run:** Act-only on all 40 dev problems at every level, for both models. Then direct and ReAct on dev at each model's chosen level only.

**Selection rule:**
1. **Common level first.** Eligible levels have Act-only dev accuracy between 40% and 85% for **both** 1.5B and 3B. If several are eligible, choose the one whose mean Act-only accuracy is closest to 62.5%.
2. **Per-model fallback.** If no level is eligible for both, each model gets its own level: the in-band level closest to 62.5% for that model. Each model's verdict (6.2) stays valid, because ReAct and Act-only are always compared on the same problems. H3 (the gap across sizes) is then confounded by difficulty and reported as such.
3. A model with no in-band level fails gate 5. Change the generator on dev only and calibrate again.
4. The chosen level or levels are written into this document and frozen with it.

With 40 dev problems per level, each accuracy is known to about ±15 points, so the band is deliberately wide.

### 3.2.2 Dev calibration v1 (16 October 2026) and why v1.2 exists

Calibration v1 (levels as in v1.1, every fact shuffled) found no eligible level. Raw data: `results/week4_dev_v1/`.

| v1.1 level | 1.5B Act-only | 3B Act-only |
|---|---|---|
| 1 (natural order) | 75% | 100% |
| 2 (all facts shuffled) | 25% | 60% |
| 3 (4 steps, all shuffled) | 0% | 35% |

The drop from level 1 to 2 was mostly an artefact. At 1.5B, level 2 problems were solved 5 of 7 times when the scene-setting sentence happened to stay first, and 5 of 33 times otherwise. With "Over 13 days the bakery sells…" before any bakery was introduced, the model grabbed a distractor (it multiplied "14 bakers" by 44 trays). That measures reading incoherent text, not planning. v1.2 keeps the scene sentence first, adds an intermediate level, and adds the per-model fallback, because the two models sit about one level apart. Level 1 is unchanged, so its v1 runs are reused.

### 3.2.3 Dev calibration v2 (16 October 2026): result

Raw data: `results/week4_dev_v2/`. Act-only accuracy on the 40 dev problems:

| Level | 1.5B | 3B | Eligible for both? |
|---|---|---|---|
| 1 | 75% | 100% | no (3B at ceiling) |
| 2 | 52.5% | 87.5% | no (3B just above 85%) |
| 3 | 35% | 72.5% | no (1.5B below 40%) |
| 4 | 0% | 50% | no |

No common level, so rule step 2 applies. **Frozen levels: 1.5B uses level 2 (Act-only 52.5%); 3B uses level 3 (72.5%, closer to 62.5% than level 4's 50%).** All gates passed at those levels:

| Model (level) | Direct | Act-only | ReAct | Format failures (Act / ReAct) | Discordance | Tokens ReAct / Act |
|---|---|---|---|---|---|---|
| 1.5B (L2) | 0% | 52.5% | 35.0% | 0% / 7.5% | 48% | 2.16× |
| 3B (L3) | 0% | 72.5% | 90.0% | 0% / 5.0% | 32% | 1.33× |

These dev differences are **exploratory** (40 problems, and the same problems chose the level). They are not evidence for or against C. They suggest the test is now able to show an effect in either direction: ReAct trails at 1.5B (−17.5 points, CI −37.5 to +2.5) and leads at 3B (+17.5, CI 0.0 to +35.0).

### 3.3 Fixed settings
| Setting | Value | Why |
|---|---|---|
| Decoding | Greedy (temperature 0) | Removes sampling noise from the primary comparison |
| Max steps | 8 (was 6) | 3 to 4 step problems need room. Hitting the cap counts as a failure |
| Exemplars | The same 2 problems in both modes, differing only in trajectory format | Isolates the effect of the Thought lines |
| Tool | `calculate[expr]`, safe AST evaluator | Same as Week 3 |
| Stop sequence | `Observation:` | Same as Week 3 |
| Answer check | Numeric exact match after removing commas and `$` | Automatic, so no grader bias |
| Code version | Git commit hash recorded before the test run | Reproducibility |
| Hardware | Colab or Kaggle T4, via [`src/run_react_bench.py`](../src/run_react_bench.py) | Measured speeds in section 9 |

## 4. Sample size

Power of the exact McNemar test to detect +10 points (4,000 simulations per point):

| N per model | 30% discordance | 37% (pilot, 1.5B) | 40% |
|---|---|---|---|
| 300, α = 0.05 | 87% | 80% | 75% |
| 340, α = 0.05 | 91% | 84% | 81% |
| 400, α = 0.05 | 95% | 90% | 87% |
| 300, Holm α = 0.025 | 81% | 71% | 65% |
| 340, Holm α = 0.025 | 87% | 76% | 73% |
| **400, Holm α = 0.025** | **92%** | **83%** | **80%** |

**Decision (v1.3): N = 480 test problems per model** (8 templates × 60). Dev calibration v2 measured 48% discordance at 1.5B, where N = 400 gives only 72% power after Holm correction. At N = 480, power is 81% at 48% discordance, 88% at 40% and 94% at 32% (the 3B dev value), all for +10 points at Holm α = 0.025.

N is fixed before the run. There is no optional stopping and no adding problems after seeing results.

## 5. Metrics

| Metric | Role |
|---|---|
| Accuracy (exact match) | **Primary** |
| Tokens per problem (input + output) | **Primary cost metric** (H2) |
| LLM calls per problem, seconds per problem | Secondary cost |
| Failure categories: misread problem (wrong expression), invented tool, multiple actions per step, ignored or overrode an observation, invalid action, step limit | Diagnostic. The pilot's 1.5B failures added the first three |
| Format-failure rate (invalid action, invented tool or no `finish`) | Validity check, see 6.1 |
| Per-template accuracy | Diagnostic, and input to the robustness check in 8.4 |

Results are reported as a **cost-accuracy plot with the Pareto frontier marked**, plus a table with 95% Wilson CIs for every accuracy and a paired bootstrap CI for every difference.

## 6. Success and failure criteria (fixed before running)

### 6.1 Validity gates (if any fails, the run is invalid and the claim isn't tested)
1. **The task needs the tool:** the direct-answer baseline scores ≤ 10% on the test set.
2. **The tool is correct:** the calculator passes all unit tests (including the injection and overflow cases).
3. **The protocol is followed:** format-failure rate ≤ 20% in *both* modes. Above that, the result measures prompt-following, not reasoning.
4. **No leakage:** no test template appears in the exemplars or the dev set (checked by script).
5. **No ceiling or floor (new):** on the dev set, Act-only scores between **40% and 85%** at both model sizes. If not, adjust difficulty (steps, distractors) **on dev only**, then re-check, before freezing. Without headroom, a 10-point difference can't appear.

### 6.2 Outcomes for each model size
| Outcome | Rule |
|---|---|
| **Supported** | Holm-adjusted McNemar p < 0.05, **and** the 95% CI of (ReAct − Act) has its lower bound > 0, **and** the point estimate is ≥ +10 points, **and** H2 holds (cost ratio ≤ 1.5) |
| **Supported, but too costly** | Accuracy criteria met, cost ratio > 1.5 |
| **Harmful (new)** | Holm-adjusted p < 0.05 **and** the 95% CI upper bound < 0. ReAct is reliably worse than Act-only |
| **Refuted** | The 95% CI upper bound of (ReAct − Act) is < +5 points, without meeting "harmful". ReAct's benefit is too small to matter, even at its most optimistic |
| **Inconclusive** | Anything else. Report the CI and the power actually achieved; no claim either way |

**Overall verdict on C:** supported if supported at *both* sizes. Partially supported at one size. Refuted if refuted or harmful at both.

## 7. Threats to validity and how they are handled

| Threat | Type | Mitigation |
|---|---|---|
| Ceiling or floor effects (seen in the pilot: Act-only 97% to 100%) | Measurement | Harder task (3.2) and the difficulty gate (6.1, gate 5) |
| Effects driven by a few templates (seen in the pilot: the 1.5B gap came from 2 templates) | Clustering | 8 test templates × 50 problems. Leave-one-template-out robustness check (8.4) |
| Exemplar-shape overlap inflates scores (seen in Week 3) | Confound | Held-out test templates (3.2) |
| ReAct prompt is longer, so differences could come from length alone | Confound | Reported, not removed. Act-only exemplars contain the same problems and actions. The comparison is "with vs without Thought lines", which is the claim |
| Prompt tuned on the test set | Overfitting | All prompt, parser and difficulty changes happen on the dev set. The protocol is frozen and committed before the test run |
| Picking the best of many runs | Selection bias | One pre-registered run per condition. In the Week 4 audit, gaps picked as best-of-24 shrank by 28% to 40% on held-out items |
| Many comparisons | False positives | One primary hypothesis per model; Holm across the two sizes. Everything else is labelled exploratory |
| Parser bugs that favour one mode | Measurement | Same parser for both modes, tested on dev transcripts. 20 test transcripts per mode audited by hand after the run |
| Results specific to Qwen | External validity | Stated as a limitation. A follow-up could repeat with another model family |
| Synthetic problems may not reflect real tasks | External validity | Stated as a limitation. The claim is scoped to this task |
| Greedy decoding hides variance | Reliability | Exploratory robustness check: 3 runs at temperature 0.7, reported separately |

## 8. Analysis plan (written before seeing data)

1. Check the five validity gates. Stop and report if any fails.
2. For each model: 2×2 table of (Act right/wrong) × (ReAct right/wrong). Exact McNemar on the discordant cells. Paired bootstrap CI (10,000 resamples) on the difference.
3. Holm-adjust the two p-values. Apply section 6.2.
4. **Robustness (new):** repeat step 2 leaving out each test template in turn. If removing any single template flips the outcome category, report the result as template-dependent.
5. Cost ratio with a bootstrap CI.
6. Cost-accuracy plot (all modes and models) with the Pareto frontier.
7. Exploratory only: failure-category breakdown, per-template accuracy, the 0.5B anchor, the temperature-0.7 runs.

## 9. Budget

Measured on a free Colab T4 during the pilot (seconds per problem):

| Model | Direct | Act-only | ReAct |
|---|---|---|---|
| 1.5B | 1.5 | 5.9 | 9.3 |
| 3B | 2.0 | 7.7 | 16.1 |

| Item | Estimate |
|---|---|
| Episodes | 480 problems × 3 modes (direct, Act, ReAct) × 2 models = 2,880 |
| GPU time at dev-calibration speeds | 1.5B at level 2: about 24 s per problem for all three modes, so about 3.2 h. 3B at level 3: about 33 s, so about 4.3 h. **About 7.5 GPU hours in total** |
| Tokens | ReAct used 2.16× (1.5B) and 1.33× (3B) the tokens of Act-only on dev |
| Platform | Kaggle's weekly GPU quota (about 30 h) covers it. On Colab's free tier, expect 3 to 5 sessions; the runner resumes after disconnects |
| Fellow time | Generator and templates: 4 h. Dev iteration and difficulty calibration: 4 h. Run and analysis: 3 h. Write-up: 3 h |

## 10. Pre-registration checklist
- [x] Harder generator written (3 to 4 steps, distractors, reordered operations, unit conversions): `src/react_tasks_v2.py`, all automatic checks pass
- [x] Calibration v1 run on Colab (no eligible level; generator flaw found, see 3.2.2)
- [x] Calibration v2 run on Colab; levels chosen by the rule in 3.2.1 (1.5B: level 2, 3B: level 3); all gates passed
- [x] Test templates written and checked for exemplar overlap (script: no overlap at any level)
- [ ] Dev-set iteration finished; difficulty gate passed at both sizes; prompts and parser frozen
- [x] This document committed (this file); the commit hash is printed by the test-run notebook and stored with the results
- [ ] Test set generated from a fixed seed, never inspected before the run
- [ ] Single run per condition; all outputs saved, including failures

## 11. Revision history

| Version | Date | Change | Reason |
|---|---|---|---|
| 1.0 | 15 Oct 2026 | First draft | Designed from the 0.5B Week 3 result |
| 1.1 | 16 Oct 2026 | Harder task (3 to 4 steps, distractors, reordered operations, unit conversions) | Pilot ceiling: Act-only 97% to 100% |
| 1.1 | 16 Oct 2026 | Difficulty gate added (6.1, gate 5) | Guarantees headroom before freezing |
| 1.1 | 16 Oct 2026 | "Harmful" outcome added; test treated as explicitly two-sided | Pilot: ReAct 30 points worse at 1.5B |
| 1.1 | 16 Oct 2026 | N raised from 300 to 400 per model | Pilot discordance 37% → 83% power after Holm |
| 1.1 | 16 Oct 2026 | Max steps 6 → 8 | Longer problems |
| 1.1 | 16 Oct 2026 | Leave-one-template-out robustness check | Pilot gap came from 2 templates |
| 1.1 | 16 Oct 2026 | New failure categories: misread problem, invented tool, multiple actions per step | Observed in pilot transcripts |
| 1.1 | 16 Oct 2026 | Budget re-estimated from measured T4 speeds | Pilot timings |
| 1.2 | 16 Oct 2026 | Scene sentence always first; four levels instead of three; per-model level fallback | Calibration v1: no eligible level, and full shuffling made text incoherent (3.2.2) |
| 1.3 | 16 Oct 2026 | Levels frozen by the 3.2.1 rule: 1.5B level 2, 3B level 3 | Calibration v2 (3.2.3) |
| 1.3 | 16 Oct 2026 | N raised from 400 to 480 per model; budget re-estimated | Calibration v2 measured 48% discordance at 1.5B: N = 400 gave 72% power, N = 480 gives 81% |

All changes were made **before** any test-set problem was run, using pilot, dev-calibration and design information only. No test problem has been run. This is a legitimate pre-registration amendment.
