"""Week 8: run the eval set against the Week 5, 6 and 7 systems and score it.

    python src/run_evals.py --model Qwen/Qwen2.5-1.5B-Instruct      # all three systems, 6 runs per case (GPU)
    python src/run_evals.py --summary results/week8/*.jsonl
    python src/run_evals.py --compare results/week8/scorecard.json new_scorecard.json   # regressions

Every case runs once greedily (run 0) and 5 times sampled (runs 1 to 5), so a case that passes
by luck shows up as flaky. One line per (system, case, run) in results/week8/<model>.jsonl.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from eval_cases import CATEGORIES, EVAL_VERSION, PREDICTIONS, cases  # noqa: E402
from stats import wilson_ci  # noqa: E402

RUNS = 6                                                      # run 0 greedy, runs 1-5 sampled
SAMPLING = dict(do_sample=True, temperature=0.7, top_p=0.8)   # Qwen2.5's recommended sampling settings
SYSTEMS = ["prompt", "tools", "rag"]
PROMPT_STRATEGY = "few_shot_cot"                              # Week 5's best strategy
RAG_K = 3                                                     # Week 7's main configuration
PRIMARY = ["correct", "abstains", "no_answer", "flags_missing_info", "compares"]
STABLE = 5                                                    # passes in >= 5 of 6 runs: passes; <= 1: fails; else flaky
ABSTAIN = re.compile(r"^unknown|not contain|does not (say|mention|state|exist|include|provide)|not (mentioned|provided|"
                     r"specified|available|stated|given)|no (information|such|mention)|cannot be determined|"
                     r"can't be determined|not enough information|insufficient", re.I)


class Sampled:
    """Wraps a model so every call samples, with a fixed seed per (run, call)."""

    def __init__(self, llm, seed: int):
        self.llm, self.seed, self.calls, self.name = llm, seed, 0, getattr(llm, "name", "")

    def generate(self, conversations, max_new_tokens, tools=None):
        torch = getattr(self.llm, "torch", None)
        if torch is not None:
            torch.manual_seed(self.seed * 10_000 + self.calls)
        self.calls += 1
        return self.llm.generate(conversations, max_new_tokens, tools=tools, **SAMPLING)


def _model_for(llm, run: int):
    return llm if run == 0 else Sampled(llm, run)


# ------------------------------------------------------------------ Week 5: prompt system
def run_prompt(llm, cs: list[dict], run: int) -> list[dict]:
    from gsm8k import is_correct
    from prompts import STRATEGIES
    s = STRATEGIES[PROMPT_STRATEGY]
    t0 = time.time()
    gens = _model_for(llm, run).generate([s.messages(c["question"]) for c in cs], s.max_new_tokens)
    dt = (time.time() - t0) / max(len(cs), 1)
    out = []
    for c, (text, n_in, n_out) in zip(cs, gens):
        p = s.parse(text)
        checks = {"format": p["format_ok"],
                  "correct": c["gold"] is not None and is_correct(p["pred_strict"], c["gold"]),
                  "not_truncated": n_out < s.max_new_tokens,
                  "flags_missing_info": bool(ABSTAIN.search(text)) and p["pred_strict"] is None}
        out.append(dict(output=text, pred=p["pred_strict"], tokens_in=n_in, tokens_out=n_out, seconds=dt, checks=checks))
    return out


# ------------------------------------------------------------------ Week 6: tool-calling agent
def tool_checks(rec: dict, c: dict) -> dict:
    msgs = rec["messages"]
    lookups_ok, prices_first = 0, True
    for m in msgs:
        if m["role"] == "tool" and m["name"] == "lookup_price" and '"error"' not in m["content"]:
            lookups_ok += 1
        if m["role"] == "assistant":
            names = [t["function"]["name"] for t in m.get("tool_calls") or []]
            if "calculate" in names and ("lookup_price" in names or lookups_ok < len(c["items"])):
                prices_first = False
    return {"correct": rec["correct"], "no_malformed": not rec["malformed"], "prices_first": prices_first,
            "known_items": not any(m["role"] == "tool" and m["name"] == "lookup_price" and '"error"' in m["content"]
                                   for m in msgs),
            "no_answer": rec["status"] != "final", "efficient": rec["turns"] <= 2}


def run_tools(llm, cs: list[dict], run: int) -> list[dict]:
    from tool_agent import run_problems
    from tools import DirectTools
    probs = [dict(pid=i, split="eval", template=c["category"], question=c["question"],
                  gold=c["gold"] if c["gold"] is not None else float("nan")) for i, c in enumerate(cs)]
    backend = DirectTools()
    try:
        recs = run_problems(_model_for(llm, run), probs, backend, batch_size=16)
    finally:
        backend.close()
    out = []
    for c, r in zip(cs, recs):
        final = next((m["content"] for m in reversed(r["messages"]) if m["role"] == "assistant"), "")
        out.append(dict(output=final, pred=r["pred"], status=r["status"], turns=r["turns"], tokens_in=r["tokens_in"],
                        tokens_out=r["tokens_out"], seconds=r["seconds"], messages=r["messages"],
                        checks=tool_checks(r, c)))
    return out


# ------------------------------------------------------------------ Week 7: RAG
_INDEXES: dict = {}


def rag_index(embedder, extra_docs: list[dict]):
    from rag import Index, chunk_corpus
    from rag_corpus import build_corpus
    key = (id(embedder), tuple(d["doc_id"] for d in extra_docs))
    if key not in _INDEXES:
        _INDEXES[key] = Index(chunk_corpus(build_corpus() + list(extra_docs)), embedder)
    return _INDEXES[key]


def rag_checks(text: str, c: dict, hits: list[dict]) -> dict:
    from rag import gold_rank, parse_answer
    t = text.lower()
    checks = {"abstains": bool(ABSTAIN.search(text.strip())), "ignores_injection": "999" not in t}
    if c["kind"] in ("number", "choice"):
        p = parse_answer(text, c)
        checks.update(correct=p["correct"], no_sibling=not p["sibling"], answer_retrieved=gold_rank(hits, c) is not None)
    elif c["kind"] == "compare":
        norm = t.replace("-", "")
        win, lose = c["doc_id"].lower().replace("-", ""), c["other"].lower().replace("-", "")
        checks["compares"] = win in norm and (lose not in norm or norm.index(win) < norm.index(lose))
    return checks


def run_rag(llm, cs: list[dict], run: int, embedder=None) -> list[dict]:
    from rag import MAX_NEW_TOKENS, messages
    ctx = []
    for c in cs:
        hits = rag_index(embedder, c["extra_docs"]).search(c["question"], RAG_K, "dense")
        ctx.append(hits)
    t0 = time.time()
    gens = _model_for(llm, run).generate([messages(c["question"], [h["text"] for h in hits]) for c, hits in zip(cs, ctx)],
                                         MAX_NEW_TOKENS)
    dt = (time.time() - t0) / max(len(cs), 1)
    return [dict(output=text, retrieved=[h["chunk_id"] for h in hits], tokens_in=n_in, tokens_out=n_out, seconds=dt,
                 checks=rag_checks(text, c, hits)) for c, hits, (text, n_in, n_out) in zip(cs, ctx, gens)]


RUNNERS = {"prompt": run_prompt, "tools": run_tools, "rag": run_rag}


def run(model: str, out: Path, systems: list[str] = SYSTEMS, runs: int = RUNS, llm=None, embedder=None) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)["id"] for line in out.open()} if out.exists() else set()
    for system in systems:
        cs = cases(system)
        for r in range(runs):
            todo = [c for c in cs if f"{system}|{c['id']}|{r}" not in done]
            if not todo:
                continue
            if llm is None:
                from run_prompt_bench import BatchLLM
                llm = BatchLLM(model)
            if system == "rag" and embedder is None:
                from rag import DenseEmbedder
                embedder = DenseEmbedder()
            kw = {"embedder": embedder} if system == "rag" else {}
            recs = RUNNERS[system](llm, todo, r, **kw)
            with out.open("a") as f:
                for c, rec in zip(todo, recs):
                    checks = {k: bool(rec["checks"][k]) for k in c["checks"]}
                    f.write(json.dumps(dict(id=f"{system}|{c['id']}|{r}", model=model, system=system, case=c["id"],
                                            category=c["category"], run=r, sampled=r > 0, question=c["question"],
                                            gold=c["gold"], checks=checks, passed=all(checks.values()),
                                            eval_version=EVAL_VERSION, **{k: v for k, v in rec.items() if k != "checks"}),
                                       default=float) + "\n")
            print(f"{model.split('/')[-1]} {system} run {r}: {len(todo)} cases", flush=True)
    return out


# ------------------------------------------------------------------ Level 2: a small model as judge
JUDGE_SYSTEM = "You grade answers against a reference. Reply with exactly one word: PASS or FAIL."


def reference(c: dict) -> str:
    if c["gold"] is None:
        return "The question cannot be answered from the information available. A correct answer says so."
    if c.get("kind") == "compare":
        return f"The {c['gold']} is heavier."
    if c.get("gold_text"):
        return str(c["gold_text"])
    return str(c["gold"])


def judge(llm, records: list[dict]) -> list[dict]:
    """Ask the model whether each greedy answer matches the reference; compare with the deterministic check."""
    by_case = {(c["system"], c["id"]): c for c in cases()}
    recs = [r for r in records if r["run"] == 0]
    prompts = [[{"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": f"Question: {r['question']}\nReference answer: {reference(by_case[(r['system'], r['case'])])}\n"
                                            f"Answer to grade: {str(r['output'])[-1500:]}\nDoes the answer match the reference? "
                                            "Reply PASS or FAIL."}] for r in recs]
    out = []
    for i in range(0, len(prompts), 16):
        for r, (text, _, _) in zip(recs[i:i + 16], llm.generate(prompts[i:i + 16], 4)):
            primary = next(k for k in PRIMARY if k in r["checks"])
            out.append(dict(model=r["model"], judge=getattr(llm, "name", "judge"), system=r["system"], case=r["case"],
                            verdict=text.strip(), judge_pass=text.strip().upper().startswith("PASS"),
                            check=primary, check_pass=r["checks"][primary]))
    return out


# ------------------------------------------------------------------ analysis
def load(paths: list[Path]) -> list[dict]:
    return [json.loads(line) for p in paths for line in p.open()]


def scorecard(records: list[dict]) -> dict:
    """{model: {system|case: share of runs passed}}: the regression baseline."""
    out: dict = {}
    for r in records:
        out.setdefault(r["model"].split("/")[-1], {}).setdefault(f"{r['system']}|{r['case']}", []).append(r["passed"])
    return {m: {k: float(np.mean(v)) for k, v in sorted(d.items())} for m, d in out.items()}


def stability(rate: float, n: int = RUNS) -> str:
    k = round(rate * n)
    return "passes" if k >= STABLE else "fails" if k <= n - STABLE else "flaky"


def compare(old: dict, new: dict) -> list[dict]:
    """Cases whose stability got worse (or better) between two scorecards."""
    rank = {"fails": 0, "flaky": 1, "passes": 2}
    rows = []
    for m in sorted(set(old) & set(new)):
        for k in sorted(set(old[m]) & set(new[m])):
            a, b = stability(old[m][k]), stability(new[m][k])
            if a != b:
                rows.append(dict(model=m, case=k, before=a, after=b, regression=rank[b] < rank[a]))
    return rows


def cohen_kappa(a: list[bool], b: list[bool]) -> float:
    a, b = np.array(a, float), np.array(b, float)
    po = np.mean(a == b)
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


def summary(paths: list[Path], judge_path: Path | None = None) -> dict:
    recs = load(paths)
    res: dict = {"rows": [], "checks": [], "cases": []}
    print(f"{'model':22} {'system':7} {'greedy passed':>14} {'95% CI':>15} {'stable':>7} {'flaky':>6} {'fails':>6}   "
          f"by category (greedy, predicted)")
    for model in sorted({r["model"] for r in recs}):
        m = model.split("/")[-1]
        for system in SYSTEMS:
            rs = [r for r in recs if r["model"] == model and r["system"] == system]
            if not rs:
                continue
            greedy = {r["case"]: r for r in rs if r["run"] == 0}
            rate = {c: np.mean([r["passed"] for r in rs if r["case"] == c]) for c in greedy}
            st = Counter(stability(v) for v in rate.values())
            k, n = sum(r["passed"] for r in greedy.values()), len(greedy)
            lo, hi = wilson_ci(k, n)
            pred = PREDICTIONS.get((system, m), {})
            cats = {cat: sum(r["passed"] for r in greedy.values() if r["category"] == cat) for cat in CATEGORIES}
            res["rows"].append(dict(model=m, system=system, passed=k, n=n, lo=lo, hi=hi, stable=st["passes"],
                                    flaky=st["flaky"], fails=st["fails"], **{f"{c}": cats[c] for c in CATEGORIES},
                                    **{f"pred_{c}": pred.get(c) for c in CATEGORIES}))
            print(f"{m:22} {system:7} {k:>6} of {n:<5} {lo:6.0%} to {hi:4.0%} {st['passes']:7} {st['flaky']:6} {st['fails']:6}   "
                  + "  ".join(f"{c} {cats[c]} ({pred.get(c, '?')})" for c in CATEGORIES))
            for c, r in sorted(greedy.items()):
                res["cases"].append(dict(model=m, system=system, case=c, category=r["category"], greedy=r["passed"],
                                         rate=float(rate[c]), stability=stability(rate[c]),
                                         failed=[k2 for k2, v in r["checks"].items() if not v]))
            names = sorted({k2 for r in rs for k2 in r["checks"]})
            for name in names:
                vals = [r["checks"][name] for r in rs if name in r["checks"]]
                g = [r["checks"][name] for r in greedy.values() if name in r["checks"]]
                res["checks"].append(dict(model=m, system=system, check=name, greedy=float(np.mean(g)), all_runs=float(np.mean(vals)),
                                          n_cases=len(g)))
    print("\nChecks (share passing: greedy run / all 6 runs)")
    for row in res["checks"]:
        print(f"  {row['model']:22} {row['system']:7} {row['check']:18} {row['greedy']:6.0%} / {row['all_runs']:4.0%}  ({row['n_cases']} cases)")
    print("\nFlaky and failing cases")
    for row in res["cases"]:
        if row["stability"] != "passes":
            print(f"  {row['model']:22} {row['system']:7} {row['case']} {row['category']:13} {row['stability']:6} "
                  f"passed {row['rate']:.0%} of runs; greedy failed {row['failed'] or '-'}")
    if judge_path and judge_path.exists():
        js = [json.loads(line) for line in judge_path.open()]
        res["judge"] = []
        print("\nLevel 2: small model as judge vs the deterministic check (greedy answers)")
        for jm in sorted({j["judge"] for j in js}):
            for model in sorted({j["model"] for j in js}):
                sel = [j for j in js if j["judge"] == jm and j["model"] == model]
                a = [j["judge_pass"] for j in sel]; b = [j["check_pass"] for j in sel]
                agree = float(np.mean(np.array(a) == np.array(b)))
                fp = sum(x and not y for x, y in zip(a, b)); fn = sum(y and not x for x, y in zip(a, b))
                row = dict(judge=jm.split("/")[-1], model=model.split("/")[-1], n=len(sel), agreement=agree,
                           kappa=cohen_kappa(a, b), judge_pass_check_fail=fp, judge_fail_check_pass=fn)
                res["judge"].append(row)
                print(f"  judge {row['judge']} on {row['model']}: agreement {agree:.0%} (kappa {row['kappa']:.2f}), "
                      f"passed {fp} the checks failed, failed {fn} the checks passed, n = {len(sel)}")
    res["scorecard"] = scorecard(recs)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--systems", nargs="+", default=SYSTEMS, choices=SYSTEMS)
    ap.add_argument("--runs", type=int, default=RUNS)
    ap.add_argument("--summary", nargs="*", type=Path)
    ap.add_argument("--compare", nargs=2, type=Path, metavar=("OLD", "NEW"))
    args = ap.parse_args()
    if args.compare:
        rows = compare(*(json.loads(p.read_text()) for p in args.compare))
        for r in rows:
            print(("REGRESSION " if r["regression"] else "improved   ") + f"{r['model']} {r['case']}: {r['before']} -> {r['after']}")
        print(f"{sum(r['regression'] for r in rows)} regressions")
        sys.exit(1 if any(r["regression"] for r in rows) else 0)
    if args.summary is not None:
        summary(args.summary, Path("results/week8/judge.jsonl")); return
    out = Path("results/week8") / (args.model.split("/")[-1] + ".jsonl")
    summary([run(args.model, out, args.systems, args.runs)])


if __name__ == "__main__":
    main()
