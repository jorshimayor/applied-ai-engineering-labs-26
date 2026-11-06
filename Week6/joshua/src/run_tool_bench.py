"""Week 6: run the two-tool pipeline over both routes (direct functions, MCP server).

    python src/run_tool_bench.py --model Qwen/Qwen2.5-1.5B-Instruct            # test split, both routes
    python src/run_tool_bench.py --model Qwen/Qwen2.5-0.5B-Instruct --split dev --n 12
    python src/run_tool_bench.py --summary results/week6/*.jsonl

One line per (route, problem) in results/week6/<model>.jsonl; a re-run skips what's done.
"""
from __future__ import annotations

import argparse
import inspect
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import tool_agent  # noqa: E402
import tools  # noqa: E402
from run_prompt_bench import BatchLLM  # noqa: E402
from stats import discordant, holm, mcnemar_exact, paired_bootstrap_ci, wilson_ci  # noqa: E402
from tool_agent import MCPTools, run_problems  # noqa: E402
from tool_tasks import TASKS_VERSION, make_problems  # noqa: E402
from tools import MALFORMED, TOOLS_VERSION, DirectTools  # noqa: E402

ROUTES = {"direct": DirectTools, "mcp": MCPTools}
EQUIV = 0.05   # H2 margin: MCP within ±5 points of direct


def run(model: str, routes: list[str], split: str, n: int, out: Path, batch_size: int = 16, llm=None) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["id"] for l in out.open()} if out.exists() else set()
    problems = make_problems(split, n)
    for route in routes:
        todo = [p for p in problems if f"{route}|{p['pid']}" not in done]
        if not todo:
            print(f"{route}: nothing to do"); continue
        llm = llm or BatchLLM(model)
        backend = ROUTES[route]()
        print(f"{model} on {llm.device}, route {route}: {len(todo)} problems -> {out}", flush=True)
        n_done = [0]

        def save(rec):
            rec.update(model=model, tasks_version=TASKS_VERSION, tools_version=TOOLS_VERSION)
            with out.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            n_done[0] += 1
            if n_done[0] % batch_size == 0 or n_done[0] == len(todo):
                print(f"  [{n_done[0]}/{len(todo)}] {route}", flush=True)
        try:
            run_problems(llm, todo, backend, batch_size=batch_size, on_done=save)
        finally:
            backend.close()
    return out


def loc(obj) -> int:
    """Non-blank, non-comment source lines."""
    src = inspect.getsource(obj) if not isinstance(obj, Path) else obj.read_text()
    return sum(1 for line in src.splitlines() if line.strip() and not line.strip().startswith("#"))


def loc_section(path: Path, start: str, end: str) -> int:
    text = path.read_text()
    body = text[text.index(start):text.index(end)]
    return sum(1 for line in body.splitlines() if line.strip() and not line.strip().startswith("#"))


def load(path: Path) -> dict[str, dict[int, dict]]:
    by: dict[str, dict[int, dict]] = {}
    for r in map(json.loads, path.open()):
        by.setdefault(r["route"], {})[r["pid"]] = r
    return by


def summary(paths: list[Path]) -> dict:
    out, h1 = {}, []
    for path in paths:
        by = load(path)
        model = next(iter(next(iter(by.values())).values()))["model"]
        print(f"\n=== {model}  ({path})")
        print(f"{'route':7} {'n':>4} {'success':>8} {'95% CI':>15} {'fail-closed':>12} {'runs w/ malformed':>18} "
              f"{'recovered':>10} {'tool calls':>11} {'tool errs':>10} {'turns':>6} {'tool ms':>8}")
        for route in ROUTES:
            if route not in by:
                continue
            rs = list(by[route].values()); n = len(rs)
            k = sum(r["correct"] for r in rs); lo, hi = wilson_ci(k, n)
            bad = [r for r in rs if r["malformed"]]
            ms = np.median([t for r in rs for t in r["tool_seconds"]]) * 1000 if any(r["tool_seconds"] for r in rs) else float("nan")
            print(f"{route:7} {n:4} {k / n:8.1%} {lo:6.1%} to {hi:5.1%} {sum(r['correct_fail_closed'] for r in rs) / n:12.1%} "
                  f"{len(bad) / n:18.1%} {(sum(r['correct'] for r in bad) / len(bad) if bad else float('nan')):10.1%} "
                  f"{np.mean([r['tool_calls'] for r in rs]):11.2f} {np.mean([r['tool_errors'] for r in rs]):10.2f} "
                  f"{np.mean([r['turns'] for r in rs]):6.2f} {ms:8.2f}")
            types = Counter(e["type"] for r in rs for e in r["malformed"])
            status = Counter(r["status"] for r in rs)
            print(f"        malformed replies: {dict(types) or 'none'} | end states: {dict(status)}")
        if "direct" in by:   # H1: repair loop vs fail-closed, on the direct route
            pids = sorted(by["direct"])
            a = np.array([by["direct"][p]["correct"] for p in pids], float)
            b = np.array([by["direct"][p]["correct_fail_closed"] for p in pids], float)
            eff, lo, hi = paired_bootstrap_ci(a, b)
            h1.append((model, eff, lo, hi, mcnemar_exact(*discordant(a, b))))
        if {"direct", "mcp"} <= set(by):   # H2: MCP vs direct, equivalence within ±5 points
            pids = sorted(set(by["direct"]) & set(by["mcp"]))
            a = np.array([by["mcp"][p]["correct"] for p in pids], float)
            b = np.array([by["direct"][p]["correct"] for p in pids], float)
            eff, lo, hi = paired_bootstrap_ci(a, b)
            verdict = "equivalent" if -EQUIV < lo and hi < EQUIV else "different" if hi < -EQUIV or lo > EQUIV else "inconclusive"
            print(f"  H2 MCP - direct: {eff * 100:+.1f} pts (95% CI {lo * 100:+.1f} to {hi * 100:+.1f}), "
                  f"discordant {discordant(a, b)}, margin ±{EQUIV * 100:.0f} -> {verdict}")
            out.setdefault(model, {})["h2"] = dict(effect=eff, lo=lo, hi=hi, verdict=verdict)
    for (model, eff, lo, hi, p), ph in zip(h1, holm([r[4] for r in h1])):
        print(f"  H1 {model}: repair - fail-closed {eff * 100:+.1f} pts (95% CI {lo * 100:+.1f} to {hi * 100:+.1f}), Holm p = {ph:.2g}")
        out.setdefault(model, {})["h1"] = dict(effect=eff, lo=lo, hi=hi, p_holm=ph)
    failures(paths)
    here = Path(__file__).parent
    direct = loc_section(here / "tools.py", "# -- direct route", "# -- parsing")
    print(f"\nIntegration code (non-blank, non-comment lines): direct route = {direct} (hand-written schemas "
          f"+ DirectTools); MCP route = server {loc(here / 'mcp_server.py')} + client {loc(tool_agent.MCPTools)}")
    return out


def _malformed_shape(text: str, mtype: str) -> str:
    """Finer label for a malformed reply, from its text (exploratory)."""
    t = text.strip()
    if t.startswith(">"):
        return "stray '>' before the JSON"
    if t.startswith('{"name"'):
        return "tool call without <tool_call> tags"
    if '"answer": "' in t:
        return "answer as a string"
    if mtype == "json_error" and "lookup_price(" in t:
        return "lookup_price nested inside calculate"
    return mtype


def failures(paths: list[Path]) -> dict:
    """Exploratory: where runs went wrong, beyond the malformed-reply counts."""
    out = {}
    print("\nExploratory: how runs failed (share of the 200 runs unless stated)")
    print(f"{'model':22} {'route':7} {'calc in 1st reply':>17} {'lookup err':>11} {'no good calc':>13} "
          f"{'success | calc ok':>18} {'success | no calc ok':>21} {'same bad reply':>15}")
    for path in paths:
        for route, recs in load(path).items():
            rs = list(recs.values()); n = len(rs)
            model = rs[0]["model"].split("/")[-1]
            early = no_calc = lookup_err = repeats = 0
            ok = [0, 0]; nok = [0, 0]
            shapes = Counter()
            for r in rs:
                first = next(m for m in r["messages"] if m["role"] == "assistant")
                names = [c["function"]["name"] for c in first.get("tool_calls") or []]
                early += "calculate" in names and "lookup_price" in names
                lookup_err += any(m["role"] == "tool" and m["name"] == "lookup_price" and '"error"' in m["content"]
                                  for m in r["messages"])
                sums = [json.loads(m["content"])["result"] for m in r["messages"]
                        if m["role"] == "tool" and m["name"] == "calculate" and '"result"' in m["content"]]
                if r["status"] == "final" and not sums:
                    no_calc += 1
                bucket = ok if sums else nok
                bucket[0] += 1; bucket[1] += r["correct"]
                outs = r["malformed_outputs"]
                repeats += len(outs) > 1 and len(set(outs)) == 1
                shapes.update(_malformed_shape(o, e["type"]) for o, e in zip(outs, r["malformed"]))
            row = dict(calc_in_first_reply=early / n, lookup_error=lookup_err / n, final_without_calc=no_calc / n,
                       n_calc_ok=ok[0], success_calc_ok=ok[1] / max(ok[0], 1),
                       n_no_calc_ok=nok[0], success_no_calc_ok=nok[1] / max(nok[0], 1), same_bad_reply=repeats / n,
                       malformed_shapes=dict(shapes))
            out.setdefault(model, {})[route] = row
            print(f"{model:22} {route:7} {row['calc_in_first_reply']:17.1%} {row['lookup_error']:11.1%} "
                  f"{row['final_without_calc']:13.1%} {row['success_calc_ok']:8.1%} of {ok[0]:>3} "
                  f"{row['success_no_calc_ok']:11.1%} of {nok[0]:>3} "
                  f"{row['same_bad_reply']:15.1%}")
            print(f"{'':30} malformed shapes: {dict(shapes.most_common())}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--routes", nargs="+", default=list(ROUTES), choices=list(ROUTES))
    ap.add_argument("--split", default="test", choices=["dev", "test"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", type=Path, help="default: results/week6/<model>.jsonl (test) or results/week6/dev/<model>.jsonl")
    ap.add_argument("--summary", nargs="*", type=Path, help="only print summaries of these files")
    args = ap.parse_args()
    if args.summary is not None:
        summary(args.summary)
        return
    name = args.model.split("/")[-1] + ".jsonl"
    out = args.out or (Path("results/week6") / name if args.split == "test" else Path("results/week6/dev") / name)
    summary([run(args.model, args.routes, args.split, args.n, out, args.batch_size)])


if __name__ == "__main__":
    main()
