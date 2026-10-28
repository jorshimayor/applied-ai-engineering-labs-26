"""Week 5: run the prompt library on GSM8K with batched generation. GPU if available.

    python src/run_prompt_bench.py --model Qwen/Qwen2.5-1.5B-Instruct
    python src/run_prompt_bench.py --summary results/week5/*.jsonl

Results append one line per (strategy, problem) to results/week5/<model>.jsonl,
so a re-run skips what's done.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from gsm8k import is_correct, sample  # noqa: E402
from prompts import BASELINE, LIBRARY_VERSION, STRATEGIES  # noqa: E402
from stats import discordant, holm, mcnemar_exact, paired_bootstrap_ci, wilson_ci  # noqa: E402


class BatchLLM:
    """Greedy batched generation from a Hugging Face chat model (left padding)."""

    def __init__(self, name: str, device: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch, self.name = torch, name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if self.device == "cuda" else torch.float32
        self.tok = AutoTokenizer.from_pretrained(name, padding_side="left")
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype).to(self.device).eval()

    def generate(self, conversations: list[list[dict]], max_new_tokens: int) -> list[tuple[str, int, int]]:
        texts = [self.tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True) for c in conversations]
        enc = self.tok(texts, return_tensors="pt", padding=True).to(self.device)
        with self.torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                      pad_token_id=self.tok.pad_token_id)
        width = enc.input_ids.shape[1]
        results = []
        for i in range(len(texts)):
            new = out[i, width:]
            stop = (new == self.tok.eos_token_id) | (new == self.tok.pad_token_id)
            n_out = int(stop.float().argmax()) if stop.any() else len(new)
            results.append((self.tok.decode(new[:n_out], skip_special_tokens=True),
                            int(enc.attention_mask[i].sum()), n_out))
        return results


def run(model: str, strategies: list[str], n: int, out: Path, batch_size: int = 16, llm=None) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["id"] for l in out.open()} if out.exists() else set()
    problems = sample(n)
    todo = {s: [p for p in problems if f"{s}|{p['id']}" not in done] for s in strategies}
    total = sum(map(len, todo.values()))
    if not total:
        print(f"{out}: nothing to do ({len(done)} rows saved)")
        return out
    llm = llm or BatchLLM(model)
    print(f"{model} on {llm.device}; {total} generations to go -> {out}")
    t_start, n_done = time.time(), 0
    for s_name, items in todo.items():
        s = STRATEGIES[s_name]
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            t0 = time.time()
            gens = llm.generate([s.messages(p["question"]) for p in batch], s.max_new_tokens)
            dt = (time.time() - t0) / len(batch)
            with out.open("a") as f:
                for p, (text, n_in, n_out) in zip(batch, gens):
                    parsed = s.parse(text)
                    f.write(json.dumps(dict(
                        id=f"{s_name}|{p['id']}", model=model, strategy=s_name, library_version=LIBRARY_VERSION,
                        pid=p["id"], gold=p["gold"], output=text, **parsed,
                        correct=is_correct(parsed["pred_strict"], p["gold"]),
                        correct_lenient=is_correct(parsed["pred_lenient"], p["gold"]),
                        tokens_in=n_in, tokens_out=n_out, seconds=dt)) + "\n")
            n_done += len(batch)
            eta = (time.time() - t_start) / n_done * (total - n_done)
            print(f"[{n_done}/{total}] {s_name:18} batch {i // batch_size + 1}  {dt:4.2f} s/problem  ETA {eta / 60:4.1f} min", flush=True)
    return out


def load(path: Path) -> dict[str, dict[int, dict]]:
    by: dict[str, dict[int, dict]] = {}
    for r in map(json.loads, path.open()):
        by.setdefault(r["strategy"], {})[r["pid"]] = r
    return by


def summary(paths: list[Path]) -> None:
    for path in paths:
        by = load(path)
        model = next(iter(next(iter(by.values())).values()))["model"]
        print(f"\n=== {model}  ({path})")
        print(f"{'strategy':18} {'n':>4} {'strict':>7} {'95% CI':>15} {'lenient':>8} {'format fail':>12} {'tokens out':>11}")
        for s in STRATEGIES:
            if s not in by:
                continue
            rs = list(by[s].values()); n = len(rs); k = sum(r["correct"] for r in rs)
            lo, hi = wilson_ci(k, n)
            print(f"{s:18} {n:4} {k / n:7.1%} {lo:6.1%} to {hi:5.1%} {sum(r['correct_lenient'] for r in rs) / n:8.1%} "
                  f"{1 - sum(r['format_ok'] for r in rs) / n:12.1%} {sum(r['tokens_out'] for r in rs) / n:11.0f}")
        if BASELINE in by:
            others = [s for s in STRATEGIES if s in by and s != BASELINE]
            rows = []
            for s in others:
                pids = sorted(set(by[s]) & set(by[BASELINE]))
                a = [by[s][p]["correct"] for p in pids]; b = [by[BASELINE][p]["correct"] for p in pids]
                eff, lo, hi = paired_bootstrap_ci(a, b)
                rows.append((s, eff, lo, hi, mcnemar_exact(*discordant(a, b))))
            for (s, eff, lo, hi, p), ph in zip(rows, holm([r[4] for r in rows])):
                print(f"  {s:18} vs {BASELINE}: {eff * 100:+6.1f} pts (95% CI {lo * 100:+.1f} to {hi * 100:+.1f}), Holm p = {ph:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--strategies", nargs="+", default=list(STRATEGIES), choices=list(STRATEGIES))
    ap.add_argument("--n", type=int, default=400, help="GSM8K test problems (fixed seeded sample)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", type=Path, help="default: results/week5/<model>.jsonl")
    ap.add_argument("--summary", nargs="*", type=Path, help="only print summaries of these files")
    args = ap.parse_args()
    if args.summary is not None:
        summary(args.summary)
        return
    out = args.out or Path("results/week5") / f"{args.model.split('/')[-1]}.jsonl"
    summary([run(args.model, args.strategies, args.n, out, args.batch_size)])


if __name__ == "__main__":
    main()
