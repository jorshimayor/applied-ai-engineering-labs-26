"""Week 7: run the RAG pipeline under each condition, then analyse.

    python src/run_rag_bench.py --model Qwen/Qwen2.5-1.5B-Instruct            # test split, all conditions
    python src/run_rag_bench.py --model Qwen/Qwen2.5-0.5B-Instruct --split dev --n 24 --conditions closed dense_k3
    python src/run_rag_bench.py --sweep                                        # retrieval only, no LLM
    python src/run_rag_bench.py --summary results/week7/*.jsonl

One line per (condition, question) in results/week7/<model>.jsonl; a re-run skips what's done.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from rag import (CHUNK_OVERLAP, CHUNK_WORDS, MAX_NEW_TOKENS, RAG_VERSION, Index, chunk_corpus,  # noqa: E402
                 gold_rank, messages, parse_answer)
from rag_corpus import CORPUS_VERSION, build_corpus, make_questions, regold  # noqa: E402
from stats import discordant, holm, mcnemar_exact, paired_bootstrap_ci, wilson_ci  # noqa: E402

# condition -> (retrieval method or None / "oracle", k, corpus edition)
CONDITIONS = {
    "closed": (None, 0, 1),
    "oracle": ("oracle", 1, 1),
    "dense_k1": ("dense", 1, 1),
    "dense_k3": ("dense", 3, 1),
    "dense_k5": ("dense", 5, 1),
    "dense_k10": ("dense", 10, 1),
    "bm25_k3": ("bm25", 3, 1),
    "hybrid_k3": ("hybrid", 3, 1),
    "swap_dense_k3": ("dense", 3, 2),
}
MAIN = "dense_k3"
H1_MIN = 0.20   # RAG must beat closed-book by at least 20 points
EQUIV = 0.05    # H2 margin: RAG within ±5 points of the oracle
SWEEP_CHUNKS = [(32, 8), (64, 16), (128, 32), (None, 0)]


class Pipeline:
    """The documents, the questions and one index per corpus edition, built once."""

    def __init__(self, split: str, n: int, embedder=None):
        self.embedder = embedder
        self.docs = {1: build_corpus(1), 2: build_corpus(2)}
        q1 = make_questions(split, n, docs=self.docs[1])
        self.questions = {1: q1, 2: regold(q1, self.docs[2])}
        self.by_id = {e: {d["doc_id"]: d for d in ds} for e, ds in self.docs.items()}
        self._index = {}

    def index(self, edition: int) -> Index:
        if edition not in self._index:
            methods = ("bm25", "dense") if self.embedder is not None else ("bm25",)
            self._index[edition] = Index(chunk_corpus(self.docs[edition]), self.embedder, methods)
        return self._index[edition]

    def passages(self, condition: str, q: dict) -> tuple[list[str] | None, list[dict]]:
        method, k, edition = CONDITIONS[condition]
        if method is None:
            return None, []
        if method == "oracle":
            return [self.by_id[edition][q["doc_id"]]["text"]], []
        hits = self.index(edition).search(q["question"], k, method)
        return [h["text"] for h in hits], hits


def run(model: str, conditions: list[str], split: str, n: int, out: Path, batch_size: int = 16,
        llm=None, embedder=None, pipe: Pipeline | None = None) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)["id"] for line in out.open()} if out.exists() else set()
    pipe = pipe or Pipeline(split, n, embedder)
    for cond in conditions:
        edition = CONDITIONS[cond][2]
        todo = [q for q in pipe.questions[edition] if f"{cond}|{q['qid']}" not in done]
        if not todo:
            print(f"{cond}: nothing to do"); continue
        if llm is None:
            from run_prompt_bench import BatchLLM
            llm = BatchLLM(model)
        print(f"{model}, {cond}: {len(todo)} questions -> {out}", flush=True)
        for i in range(0, len(todo), batch_size):
            batch = todo[i:i + batch_size]
            ctx = [pipe.passages(cond, q) for q in batch]
            t0 = time.time()
            gens = llm.generate([messages(q["question"], p) for q, (p, _) in zip(batch, ctx)], MAX_NEW_TOKENS)
            dt = (time.time() - t0) / len(batch)
            with out.open("a") as f:
                for q, (passages, hits), (text, n_in, n_out) in zip(batch, ctx, gens):
                    f.write(json.dumps(dict(
                        id=f"{cond}|{q['qid']}", model=model, condition=cond, qid=q["qid"], split=split,
                        doc_id=q["doc_id"], attribute=q["attribute"], question=q["question"], gold=q["gold"],
                        output=text, **parse_answer(text, q),
                        retrieved=[h["chunk_id"] for h in hits], gold_rank=gold_rank(hits, q) if hits else None,
                        n_passages=len(passages or []), tokens_in=n_in, tokens_out=n_out, seconds=dt,
                        corpus_version=CORPUS_VERSION, rag_version=RAG_VERSION,
                        embed_model=getattr(pipe.embedder, "name", None))) + "\n")
            print(f"  [{min(i + batch_size, len(todo))}/{len(todo)}] {cond}", flush=True)
    return out


# ------------------------------------------------------------------ retrieval only
def retrieval_sweep(embedder, out: Path | None = None, split: str = "test", n: int = 200) -> list[dict]:
    """hit@k and MRR for every chunking x header x method, on the same questions. No LLM."""
    docs = build_corpus(1)
    qs = make_questions(split, n, docs=docs)
    rows = []
    for words, overlap in SWEEP_CHUNKS:
        for header in (False, True):
            ix = Index(chunk_corpus(docs, words, overlap, header), embedder,
                       ("bm25", "dense") if embedder is not None else ("bm25",))
            for method in (["bm25", "dense", "hybrid"] if embedder is not None else ["bm25"]):
                ranks = [gold_rank(ix.search(q["question"], 10, method), q) for q in qs]
                row = dict(chunk_words=words or "whole doc", overlap=overlap, header=header, method=method,
                           n_chunks=len(ix.chunks), mrr=float(np.mean([1 / r if r else 0 for r in ranks])),
                           **{f"hit@{k}": float(np.mean([bool(r and r <= k) for r in ranks])) for k in (1, 3, 5, 10)})
                rows.append(row)
                print(f"{str(row['chunk_words']):>9} words, header {str(header):5} {method:6}  "
                      + "  ".join(f"hit@{k} {row[f'hit@{k}']:.0%}" for k in (1, 3, 5, 10)) + f"  MRR {row['mrr']:.2f}")
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(dict(embed_model=getattr(embedder, "name", None), corpus_version=CORPUS_VERSION,
                                       rag_version=RAG_VERSION, rows=rows), indent=1))
    return rows


# ------------------------------------------------------------------ analysis
def load(path: Path) -> dict[str, dict[int, dict]]:
    by: dict[str, dict[int, dict]] = {}
    for r in map(json.loads, path.open()):
        by.setdefault(r["condition"], {})[r["qid"]] = r
    return by


def table(paths: list[Path]) -> list[dict]:
    """One row per (model, condition): the numbers behind the main results table."""
    rows = []
    for path in paths:
        by = load(path)
        for cond in CONDITIONS:
            if cond not in by:
                continue
            rs = list(by[cond].values()); n = len(rs); k = sum(r["correct"] for r in rs)
            lo, hi = wilson_ci(k, n)
            ranked = [r for r in rs if r["retrieved"]]
            rows.append(dict(model=rs[0]["model"].split("/")[-1], condition=cond, n=n, accuracy=k / n, ci_lo=lo, ci_hi=hi,
                             abstained=np.mean([r["abstained"] for r in rs]),
                             sibling_answer=np.mean([r["sibling"] for r in rs]),
                             answer_retrieved=np.mean([r["gold_rank"] is not None for r in ranked]) if ranked else None,
                             tokens_in=np.mean([r["tokens_in"] for r in rs]),
                             stale=np.mean([r["stale"] for r in rs]) if cond.startswith("swap") else None))
    return rows


def paired(by: dict, a: str, b: str):
    qids = sorted(set(by[a]) & set(by[b]))
    x = np.array([by[a][q]["correct"] for q in qids], float)
    y = np.array([by[b][q]["correct"] for q in qids], float)
    return x, y


def failures(by: dict, cond: str = MAIN) -> dict:
    """Where the main condition's wrong answers come from."""
    rs = list(by[cond].values()); n = len(rs)
    miss = [r for r in rs if r["gold_rank"] is None]
    hit = [r for r in rs if r["gold_rank"] is not None]
    wrong_hit = [r for r in hit if not r["correct"]]
    by_rank = {k: (sum(r["correct"] for r in hit if r["gold_rank"] == k), sum(r["gold_rank"] == k for r in hit))
               for k in sorted({r["gold_rank"] for r in hit})}
    return dict(
        n=n, correct=sum(r["correct"] for r in rs),
        retrieval_miss=len(miss), retrieval_miss_correct=sum(r["correct"] for r in miss),
        read_wrong_sibling=sum(r["sibling"] for r in wrong_hit),
        read_wrong_abstained=sum(r["abstained"] for r in wrong_hit),
        read_wrong_other=sum(not r["sibling"] and not r["abstained"] for r in wrong_hit),
        accuracy_by_gold_rank=by_rank,
        accuracy_by_attribute={a: (sum(r["correct"] for r in rs if r["attribute"] == a), sum(r["attribute"] == a for r in rs))
                               for a in sorted({r["attribute"] for r in rs})},
        miss_with_sibling_retrieved=sum(any(c[0] == r["doc_id"][0] for c in r["retrieved"]) for r in miss))


def summary(paths: list[Path]) -> dict:
    out, h1 = {}, []
    rows = table(paths)
    print(f"{'model':22} {'condition':14} {'n':>4} {'accuracy':>9} {'95% CI':>15} {'abstain':>8} {'sibling':>8} "
          f"{'answer in ctx':>14} {'prompt tok':>10}")
    for r in rows:
        ar = "" if r["answer_retrieved"] is None else f"{r['answer_retrieved']:.1%}"
        print(f"{r['model']:22} {r['condition']:14} {r['n']:4} {r['accuracy']:9.1%} {r['ci_lo']:6.1%} to {r['ci_hi']:5.1%} "
              f"{r['abstained']:8.1%} {r['sibling_answer']:8.1%} {ar:>14} {r['tokens_in']:10.0f}"
              + (f"   stale answers {r['stale']:.1%}" if r["stale"] is not None else ""))
    for path in paths:
        by = load(path)
        model = next(iter(next(iter(by.values())).values()))["model"].split("/")[-1]
        res = out.setdefault(model, {})
        if {MAIN, "closed"} <= set(by):      # H1: RAG beats closed-book
            a, b = paired(by, MAIN, "closed")
            eff, lo, hi = paired_bootstrap_ci(a, b)
            h1.append((model, eff, lo, hi, mcnemar_exact(*discordant(a, b)), discordant(a, b)))
        if {MAIN, "oracle"} <= set(by):      # H2: RAG within ±5 of the oracle
            a, b = paired(by, MAIN, "oracle")
            eff, lo, hi = paired_bootstrap_ci(a, b)
            verdict = "equivalent" if -EQUIV < lo and hi < EQUIV else "different" if hi < -EQUIV or lo > EQUIV else "inconclusive"
            res["h2"] = dict(effect=eff, lo=lo, hi=hi, discordant=discordant(a, b), verdict=verdict)
        if MAIN in by:
            res["failures"] = failures(by)
        for other in ("bm25_k3", "hybrid_k3", "swap_dense_k3", "dense_k1", "dense_k10"):   # exploratory
            if {MAIN, other} <= set(by):
                a, b = paired(by, other, MAIN)
                eff, lo, hi = paired_bootstrap_ci(a, b)
                res.setdefault("vs_main", {})[other] = dict(effect=eff, lo=lo, hi=hi, discordant=discordant(a, b))
    print("\nConfirmatory")
    for (model, eff, lo, hi, p, disc), ph in zip(h1, holm([r[4] for r in h1])):
        ok = ph < 0.05 and lo > 0 and eff >= H1_MIN
        print(f"  H1 {model}: {MAIN} - closed {eff * 100:+.1f} pts (95% CI {lo * 100:+.1f} to {hi * 100:+.1f}), "
              f"discordant {disc}, Holm p = {ph:.2g} -> {'supported' if ok else 'not supported'}")
        out[model]["h1"] = dict(effect=eff, lo=lo, hi=hi, discordant=disc, p_holm=ph, supported=ok)
    for model, res in out.items():
        if "h2" in res:
            h = res["h2"]
            print(f"  H2 {model}: {MAIN} - oracle {h['effect'] * 100:+.1f} pts (95% CI {h['lo'] * 100:+.1f} to "
                  f"{h['hi'] * 100:+.1f}), discordant {h['discordant']}, margin ±{EQUIV * 100:.0f} -> {h['verdict']}")
    print("\nExploratory (no significance claims)")
    for model, res in out.items():
        for other, v in res.get("vs_main", {}).items():
            print(f"  {model}: {other} - {MAIN} {v['effect'] * 100:+.1f} pts (95% CI {v['lo'] * 100:+.1f} to {v['hi'] * 100:+.1f})")
        f = res.get("failures")
        if f:
            print(f"  {model} {MAIN}: {f['correct']}/{f['n']} correct. Answer not retrieved: {f['retrieval_miss']} "
                  f"({f['retrieval_miss_correct']} still correct). Retrieved but wrong: sibling's value "
                  f"{f['read_wrong_sibling']}, abstained {f['read_wrong_abstained']}, other {f['read_wrong_other']}")
            print(f"     accuracy by rank of the answer chunk: "
                  + ", ".join(f"rank {k}: {c}/{n}" for k, (c, n) in f["accuracy_by_gold_rank"].items()))
            print(f"     by attribute: " + ", ".join(f"{a} {c}/{n}" for a, (c, n) in f["accuracy_by_attribute"].items()))
    out["_table"] = rows
    return out


def examples(path: Path, condition: str = MAIN, kind: str = "wrong", n: int = 5) -> list[dict]:
    """A few records to read by eye: kind is "wrong", "miss" (answer not retrieved) or "sibling"."""
    rs = list(load(path)[condition].values())
    pick = {"wrong": lambda r: not r["correct"], "miss": lambda r: r["gold_rank"] is None and r["retrieved"],
            "sibling": lambda r: r["sibling"]}[kind]
    return [r for r in rs if pick(r)][:n]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=list(CONDITIONS))
    ap.add_argument("--split", default="test", choices=["dev", "test"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--sweep", action="store_true", help="retrieval-only sweep, writes results/week7/retrieval.json")
    ap.add_argument("--summary", nargs="*", type=Path)
    args = ap.parse_args()
    if args.summary is not None:
        summary(args.summary); return
    from rag import DenseEmbedder
    emb = DenseEmbedder()
    if args.sweep:
        retrieval_sweep(emb, Path("results/week7/retrieval.json")); return
    name = args.model.split("/")[-1] + ".jsonl"
    out = args.out or (Path("results/week7") / name if args.split == "test" else Path("results/week7/dev") / name)
    summary([run(args.model, args.conditions, args.split, args.n, out, args.batch_size, embedder=emb)])


if __name__ == "__main__":
    main()

