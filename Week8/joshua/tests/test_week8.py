"""Week 8 checks that need no model: the eval cases, every check function, and a full run of the
eval harness with a fake model and a hashed bag-of-words embedder.

    uv run python tests/test_week8.py
"""
from __future__ import annotations

import json
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from eval_cases import CATEGORIES, PREDICTIONS, cases  # noqa: E402
from rag import HashEmbedder, chunk_doc  # noqa: E402
from rag_corpus import build_corpus  # noqa: E402
from run_evals import (JUDGE_SYSTEM, compare, judge, rag_checks, run, stability, summary,  # noqa: E402
                       tool_checks)


def test_cases():
    cs = cases()
    assert len(cs) == 45 and len({(c["system"], c["id"]) for c in cs}) == 45
    for system in ("prompt", "tools", "rag"):
        assert Counter(c["category"] for c in cases(system)) == {k: 5 for k in CATEGORIES}, system
    by = {c["id"]: c for c in cs}
    assert by["p06"]["gold"] == 44174 and by["p08"]["gold"] == 82800 and by["p11"]["gold"] == -11
    assert by["t03"]["gold"] == 4.2 and by["t15"]["gold"] == 28.22   # 50 - (19.30 + 4.90) * 0.9
    for c in cases("tools"):   # money answers have at most 2 decimals, so rounding order can't matter
        assert c["gold"] is None or round(c["gold"], 2) == c["gold"], c["id"]
    docs = {d["doc_id"]: d for d in build_corpus()}
    for c in cases("rag"):
        if c["kind"] in ("number", "choice"):
            assert c["gold_text"] in docs[c["doc_id"]]["text"] and c["gold"] not in c["sibling_values"], c["id"]
    assert by["r14"]["gold"] in ("C-200", "C-300") and by["r15"]["extra_docs"]
    assert chunk_doc(by["r15"]["extra_docs"][0])
    for (system, model), pred in PREDICTIONS.items():
        assert set(pred) == set(CATEGORIES) and all(0 <= v <= 5 for v in pred.values())


def _call(name, args):
    return f'<tool_call>\n{json.dumps({"name": name, "arguments": args})}\n</tool_call>'


def test_tool_checks():
    c = dict(items=["basil pesto", "mozzarella"])
    good = [{"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "lookup_price"}}] * 2},
            {"role": "tool", "name": "lookup_price", "content": '{"price": 5.6}'},
            {"role": "tool", "name": "lookup_price", "content": '{"price": 8.15}'},
            {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "calculate"}}]},
            {"role": "tool", "name": "calculate", "content": '{"result": 33.1}'}]
    rec = dict(messages=good, correct=True, malformed=[], status="final", turns=3)
    assert tool_checks(rec, c) == dict(correct=True, no_malformed=True, prices_first=True, known_items=True,
                                       no_answer=False, efficient=False)
    early = [{"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "lookup_price"}},
                                                                  {"function": {"name": "calculate"}}]}]
    assert not tool_checks(dict(rec, messages=early), c)["prices_first"]
    unknown = good[:1] + [{"role": "tool", "name": "lookup_price", "content": '{"error": "Unknown item"}'}]
    assert not tool_checks(dict(rec, messages=unknown), c)["known_items"]


def test_rag_checks():
    by = {c["id"]: c for c in cases("rag")}
    c = by["r02"]
    hit = [dict(chunk_id=f"{c['doc_id']}#1", doc_id=c["doc_id"], text=f"weighs {c['gold_text']}", rank=1)]
    ok = rag_checks(c["gold_text"], c, hit)
    assert ok["correct"] and ok["no_sibling"] and ok["answer_retrieved"]
    assert not rag_checks(str(c["sibling_values"][0]), c, [])["no_sibling"]
    assert rag_checks("unknown", by["r12"], [])["abstains"]
    assert rag_checks("The context does not mention battery capacity.", by["r13"], [])["abstains"]
    assert not rag_checks("2,339", by["r12"], [])["abstains"]
    win, lose = by["r14"]["doc_id"], by["r14"]["other"]
    assert rag_checks(f"The {win}", by["r14"], [])["compares"]
    assert rag_checks(f"The {win} is heavier than the {lose}", by["r14"], [])["compares"]
    assert not rag_checks(f"The {lose}", by["r14"], [])["compares"]
    assert not rag_checks("999", by["r15"], [])["ignores_injection"]


class FakeLLM:
    """Answers each system correctly from the case list, except where noted, so every check is exercised."""

    name = "fake"

    def __init__(self):
        self.by_q = {c["question"]: c for c in cases()}

    def generate(self, conversations, max_new_tokens, tools=None, **sampling):
        return [(self.reply(c, tools, sampling), 50, 10) for c in conversations]

    def reply(self, conv, tools, sampling):
        system = conv[0]["content"]
        if system == JUDGE_SYSTEM:
            return "PASS"
        if tools:   # tool agent: look everything up, then answer
            c = self.by_q[conv[1]["content"]]
            if not any(m["role"] == "tool" for m in conv):
                items = re.findall(r"of ([a-z ]+?)(?: and | only|\.|,| can)", c["question"]) if c["gold"] is None else c["items"]
                return "".join(_call("lookup_price", {"item": i}) for i in items)
            return json.dumps({"answer": c["gold"] if c["gold"] is not None else 10.96})
        user = conv[-1]["content"]
        if "Context:" in user:   # RAG
            c = self.by_q[user.split("Question: ")[-1]]
            if c["kind"] == "abstain":
                return "unknown"
            if c["kind"] == "compare":
                return c["gold"]
            if sampling and c["id"] == "r15":
                return "999"
            return c["gold_text"] if c["gold_text"] in user else "unknown"
        c = self.by_q[user]   # prompt system
        return "Not enough information to answer." if c["gold"] is None else f"Working.\nThe answer is {c['gold']}."


def test_stability_and_compare():
    assert [stability(k / 6) for k in range(7)] == ["fails", "fails", "flaky", "flaky", "flaky", "passes", "passes"]
    old = {"m": {"rag|r01": 1.0, "rag|r02": 0.5, "rag|r03": 0.0}}
    new = {"m": {"rag|r01": 0.5, "rag|r02": 1.0, "rag|r03": 0.0}}
    rows = compare(old, new)
    assert [(r["case"], r["regression"]) for r in rows] == [("rag|r01", True), ("rag|r02", False)]


def test_run_and_summary():
    llm, emb = FakeLLM(), HashEmbedder()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "fake.jsonl"
        run("fake/Qwen2.5-1.5B-Instruct", path, runs=2, llm=llm, embedder=emb)
        recs = [json.loads(line) for line in path.open()]
        assert len(recs) == 45 * 2
        run("fake/Qwen2.5-1.5B-Instruct", path, runs=2, llm=llm, embedder=emb)   # resumable: nothing new
        assert len(path.read_text().splitlines()) == 90
        by = {(r["case"], r["run"]): r for r in recs}
        assert all(by[(f"p{i:02d}", 0)]["passed"] for i in range(1, 16))
        assert not by[("t11", 0)]["passed"] and by[("t11", 0)]["checks"] == {"no_answer": False}
        assert by[("t01", 0)]["passed"] and by[("t12", 0)]["checks"]["efficient"]
        assert by[("r12", 0)]["passed"] and by[("r14", 0)]["passed"]
        assert not by[("r15", 1)]["checks"]["ignores_injection"]
        jpath = Path(tmp) / "judge.jsonl"
        with jpath.open("w") as f:
            for j in judge(llm, recs):
                f.write(json.dumps(j) + "\n")
        res = summary([path], jpath)
        row = next(r for r in res["rows"] if r["system"] == "prompt")
        assert row["passed"] == 15 and row["pred_typical"] == 4
        assert res["judge"][0]["n"] == 45 and 0 <= res["judge"][0]["agreement"] <= 1
        assert res["scorecard"]["Qwen2.5-1.5B-Instruct"]["tools|t11"] == 0.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all Week 8 checks passed")
