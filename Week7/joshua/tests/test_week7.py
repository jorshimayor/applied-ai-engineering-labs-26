"""Week 7 checks that need no model: the corpus, chunking, BM25, answer scoring, and the whole
run-and-analyse loop driven by a fake model and a hashed bag-of-words embedder.

    uv run python tests/test_week7.py
"""
from __future__ import annotations

import json
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from rag import BM25, HashEmbedder, Index, chunk_corpus, chunk_doc, gold_rank, parse_answer  # noqa: E402
from rag_corpus import ATTRIBUTES, CHANGED_IN_EDITION_2, build_corpus, make_questions, regold  # noqa: E402
from run_rag_bench import CONDITIONS, Pipeline, run, summary  # noqa: E402


def test_corpus():
    docs, docs2 = build_corpus(), build_corpus(2)
    assert docs == build_corpus(), "corpus must be deterministic"
    assert len(docs) == 60 and sum(d["split"] == "test" for d in docs) == 48
    test, dev = make_questions("test", 200), make_questions("dev", 84)
    assert len({(q["doc_id"], q["attribute"]) for q in test}) == 200
    assert not {q["doc_id"] for q in test} & {q["doc_id"] for q in dev}, "dev and test share a product"
    by = {d["doc_id"]: d for d in docs}
    for q in test + dev:
        assert q["gold_text"] in by[q["doc_id"]]["text"], q
        assert q["gold"] not in q["sibling_values"], f"siblings must differ: {q}"
    for a, b in zip(docs, docs2):
        changed = {k for k in ATTRIBUTES if a["values"][k] != b["values"][k]}
        assert changed == set(CHANGED_IN_EDITION_2), (a["doc_id"], changed)
    swapped = regold(test, docs2)
    assert all(q["gold"] != q["old_gold"] for q in swapped if q["attribute"] in CHANGED_IN_EDITION_2)


def test_chunking():
    doc = build_corpus()[0]
    words = doc["text"].split()
    chunks = chunk_doc(doc, 64, 16)
    assert chunks[0]["start"] == 0 and chunks[-1]["end"] == len(words)
    for a, b in zip(chunks, chunks[1:]):
        assert a["end"] - b["start"] == 16, "windows must overlap by 16 words"
    assert all(len(c["text"].split()) <= 64 for c in chunks)
    assert chunk_doc(doc, None)[0]["text"] == " ".join(words)
    assert all(c["text"].startswith(doc["title"] + ": ") for c in chunk_doc(doc, 64, 16, header=True))


def test_bm25_and_index():
    docs, qs = build_corpus(), make_questions("test", 200)
    bm = BM25(["the cat sat", "a dog ran", "the cat and the dog"])
    assert bm.scores("cat").argmax() in (0, 2) and bm.scores("zebra").sum() == 0
    ix = Index(chunk_corpus(docs, None), methods=("bm25",))   # whole documents: the code should find its doc
    assert all(ix.search(q["question"], 1, "bm25")[0]["doc_id"] == q["doc_id"] for q in qs)
    ix = Index(chunk_corpus(docs), HashEmbedder())
    for method in ("bm25", "dense", "hybrid"):
        hits = ix.search(qs[0]["question"], 5, method)
        assert [h["rank"] for h in hits] == [1, 2, 3, 4, 5]
        assert hits[0]["score"] >= hits[-1]["score"]
    hits = ix.search(qs[0]["question"], 10, "bm25")
    r = gold_rank(hits, qs[0])
    assert r is not None and qs[0]["gold_text"] in hits[r - 1]["text"]


SCORE_CASES = [   # (reply, attribute, gold, siblings, correct, sibling, abstained)
    ("16.1 kg", "weight", 16.1, [3.2], True, False, False),
    ("The C-200 weighs 16.1 kg.", "weight", 16.1, [3.2], True, False, False),
    ("3.2", "weight", 16.1, [3.2], False, True, False),
    ("€2,339", "price", 2339, [999], True, False, False),
    ("2339 euros", "price", 2339, [999], True, False, False),
    ("60 months", "warranty", 60, [12], True, False, False),
    ("unknown", "warranty", 60, [12], False, False, True),
    ("sky blue", "colour", "sky blue", ["navy blue"], True, False, False),
    ("Sky Blue.", "colour", "sky blue", ["navy blue"], True, False, False),
    ("sky blue or navy blue", "colour", "sky blue", ["navy blue"], False, False, False),
    ("blue", "colour", "sky blue", ["navy blue"], False, False, False),
    ("navy blue", "colour", "sky blue", ["navy blue"], False, True, False),
    ("Hollinby", "town", "Hollinby", ["Dunholt"], True, False, False),
    ("2024", "year", 2024, [2019], True, False, False),
]


def test_scoring():
    for reply, attr, gold, sibs, correct, sibling, abstained in SCORE_CASES:
        q = dict(doc_id="C-200", name="Corvane C-200", attribute=attr, kind=ATTRIBUTES[attr][1], gold=gold, sibling_values=sibs)
        got = parse_answer(reply, q)
        assert (got["correct"], got["sibling"], got["abstained"]) == (correct, sibling, abstained), (reply, got)


class FakeLLM:
    """Answers with the gold value when it appears in the prompt's context, otherwise 'unknown'."""

    name = "fake"

    def __init__(self, questions: list[dict]):
        self.gold = {}
        for q in questions:   # both editions: answer with whichever value the context holds
            self.gold.setdefault(q["question"], []).append(q["gold_text"])

    def generate(self, conversations, max_new_tokens, tools=None):
        out = []
        for conv in conversations:
            user = conv[-1]["content"]
            question = user.split("Question: ")[-1]
            ctx = user.split("Question: ")[0] if "Context:" in user else ""
            code = re.search(r"[A-O]-\d00", question).group()   # read only passages about this product
            ctx = " ".join(p for p in ctx.split("\n\n") if code in p)
            text = next((g for g in self.gold[question] if g in ctx), "unknown")
            out.append((text, len(user.split()), 2))
        return out


def test_run_and_summary():
    pipe = Pipeline("dev", 24, HashEmbedder())
    llm = FakeLLM(pipe.questions[1] + pipe.questions[2])
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "fake.jsonl"
        run("fake", list(CONDITIONS), "dev", 24, path, 8, llm=llm, pipe=pipe)
        rows = [json.loads(line) for line in path.open()]
        assert len(rows) == 24 * len(CONDITIONS)
        run("fake", list(CONDITIONS), "dev", 24, path, 8, llm=llm, pipe=pipe)   # re-run skips everything
        assert len(path.read_text().splitlines()) == len(rows)
        by = {}
        for r in rows:
            by.setdefault(r["condition"], []).append(r)
        assert not any(r["correct"] for r in by["closed"]) and all(r["abstained"] for r in by["closed"])
        assert all(r["correct"] for r in by["oracle"])
        for cond in ("bm25_k3", "dense_k3", "hybrid_k3", "swap_dense_k3"):   # fake reads perfectly: correct iff retrieved
            assert all(r["correct"] == (r["gold_rank"] is not None) for r in by[cond]), cond
        assert all(len(r["retrieved"]) == 10 for r in by["dense_k10"])
        res = summary([path])
        assert res["fake"]["h1"]["effect"] > 0 and "h2" in res["fake"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all Week 7 checks passed")
