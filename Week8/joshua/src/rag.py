"""Week 7: a minimal retrieval-augmented generation pipeline.

    chunk the documents -> index the chunks (BM25, dense embeddings, or both)
    -> retrieve the top k for a question -> put them in the prompt -> generate -> score

Each step is a small function so the notebook can show it on its own.
"""
from __future__ import annotations

import math
import zlib
import re
from collections import Counter

import numpy as np

RAG_VERSION = "1.0"
CHUNK_WORDS, CHUNK_OVERLAP = 64, 16          # the fixed main configuration
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
MAX_NEW_TOKENS = 24


# ------------------------------------------------------------------ chunking
def chunk_doc(doc: dict, words: int | None = CHUNK_WORDS, overlap: int = CHUNK_OVERLAP,
              header: bool = False) -> list[dict]:
    """Split one document into windows of `words` words that overlap by `overlap` words.

    words=None keeps the whole document as one chunk. header=True starts every chunk with the
    document title, so a chunk that only says "it" still names its product.
    """
    toks = doc["text"].split()
    if words is None or words >= len(toks):
        spans = [(0, len(toks))]
    else:
        step = max(1, words - overlap)
        spans = []
        for start in range(0, len(toks), step):
            spans.append((start, min(start + words, len(toks))))
            if start + words >= len(toks):
                break
    out = []
    for i, (s, e) in enumerate(spans):
        body = " ".join(toks[s:e])
        out.append(dict(chunk_id=f"{doc['doc_id']}#{i}", doc_id=doc["doc_id"], start=s, end=e,
                        text=f"{doc['title']}: {body}" if header else body))
    return out


def chunk_corpus(docs: list[dict], words: int | None = CHUNK_WORDS, overlap: int = CHUNK_OVERLAP,
                 header: bool = False) -> list[dict]:
    return [c for d in docs for c in chunk_doc(d, words, overlap, header)]


# ------------------------------------------------------------------ sparse retrieval
def tokenize(text: str) -> list[str]:
    """Lower-case words; keeps codes like "c-200" and numbers like "16.1" whole."""
    return re.findall(r"[a-z0-9€]+(?:[-.,][a-z0-9]+)*", text.lower())


class BM25:
    """Okapi BM25 (k1 = 1.5, b = 0.75) over a list of texts."""

    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(t)) for t in texts]
        self.lens = np.array([sum(d.values()) for d in self.docs], float)
        self.avg = self.lens.mean()
        df = Counter(w for d in self.docs for w in d)
        n = len(self.docs)
        self.idf = {w: math.log(1 + (n - f + 0.5) / (f + 0.5)) for w, f in df.items()}

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(len(self.docs))
        for w in set(tokenize(query)):
            if w not in self.idf:
                continue
            tf = np.array([d.get(w, 0) for d in self.docs], float)
            out += self.idf[w] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * self.lens / self.avg))
        return out


# ------------------------------------------------------------------ dense retrieval
class DenseEmbedder:
    """bge-small-en-v1.5: [CLS] vector, L2-normalised, so a dot product is cosine similarity."""

    def __init__(self, name: str = EMBED_MODEL, device: str | None = None):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch, self.name = torch, name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).to(self.device).eval()

    def encode(self, texts: list[str], query: bool = False, batch_size: int = 64) -> np.ndarray:
        if query:
            texts = [BGE_QUERY_PREFIX + t for t in texts]
        vecs = []
        for i in range(0, len(texts), batch_size):
            enc = self.tok(texts[i:i + batch_size], padding=True, truncation=True, max_length=512,
                           return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                cls = self.model(**enc).last_hidden_state[:, 0]
            vecs.append(self.torch.nn.functional.normalize(cls, dim=-1).float().cpu().numpy())
        return np.concatenate(vecs)


class HashEmbedder:
    """A model-free stand-in for tests: hashed bag of words, L2-normalised."""

    name = "hash"

    def __init__(self, dim: int = 512):
        self.dim = dim

    def encode(self, texts: list[str], query: bool = False, batch_size: int = 0) -> np.ndarray:
        out = np.zeros((len(texts), self.dim))
        for i, t in enumerate(texts):
            for w in tokenize(t):
                out[i, zlib.crc32(w.encode()) % self.dim] += 1
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


# ------------------------------------------------------------------ the index
class Index:
    """Chunks plus whatever each method needs. method: "bm25", "dense" or "hybrid"."""

    def __init__(self, chunks: list[dict], embedder=None, methods=("bm25", "dense")):
        self.chunks = chunks
        texts = [c["text"] for c in chunks]
        self.bm25 = BM25(texts) if "bm25" in methods else None
        self.embedder = embedder
        self.vecs = embedder.encode(texts) if embedder is not None and "dense" in methods else None

    def scores(self, query: str, method: str) -> np.ndarray:
        if method == "bm25":
            return self.bm25.scores(query)
        if method == "dense":
            return self.vecs @ self.embedder.encode([query], query=True)[0]
        if method == "hybrid":   # reciprocal rank fusion of the two rankings (k = 60)
            out = np.zeros(len(self.chunks))
            for m in ("bm25", "dense"):
                order = np.argsort(-self.scores(query, m), kind="stable")
                out[order] += 1 / (60 + np.arange(1, len(order) + 1))
            return out
        raise ValueError(method)

    def search(self, query: str, k: int, method: str = "dense") -> list[dict]:
        s = self.scores(query, method)
        order = np.argsort(-s, kind="stable")[:k]
        return [self.chunks[i] | {"score": float(s[i]), "rank": r + 1} for r, i in enumerate(order)]


def gold_rank(hits: list[dict], q: dict) -> int | None:
    """Rank of the first retrieved chunk that comes from the right document and contains the answer."""
    for h in hits:
        if h["doc_id"] == q["doc_id"] and q["gold_text"] in h["text"]:
            return h["rank"]
    return None


# ------------------------------------------------------------------ prompts
SYSTEM_RAG = ("You answer questions about products using only the context below. Reply with only the answer: "
              "a number or a short phrase, not a sentence. If the context does not contain the answer, reply: unknown")
SYSTEM_CLOSED = ("You answer questions about products. Reply with only the answer: a number or a short phrase, "
                 "not a sentence. If you do not know, reply: unknown")


def messages(question: str, passages: list[str] | None) -> list[dict]:
    """Closed-book when passages is None; otherwise numbered passages, best first, then the question."""
    if passages is None:
        return [{"role": "system", "content": SYSTEM_CLOSED}, {"role": "user", "content": question}]
    ctx = "\n\n".join(f"[{i}] {p}" for i, p in enumerate(passages, 1))
    return [{"role": "system", "content": SYSTEM_RAG},
            {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {question}"}]


# ------------------------------------------------------------------ scoring
NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def parse_answer(text: str, q: dict) -> dict:
    """The predicted value, whether the model abstained, and whether it gave a sibling's value."""
    from rag_corpus import CHOICES
    t = text.strip().lower()
    abstained = t.startswith("unknown") or "not contain" in t or "does not say" in t
    code = q["doc_id"].lower()                                   # "C-200 weighs 3.4 kg": drop the code first
    t_num = t.replace(q["name"].lower(), " ").replace(code, " ")
    if q["kind"] == "number":
        m = NUM.search(t_num)
        pred = float(m.group().replace(",", "")) if m else None
        same = lambda v: pred is not None and abs(pred - float(v)) < 1e-6   # noqa: E731
    else:
        named = [c for c in CHOICES[q["attribute"]] if c.lower() in t]
        pred = named[0] if len(named) == 1 else (None if not named else "|".join(named))
        same = lambda v: pred == v   # noqa: E731
    return dict(pred=pred, abstained=abstained and pred is None, correct=same(q["gold"]),
                sibling=any(same(v) for v in q["sibling_values"]),
                stale=same(q["old_gold"]) if q.get("old_gold", q["gold"]) != q["gold"] else False)
