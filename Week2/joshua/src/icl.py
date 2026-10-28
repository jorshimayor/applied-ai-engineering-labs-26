"""Few-shot (in-context learning) evaluation helpers for Week 2.

The core trick: every test query in a configuration shares the same few-shot prefix,
so we run the prefix through the model ONCE, keep its KV cache, and only process each
query's own tokens. That is what makes ~150 configurations feasible on a laptop CPU.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field

import torch
from huggingface_hub import hf_hub_download
from transformers import DynamicCache


# ----------------------------------------------------------------------------- data
@dataclass(frozen=True)
class Example:
    text: str
    label: int  # 0 = negative, 1 = positive


def load_sst2(split: str) -> list[Example]:
    """SST-2 from the SetFit mirror (plain JSONL, no `datasets` dependency)."""
    path = hf_hub_download("SetFit/sst2", f"{split}.jsonl", repo_type="dataset")
    with open(path) as f:
        return [Example(r["text"].strip(), int(r["label"])) for r in map(json.loads, f)]


def sample_demos(pool: list[Example], k: int, seed: int, n_pos: int | None = None) -> list[Example]:
    """k demonstrations; balanced by default (k//2 positive), or exactly n_pos positives."""
    rng = random.Random(seed)
    if n_pos is None:
        n_pos = k // 2 if k % 2 == 0 else rng.randint(0, 1) + k // 2
    pos = [e for e in pool if e.label == 1]
    neg = [e for e in pool if e.label == 0]
    demos = rng.sample(pos, n_pos) + rng.sample(neg, k - n_pos)
    rng.shuffle(demos)
    return demos


# --------------------------------------------------------------------------- format
@dataclass(frozen=True)
class Format:
    name: str
    template: str                       # must contain {x} and {y}; {y} must come last
    labels: tuple[str, str] = ("negative", "positive")
    instruction: str = ""
    sep: str = "\n\n"
    space_in_query: bool = False        # keep "Answer: " in the query, for tokenisers that split " 0" into " " + "0"

    def _head(self) -> str:
        return self.template.split("{y}")[0]

    def demo(self, e: Example, label: int | None = None) -> str:
        return self.template.format(x=e.text, y=self.labels[e.label if label is None else label])

    def query(self, text: str) -> str:
        # Never end a prompt on a space (Week 1 lesson): the space belongs to the label token.
        head = self._head().format(x=text)
        return head if self.space_in_query else head.rstrip(" ")

    def label_strings(self) -> tuple[str, str]:
        lead = " " if self._head().endswith(" ") and not self.space_in_query else ""
        return tuple(lead + l for l in self.labels)

    def prefix(self, demos: list[Example], labels: list[int] | None = None) -> str:
        labels = labels or [None] * len(demos)
        body = self.sep.join(self.demo(d, l) for d, l in zip(demos, labels))
        parts = [p for p in (self.instruction.strip(), body) if p]
        return self.sep.join(parts) + self.sep if parts else ""


# -------------------------------------------------------------------------- scoring
@dataclass
class Scores:
    logp: torch.Tensor          # (N, 2) log-prob of [neg, pos] label tokens, normalised over the FULL vocab
    cf_logp: torch.Tensor       # (2,)  same for the content-free input "N/A" (for calibration)
    prefix_tokens: int
    meta: dict = field(default_factory=dict)

    @property
    def label_mass(self) -> torch.Tensor:          # how much probability lands on a valid label
        return self.logp.exp().sum(-1)

    def preds(self, calibrate: bool = False) -> torch.Tensor:
        lp = self.logp - self.cf_logp if calibrate else self.logp   # Zhao et al. 2021 contextual calibration
        return lp.argmax(-1)


def label_token_ids(tok, fmt: Format) -> list[int]:
    ids = [tok.encode(s)[0] for s in fmt.label_strings()]
    assert ids[0] != ids[1], f"labels {fmt.labels} share a first token"
    return ids


@torch.no_grad()
def score(model, tok, prefix: str, queries: list[str], label_ids: list[int], batch_size: int = 25) -> tuple[torch.Tensor, int]:
    """Log-probs of the label tokens after `prefix + query`, re-using the prefix KV cache."""
    pre_ids = tok(prefix, return_tensors="pt").input_ids if prefix else None
    P = 0 if pre_ids is None else pre_ids.shape[1]
    legacy = model(pre_ids, use_cache=True).past_key_values.to_legacy_cache() if P else None

    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    out = []
    for i in range(0, len(queries), batch_size):
        enc = [tok(q).input_ids for q in queries[i:i + batch_size]]
        B, S = len(enc), max(map(len, enc))
        ids = torch.full((B, S), pad)
        mask = torch.zeros(B, S, dtype=torch.long)
        for j, e in enumerate(enc):                      # right-pad: real tokens never see the pads
            ids[j, :len(e)] = torch.tensor(e)
            mask[j, :len(e)] = 1
        kwargs = {}
        if P:
            kwargs["past_key_values"] = DynamicCache.from_legacy_cache(
                tuple((k.expand(B, -1, -1, -1), v.expand(B, -1, -1, -1)) for k, v in legacy))
            mask = torch.cat([torch.ones(B, P, dtype=torch.long), mask], dim=1)
        logits = model(input_ids=ids, attention_mask=mask, **kwargs).logits
        last = torch.tensor([len(e) - 1 for e in enc])
        logp = torch.log_softmax(logits[torch.arange(B), last].float(), dim=-1)
        out.append(logp[:, label_ids])
    return torch.cat(out), P


def evaluate(model, tok, fmt: Format, demos: list[Example], test: list[Example],
             demo_labels: list[int] | None = None, **meta) -> Scores:
    prefix = fmt.prefix(demos, demo_labels)
    queries = [fmt.query(e.text) for e in test] + [fmt.query("N/A")]   # last one = content-free probe
    logp, P = score(model, tok, prefix, queries, label_token_ids(tok, fmt))
    return Scores(logp[:-1], logp[-1], P, meta)


def summarise(s: Scores, test: list[Example]) -> dict:
    gold = torch.tensor([e.label for e in test])
    pred, pred_cal = s.preds(), s.preds(calibrate=True)
    return {
        **s.meta,
        "acc": (pred == gold).float().mean().item(),
        "acc_cal": (pred_cal == gold).float().mean().item(),
        "pos_rate": pred.float().mean().item(),            # fraction predicted positive (bias indicator)
        "label_mass": s.label_mass.mean().item(),
        "prefix_tokens": s.prefix_tokens,
        "logp": s.logp.tolist(),
        "cf_logp": s.cf_logp.tolist(),
    }
