"""GSM8K test set (Cobbe et al., 2021; MIT licence), downloaded on first use.

The dataset is not stored in this repository. `sample(400)` returns the same
400 test problems everywhere (seeded); `dev(40)` returns train-split problems
for developing prompts and parsers, so the test set is never tuned on.
"""
from __future__ import annotations

import json
import random
import re
import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/{split}.jsonl"
CACHE = Path.home() / ".cache" / "gsm8k"


def load_split(split: str) -> list[dict]:
    path = CACHE / f"{split}.jsonl"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL.format(split=split), path)
    rows = []
    for i, line in enumerate(path.read_text().splitlines()):
        r = json.loads(line)
        gold = r["answer"].split("####")[-1].strip().replace(",", "")
        rows.append(dict(id=i, question=r["question"], gold=float(gold)))
    return rows


def load_test() -> list[dict]:
    return load_split("test")


def sample(n: int = 400, seed: int = 5) -> list[dict]:
    """A fixed random subset of the 1,319 test problems, in a fixed order. Run once, not tuned on."""
    rows = load_test()
    return sorted(random.Random(seed).sample(rows, n), key=lambda r: r["id"])


def dev(n: int = 40, seed: int = 5) -> list[dict]:
    """Development problems from the train split: free to inspect while fixing prompts and parsers."""
    rows = load_split("train")
    return sorted(random.Random(seed).sample(rows, n), key=lambda r: r["id"])


NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")


def to_float(text: str | None) -> float | None:
    if text is None:
        return None
    m = NUMBER.search(str(text).replace("$", ""))
    if not m:
        return None
    try:
        return float(m.group().replace(",", "").rstrip("."))
    except ValueError:
        return None


def is_correct(pred: float | None, gold: float) -> bool:
    return pred is not None and abs(pred - gold) < 1e-6 * max(1.0, abs(gold))
