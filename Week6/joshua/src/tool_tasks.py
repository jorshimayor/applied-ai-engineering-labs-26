"""Week 6 task set: shopping questions that need two tools.

Every question names items from a fixed catalogue whose prices are random, so the model
cannot answer from memory: it must call `lookup_price` for each item, then `calculate`.
Gold answers are computed exactly with Decimal.

Split (no prompt or parser work ever looks at test):
  dev  : catalogue items 0-7,  seed 1000
  test : catalogue items 8-31, seed 6   (items the prompts never saw during development)
"""
from __future__ import annotations

import random
from decimal import ROUND_FLOOR, ROUND_HALF_UP, Decimal

TASKS_VERSION = "1.0"

# (name, unit). Prices are drawn once from a fixed seed, so they are arbitrary but stable.
_ITEMS = [
    ("basmati rice", "kg"), ("olive oil", "litre"), ("plain flour", "kg"), ("brown eggs", "dozen"),
    ("cheddar cheese", "block"), ("oat milk", "carton"), ("green tea", "box"), ("dark chocolate", "bar"),
    ("cherry tomatoes", "punnet"), ("red onions", "kg"), ("sourdough loaf", "loaf"), ("greek yoghurt", "tub"),
    ("ground coffee", "bag"), ("brown lentils", "bag"), ("frozen peas", "bag"), ("chicken thighs", "kg"),
    ("smoked salmon", "pack"), ("baby spinach", "bag"), ("sweet potatoes", "kg"), ("peanut butter", "jar"),
    ("rolled oats", "bag"), ("coconut milk", "tin"), ("chickpeas", "tin"), ("salted butter", "pack"),
    ("blueberries", "punnet"), ("tomato passata", "carton"), ("basil pesto", "jar"), ("wholemeal pasta", "bag"),
    ("maple syrup", "bottle"), ("mozzarella", "ball"), ("lemons", "bag"), ("brown sugar", "kg"),
]
_rng = random.Random(2026)
CATALOGUE = {name: {"price": Decimal(_rng.randrange(45, 1300)) / 100, "unit": unit} for name, unit in _ITEMS}
DEV_ITEMS, TEST_ITEMS = [n for n, _ in _ITEMS[:8]], [n for n, _ in _ITEMS[8:]]

TEMPLATES = ["two_items", "three_items", "change", "discount", "budget", "difference"]


def _money(x: Decimal) -> Decimal:
    return x.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def _qty(rng: random.Random) -> int:
    return rng.randint(2, 9)


_PLURAL = {"kg": "kg", "dozen": "dozen", "box": "boxes", "loaf": "loaves"}


def _units(unit: str) -> str:
    return _PLURAL.get(unit, unit + "s")


def _phrase(q: int, item: str) -> str:
    unit = CATALOGUE[item]["unit"]
    return f"{q} {_units(unit) if q > 1 else unit} of {item}"


def make_problem(template: str, rng: random.Random, items: list[str], pid: int) -> dict:
    a, b, c = rng.sample(items, 3)
    qa, qb, qc = _qty(rng), _qty(rng), _qty(rng)
    p = {name: CATALOGUE[name]["price"] for name in (a, b, c)}
    if template == "two_items":
        q = f"Ada buys {_phrase(qa, a)} and {_phrase(qb, b)}. How much does she pay in pounds?"
        gold = _money(qa * p[a] + qb * p[b])
    elif template == "three_items":
        q = f"Ben buys {_phrase(qa, a)}, {_phrase(qb, b)} and {_phrase(qc, c)}. What is the total cost in pounds?"
        gold = _money(qa * p[a] + qb * p[b] + qc * p[c])
    elif template == "change":
        cost = qa * p[a] + qb * p[b]
        note = next(n for n in (20, 50, 100, 200) if n > cost)
        q = f"Chloe buys {_phrase(qa, a)} and {_phrase(qb, b)}, and pays with a {note} pound note. How much change does she get in pounds?"
        gold = _money(note - cost)
    elif template == "discount":
        d = rng.choice([10, 15, 20, 25])
        q = f"Dev buys {_phrase(qa, a)} and {_phrase(qb, b)}. The shop takes {d}% off the total. How much does he pay in pounds?"
        gold = _money((qa * p[a] + qb * p[b]) * (100 - d) / 100)
    elif template == "budget":
        budget = rng.choice([20, 30, 40, 50])
        q = f"Ella has {budget} pounds to spend on {a} only. How many whole {_units(CATALOGUE[a]['unit'])} of {a} can she buy?"
        gold = (Decimal(budget) / p[a]).to_integral_value(rounding=ROUND_FLOOR)
    elif template == "difference":
        q = f"How many pounds more do {_phrase(qa, a)} cost than {_phrase(qb, b)}? Give a negative number if they cost less."
        gold = _money(qa * p[a] - qb * p[b])
    else:
        raise ValueError(template)
    return dict(pid=pid, template=template, question=q, gold=float(gold), items=[a, b, c][: 3 if template == "three_items" else 1 if template == "budget" else 2])


def make_problems(split: str = "test", n: int = 200) -> list[dict]:
    """n problems, templates in rotation, fixed seed per split."""
    items, seed = (DEV_ITEMS, 1000) if split == "dev" else (TEST_ITEMS, 6)
    rng = random.Random(seed)
    return [dict(make_problem(TEMPLATES[i % len(TEMPLATES)], rng, items, i), split=split) for i in range(n)]


def is_correct(pred: float | None, gold: float) -> bool:
    return pred is not None and abs(pred - gold) < 0.005 + 1e-9


if __name__ == "__main__":
    for prob in make_problems("dev", 6):
        print(prob["template"], "|", prob["question"], "->", prob["gold"])
