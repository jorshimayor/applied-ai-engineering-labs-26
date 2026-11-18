"""Week 7: a small knowledge base of fictional products, and questions about it.

Every brand, town and number is made up, so a model cannot answer from memory: the
answer has to come from the documents. Siblings in a product family share most of their
wording and differ in the numbers, which is what makes retrieval non-trivial.

    from rag_corpus import build_corpus, make_questions
    docs = build_corpus()                 # 60 documents, edition 1
    qs = make_questions("test", 200)      # questions about the 48 test products
    docs2 = build_corpus(edition=2)       # the "updated" knowledge base for the index swap
"""
from __future__ import annotations

import random

CORPUS_VERSION = "1.0"

#            brand        letter  product type          what it is for
FAMILIES = [("Aldric", "A", "water pump", "garden irrigation"),
            ("Brenna", "B", "electric bike", "city commuting"),
            ("Corvane", "C", "camera drone", "aerial photography"),
            ("Dunmore", "D", "espresso machine", "small cafes"),
            ("Elwick", "E", "robot mower", "large lawns"),
            ("Farrow", "F", "air purifier", "open-plan offices"),
            ("Galen", "G", "power bank", "long trips off the grid"),
            ("Hesketh", "H", "soldering station", "electronics repair"),
            ("Ilmore", "I", "telescope", "amateur astronomy"),
            ("Jessop", "J", "sewing machine", "heavy fabrics"),
            ("Kirrin", "K", "pressure washer", "patios and driveways"),
            ("Lowther", "L", "3D printer", "classroom workshops"),
            ("Mardale", "M", "dehumidifier", "damp basements"),
            ("Netherby", "N", "label printer", "small warehouses"),
            ("Ospry", "O", "kayak motor", "lakes and slow rivers")]
MODEL_NUMBERS = [100, 200, 300, 400]
DEV_FAMILIES = {"Hesketh", "Mardale", "Ospry"}   # dev questions only ever ask about these

TOWNS = ["Ashby Marsh", "Brackenford", "Coldwater Edge", "Dunholt", "Eastreach", "Fennick",
         "Greywell", "Hollinby", "Irestead", "Kettleby Cross", "Larkmoor", "Morrowfield"]
COLOURS = ["slate grey", "moss green", "signal red", "ivory", "navy blue", "mustard yellow",
           "copper", "matte black", "sky blue", "burnt orange"]
WARRANTIES = [6, 12, 18, 24, 36, 48, 60]

# attribute -> (question template, kind). "number" answers are compared as numbers; "choice"
# answers must name the right option and no other option from the same list.
ATTRIBUTES = {
    "year": ("In what year did the {name} first come out?", "number"),
    "town": ("In which town is the {name} assembled?", "choice"),
    "weight": ("How much does the {name} weigh, in kilograms?", "number"),
    "watts": ("What is the power rating of the {name} in watts?", "number"),
    "colour": ("What colour is the {name} sold in?", "choice"),
    "price": ("What is the list price of the {name} in euros?", "number"),
    "warranty": ("How many months of warranty come with the {name}?", "number"),
}
CHOICES = {"town": TOWNS, "colour": COLOURS}
CHANGED_IN_EDITION_2 = ["weight", "watts", "colour", "price", "warranty"]   # year and town stay


def _distinct(rng: random.Random, pool, n: int) -> list:
    """n values for one family, all different when the pool allows it."""
    pool = list(pool)
    return rng.sample(pool, n) if len(pool) >= n else [rng.choice(pool) for _ in range(n)]


def _family_values(rng: random.Random) -> list[dict]:
    n = len(MODEL_NUMBERS)
    cols = dict(
        year=_distinct(rng, range(2014, 2026), n),
        town=_distinct(rng, TOWNS, n),
        weight=[w / 10 for w in _distinct(rng, range(8, 400), n)],
        watts=[w * 5 for w in _distinct(rng, range(4, 400), n)],
        colour=_distinct(rng, COLOURS, n),
        price=[p * 10 - 1 for p in _distinct(rng, range(5, 400), n)],
        warranty=_distinct(rng, WARRANTIES, n),
    )
    return [{a: cols[a][i] for a in ATTRIBUTES} for i in range(n)]


def _draw(rng: random.Random, attr: str):
    return {"weight": lambda: rng.randrange(8, 400) / 10, "watts": lambda: rng.randrange(4, 400) * 5,
            "colour": lambda: rng.choice(COLOURS), "price": lambda: rng.randrange(5, 400) * 10 - 1,
            "warranty": lambda: rng.choice(WARRANTIES)}[attr]()


def value_text(attr: str, v) -> str:
    """How a value is written in the documents (used to find the answer-bearing chunk)."""
    if attr == "weight":
        return f"{v:.1f} kg"
    if attr == "watts":
        return f"{v} W"
    if attr == "price":
        return f"€{v:,}"
    if attr == "warranty":
        return f"{v}-month"
    return str(v)


def _text(brand: str, letter: str, num: int, ptype: str, use: str, v: dict) -> str:
    name, code = f"{brand} {letter}-{num}", f"{letter}-{num}"
    return "\n\n".join([
        f"The {name} is a {ptype} made by {brand}. It was first released in {v['year']} and is assembled at the "
        f"{brand} works in {v['town']}. Like the rest of the {brand} {letter}-series, it is designed for {use} "
        f"and ships with a printed quick-start guide and a two-year supply of consumables where these apply.",
        f"{brand} builds every {letter}-series {ptype} around a modular frame, so worn parts can be replaced at "
        f"home with ordinary tools. The {code} weighs {value_text('weight', v['weight'])} and is rated at "
        f"{value_text('watts', v['watts'])}. It comes in {v['colour']} only, and the finish is tested against "
        f"scratches and fading in direct sunlight.",
        f"Spare parts and service manuals for the {letter}-series are published on the {brand} support pages. "
        f"{brand} sells the {code} at a list price of {value_text('price', v['price'])} and covers it with a "
        f"{value_text('warranty', v['warranty'])} warranty. Owners can register the product online within 30 days "
        f"of purchase to receive firmware updates and recall notices.",
    ])


def build_corpus(edition: int = 1, seed: int = 7) -> list[dict]:
    """One document per product. Edition 2 changes CHANGED_IN_EDITION_2 for every product."""
    rng = random.Random(seed)
    docs = []
    for brand, letter, ptype, use in FAMILIES:
        values = _family_values(rng)
        if edition == 2:   # every changed value really changes, and siblings stay distinct
            rng2 = random.Random(f"{seed}-{brand}-edition-2")
            for a in CHANGED_IN_EDITION_2:
                used = set()
                for v in values:
                    nv = _draw(rng2, a)
                    while nv == v[a] or nv in used:
                        nv = _draw(rng2, a)
                    used.add(nv)
                    v[a] = nv
        for num, v in zip(MODEL_NUMBERS, values):
            docs.append(dict(doc_id=f"{letter}-{num}", name=f"{brand} {letter}-{num}", brand=brand, letter=letter,
                             family=brand, product_type=ptype, edition=edition, values=v,
                             split="dev" if brand in DEV_FAMILIES else "test",
                             title=f"{brand} {letter}-{num} {ptype}", text=_text(brand, letter, num, ptype, use, v)))
    return docs


def make_questions(split: str = "test", n: int = 200, seed: int = 11, docs: list[dict] | None = None) -> list[dict]:
    """n (product, attribute) questions from one split, without repeats, in a fixed order."""
    docs = docs or build_corpus()
    pairs = [(d, a) for d in docs if d["split"] == split for a in ATTRIBUTES]
    rng = random.Random(seed)
    rng.shuffle(pairs)
    if n > len(pairs):
        raise ValueError(f"only {len(pairs)} {split} questions exist")
    out = []
    for i, (d, a) in enumerate(pairs[:n]):
        template, kind = ATTRIBUTES[a]
        siblings = [s["values"][a] for s in docs if s["family"] == d["family"] and s["doc_id"] != d["doc_id"]]
        out.append(dict(qid=i, split=split, doc_id=d["doc_id"], name=d["name"], attribute=a, kind=kind,
                        question=template.format(name=d["name"]), gold=d["values"][a],
                        gold_text=value_text(a, d["values"][a]), sibling_values=siblings))
    return out


def regold(questions: list[dict], docs: list[dict]) -> list[dict]:
    """The same questions, with gold answers taken from another edition of the corpus."""
    by_id = {d["doc_id"]: d for d in docs}
    out = []
    for q in questions:
        d = by_id[q["doc_id"]]
        siblings = [s["values"][q["attribute"]] for s in docs if s["family"] == d["family"] and s["doc_id"] != d["doc_id"]]
        out.append(q | dict(gold=d["values"][q["attribute"]], gold_text=value_text(q["attribute"], d["values"][q["attribute"]]),
                            sibling_values=siblings, old_gold=q["gold"]))
    return out
