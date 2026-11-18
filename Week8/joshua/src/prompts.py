"""Week 5 prompt library: five prompting strategies for GSM8K-style maths.

Each strategy is a template (system prompt, optional worked examples, cue) plus
a parser. Parsing is reported two ways:
  strict  : the answer must appear in the requested format
  lenient : the last number in the output, whatever the format
The gap between them is the cost of format failures, separate from reasoning.

The 8 worked examples are original, written in the style of Wei et al. (2022).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

LIBRARY_VERSION = "1.1"   # 1.1: LaTeX-tolerant answer line, 512-token reasoning budget (fixed on the dev split)

# (question, step-by-step rationale, answer)
EXAMPLES = [
    ("A baker makes 24 muffins in the morning and 36 in the afternoon. She sells 45 muffins. How many muffins are left?",
     "She makes 24 + 36 = 60 muffins. After selling 45, she has 60 - 45 = 15 left.", 15),
    ("Tom has 3 boxes of pencils with 12 pencils in each box. He gives 7 pencils to his sister. How many pencils does Tom have now?",
     "Tom has 3 * 12 = 36 pencils. After giving away 7, he has 36 - 7 = 29.", 29),
    ("A train ticket costs $18 for adults and $9 for children. How much do 2 adults and 3 children pay in total?",
     "The adults pay 2 * 18 = 36 dollars. The children pay 3 * 9 = 27 dollars. Together that is 36 + 27 = 63 dollars.", 63),
    ("Maria reads 15 pages a day. Her book has 240 pages. After 10 days, how many pages does she still have to read?",
     "In 10 days she reads 15 * 10 = 150 pages. She still has 240 - 150 = 90 pages to read.", 90),
    ("A farmer has 5 rows of apple trees with 8 trees in each row. Each tree gives 20 apples. How many apples does the farmer get?",
     "There are 5 * 8 = 40 trees. They give 40 * 20 = 800 apples.", 800),
    ("A shop sells notebooks for $4 each. Leo buys 6 notebooks and pays with a $50 bill. How much change does he get?",
     "The notebooks cost 6 * 4 = 24 dollars. His change is 50 - 24 = 26 dollars.", 26),
    ("A class of 28 students is split into groups of 4. Each group needs 3 sheets of paper. How many sheets of paper are needed?",
     "There are 28 / 4 = 7 groups. They need 7 * 3 = 21 sheets.", 21),
    ("Sam earns $12 an hour. He works 5 hours on Saturday and 3 hours on Sunday. How much does he earn over the weekend?",
     "He works 5 + 3 = 8 hours. He earns 8 * 12 = 96 dollars.", 96),
]

SYSTEM = {
    "direct": "Solve the maths word problem. Reply with one line only, in the form 'The answer is N.', where N is a number. Do not show any working.",
    "cot": "Solve the maths word problem. Work through it step by step, then finish with a final line in the form 'The answer is N.', where N is a number.",
    "json": 'Solve the maths word problem. Reply with a JSON object only, in the form {"reasoning": "<step-by-step working>", "answer": <number>}.',
}

# "The answer is N." Also accepts LaTeX wrappers small models add: "the answer is \\( \\boxed{9} \\)", "\\( N = 530 \\)"
ANSWER_LINE = re.compile(r"answer is\s*:?\s*(?:\\[(\[]\s*)?(?:\\boxed\{\s*)?(?:[A-Za-z]\s*=\s*)?\$?\s*(-?\d[\d,]*(?:\.\d+)?)", re.IGNORECASE)
LAST_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _num(s: str) -> float | None:
    try:
        return float(s.replace(",", ""))
    except (ValueError, AttributeError):
        return None


def lenient_parse(text: str) -> float | None:
    nums = LAST_NUMBER.findall(text.replace("$", ""))
    return _num(nums[-1]) if nums else None


@dataclass(frozen=True)
class Strategy:
    name: str
    description: str
    style: str               # direct | cot | json
    few_shot: bool
    cue: str = ""            # appended to the question (e.g. "Let's think step by step.")
    max_new_tokens: int = 512           # same budget for every reasoning strategy

    def _answer_text(self, rationale: str, answer: int) -> str:
        if self.style == "direct":
            return f"The answer is {answer}."
        if self.style == "cot":
            return f"{rationale}\nThe answer is {answer}."
        return json.dumps({"reasoning": rationale, "answer": answer})

    def messages(self, question: str) -> list[dict]:
        msgs = [{"role": "system", "content": SYSTEM[self.style]}]
        if self.few_shot:
            for q, rationale, a in EXAMPLES:
                msgs.append({"role": "user", "content": q})
                msgs.append({"role": "assistant", "content": self._answer_text(rationale, a)})
        msgs.append({"role": "user", "content": question + (f"\n{self.cue}" if self.cue else "")})
        return msgs

    def parse(self, text: str) -> dict:
        """Returns strict and lenient predictions and whether the format was followed."""
        strict = None
        if self.style == "json":
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m:
                try:
                    obj = json.loads(m.group())
                    if isinstance(obj, dict) and isinstance(obj.get("answer"), (int, float)) and not isinstance(obj["answer"], bool):
                        strict = float(obj["answer"])
                except json.JSONDecodeError:
                    pass
        else:
            hits = ANSWER_LINE.findall(text)
            strict = _num(hits[-1]) if hits else None
        return {"pred_strict": strict, "pred_lenient": strict if strict is not None else lenient_parse(text),
                "format_ok": strict is not None}


STRATEGIES: dict[str, Strategy] = {s.name: s for s in [
    Strategy("zero_shot_direct", "Instruction only; answer without working", "direct", False, max_new_tokens=24),
    Strategy("few_shot_direct", "8 worked examples showing only the answer (Wei et al.'s 'standard prompting')", "direct", True, max_new_tokens=24),
    Strategy("zero_shot_cot", "Instruction plus 'Let's think step by step.' (Kojima et al., 2022)", "cot", False, cue="Let's think step by step."),
    Strategy("few_shot_cot", "8 worked examples with step-by-step reasoning (Wei et al., 2022)", "cot", True),
    Strategy("few_shot_cot_json", "Same reasoning examples, but output must be JSON with 'reasoning' and 'answer'", "json", True),
]}
BASELINE = "few_shot_direct"
