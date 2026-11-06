"""A minimal, hand-written ReAct agent with one tool (a calculator). Week 3.

ReAct (Yao et al., 2022) interleaves free-text reasoning ("Thought") with tool calls
("Action") whose results are fed back in ("Observation"). The whole agent is this loop:

    trajectory = ""
    repeat:
        text = LLM(prompt + trajectory, stop at "Observation:")   # model writes Thought + Action
        if Action is finish[...]: return answer
        result = tool(Action)                                      # WE run the tool, not the model
        trajectory += text + "Observation: " + result

The stop sequence is what hands control back to our code. Without it the model would
invent its own Observation (try `--no-stop`).

Usage:
    uv run python src/react_agent.py "A truck carries 1,284 boxes of 36 cans. 2,050 cans are damaged. How many good cans?"
"""
from __future__ import annotations

import ast
import operator
import re
import sys
import time
from dataclasses import dataclass, field

# ------------------------------------------------------------------------------ tool
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow,
        ast.USub: operator.neg, ast.UAdd: operator.pos}


def calculate(expression: str) -> str:
    """Safely evaluate arithmetic. Never use eval() on model output: it is untrusted input."""
    expr = re.sub(r"(?<=\d),(?=\d{3})", "", expression).replace("$", "").strip()   # 1,284 -> 1284

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            if isinstance(node.op, ast.Pow) and abs(ev(node.right)) > 100:
                raise ValueError("exponent too large")
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        raise ValueError(f"unsupported syntax: {ast.dump(node)[:40]}")

    try:
        value = ev(ast.parse(expr, mode="eval"))
    except ZeroDivisionError:
        return "Error: division by zero"
    except Exception as e:  # syntax errors, names, function calls...
        return f"Error: could not evaluate {expression!r} ({type(e).__name__}). Use only numbers and + - * / ( )."
    return str(int(value)) if float(value).is_integer() else f"{value:.6g}"


# ---------------------------------------------------------------------------- prompts
TOOL_DOC = """Available actions:
  calculate[expression]  evaluates an arithmetic expression, e.g. calculate[12 * 7 + 3]
  finish[answer]         returns the final numeric answer and ends the task"""

EXEMPLARS = {
    "react": """Question: A shop sells 125 pens per day for 48 days. Customers return 300 pens. How many pens did the shop sell in the end?
Thought: The shop sells 125 pens a day for 48 days, so first I need 125 * 48.
Action: calculate[125 * 48]
Observation: 6000
Thought: 300 pens were returned, so I subtract 300 from 6000.
Action: calculate[6000 - 300]
Observation: 5700
Thought: The shop sold 5700 pens in the end.
Action: finish[5700]

Question: A factory packs 2,340 bolts into bags of 45 bolts. Each bag sells for $7. How much money does the factory make?
Thought: First I find the number of bags: 2340 / 45.
Action: calculate[2340 / 45]
Observation: 52
Thought: There are 52 bags at $7 each, so the money is 52 * 7.
Action: calculate[52 * 7]
Observation: 364
Thought: The factory makes $364.
Action: finish[364]""",

    "act": """Question: A shop sells 125 pens per day for 48 days. Customers return 300 pens. How many pens did the shop sell in the end?
Action: calculate[125 * 48]
Observation: 6000
Action: calculate[6000 - 300]
Observation: 5700
Action: finish[5700]

Question: A factory packs 2,340 bolts into bags of 45 bolts. Each bag sells for $7. How much money does the factory make?
Action: calculate[2340 / 45]
Observation: 52
Action: calculate[52 * 7]
Observation: 364
Action: finish[364]""",

    "cot": """Question: A shop sells 125 pens per day for 48 days. Customers return 300 pens. How many pens did the shop sell in the end?
Thought: The shop sells 125 * 48 = 6000 pens. 300 are returned, so 6000 - 300 = 5700.
Answer: 5700

Question: A factory packs 2,340 bolts into bags of 45 bolts. Each bag sells for $7. How much money does the factory make?
Thought: There are 2340 / 45 = 52 bags. At $7 each that is 52 * 7 = 364.
Answer: 364""",

    "direct": """Question: A shop sells 125 pens per day for 48 days. Customers return 300 pens. How many pens did the shop sell in the end?
Answer: 5700

Question: A factory packs 2,340 bolts into bags of 45 bolts. Each bag sells for $7. How much money does the factory make?
Answer: 364""",
}

INSTRUCTIONS = {
    "react": f"Solve the math word problem by interleaving Thought, Action and Observation steps. "
             f"Thought reasons about what to do next. Action calls a tool. Observation is the tool result, "
             f"which the system provides.\n{TOOL_DOC}",
    "act": f"Solve the math word problem by taking Actions. Each Observation is the tool result, "
           f"which the system provides.\n{TOOL_DOC}",
    "cot": "Solve the math word problem. Think step by step in a Thought, then give the final number after 'Answer:'.",
    "direct": "Solve the math word problem. Reply only with 'Answer:' followed by the final number.",
}


def system_prompt(mode: str) -> str:
    return f"{INSTRUCTIONS[mode]}\n\nHere are examples:\n\n{EXEMPLARS[mode]}"


# ------------------------------------------------------------------------------- LLM
class LocalLLM:
    """Greedy completion from a local Hugging Face chat model, with stop strings."""

    def __init__(self, name: str = "Qwen/Qwen2.5-0.5B-Instruct", device: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch, self.name = torch, name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == "cuda":   # half precision on GPU: bf16 where supported (A100, L4), fp16 otherwise (T4)
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype).to(self.device).eval()
        self.tokens_in = self.tokens_out = 0

    def __call__(self, system: str, user: str, partial: str, stop: list[str] | None, max_new_tokens: int = 160) -> str:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        # The assistant turn is left OPEN and pre-filled with the trajectory so far: the model continues it.
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + partial
        enc = self.tok(text, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                      stop_strings=stop or None, tokenizer=self.tok,
                                      pad_token_id=self.tok.eos_token_id)
        new = out[0, enc.input_ids.shape[1]:]
        self.tokens_in += enc.input_ids.shape[1]; self.tokens_out += len(new)
        return self.tok.decode(new, skip_special_tokens=True)


# ------------------------------------------------------------------------------ agent
ACTION_RE = re.compile(r"Action:\s*(\w+)\[(.*)\]")
NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")


@dataclass
class Result:
    mode: str
    question: str
    answer: str | None
    trajectory: str
    steps: int = 0
    tool_calls: int = 0
    status: str = "ok"            # ok | max_steps | invalid_action | no_answer
    observations: list[str] = field(default_factory=list)
    seconds: float = 0.0


def to_number(s: str | None) -> float | None:
    m = NUMBER_RE.search(s or "")
    try:
        return float(m.group().replace(",", "").rstrip(".")) if m else None
    except ValueError:
        return None


def run_agent(llm, question: str, mode: str = "react", max_steps: int = 6, use_stop: bool = True,
              verbose: bool = False) -> Result:
    t0 = time.time()
    system, user = system_prompt(mode), f"Question: {question}"

    if mode in ("direct", "cot"):                                     # single call, no tools
        out = llm(system, user, "", stop=["\nQuestion:"], max_new_tokens=20 if mode == "direct" else 200)
        m = re.search(r"Answer:\s*(.+)", out)
        return Result(mode, question, m.group(1).strip() if m else None, out, 1, 0,
                      "ok" if m else "no_answer", seconds=time.time() - t0)

    trajectory, res = "", Result(mode, question, None, "")
    for step in range(1, max_steps + 1):
        res.steps = step
        out = llm(system, user, trajectory, stop=["Observation:"] if use_stop else None)
        if not use_stop:                                             # demo: show what the model invents on its own
            if verbose:
                print(out)
            res.trajectory, res.status, res.seconds = out, "no_stop_demo", time.time() - t0
            return res
        out = out.split("Observation:")[0].rstrip() + "\n"            # drop any self-written observation
        trajectory += out
        if verbose:
            print(out, end="")
        act = ACTION_RE.findall(out)
        if not act:
            res.status = "invalid_action"
            obs = "Invalid action. Use calculate[expression] or finish[answer]."
        else:
            name, arg = act[-1]
            if name == "finish":
                res.answer, res.status = arg.strip(), "ok"
                break
            if name == "calculate":
                obs = calculate(arg); res.tool_calls += 1
            else:
                obs = f"Unknown action {name!r}. Use calculate[expression] or finish[answer]."
        res.observations.append(obs)
        trajectory += f"Observation: {obs}\n"
        if verbose:
            print(f"Observation: {obs}")
    else:
        res.status = "max_steps"
    if res.answer is not None:
        res.status = "ok"
    res.trajectory, res.seconds = trajectory, time.time() - t0
    return res


if __name__ == "__main__":
    q = " ".join(a for a in sys.argv[1:] if not a.startswith("--")) or \
        "A truck carries 1,284 boxes with 36 cans in each box. 2,050 cans are damaged. How many good cans are there?"
    llm = LocalLLM()
    print(f"Question: {q}\n")
    r = run_agent(llm, q, mode="react", use_stop="--no-stop" not in sys.argv, verbose=True)
    print(f"\n=> answer={r.answer}  status={r.status}  steps={r.steps}  tool_calls={r.tool_calls}  ({r.seconds:.0f}s)")
