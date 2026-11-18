"""Week 6 tools, their schemas, and the parser that classifies every model reply.

Two tools:
  lookup_price(item)      price of one unit of a catalogue item (the model can't know it)
  calculate(expression)   exact arithmetic (Week 3's safe parser, never eval)

A reply is one of:
  tool calls    one or more <tool_call>{"name": ..., "arguments": {...}}</tool_call> blocks
  final answer  only the JSON object {"answer": <number>}
or it is malformed, in one of four named ways (MALFORMED). A tool that runs but fails
(unknown item, bad expression) is a tool error, not a malformed reply: the error is
returned to the model as the tool result, as any real tool would.
"""
from __future__ import annotations

import difflib
import json
import re
import sys
from pathlib import Path

import jsonschema

sys.path.insert(0, str(Path(__file__).parent))
from react_agent import calculate as _safe_calc  # noqa: E402
from tool_tasks import CATALOGUE  # noqa: E402

TOOLS_VERSION = "1.0"
MALFORMED = ["json_error", "unknown_tool", "schema_error", "final_format_error"]


class ToolError(Exception):
    """The call was well formed but the tool could not do it."""


# ------------------------------------------------------------------ tool functions
def lookup_price(item: str) -> dict:
    key = item.strip().lower()
    if key not in CATALOGUE:
        close = difflib.get_close_matches(key, CATALOGUE, n=1, cutoff=0.75)
        raise ToolError(f"Unknown item {item!r}." + (f" Did you mean {close[0]!r}?" if close else ""))
    return {"item": key, "price": float(CATALOGUE[key]["price"]), "unit": CATALOGUE[key]["unit"], "currency": "GBP"}


def calculate(expression: str) -> dict:
    out = _safe_calc(expression)
    if out.startswith("Error"):
        raise ToolError(out.removeprefix("Error: "))
    return {"result": float(out)}


FUNCTIONS = {"lookup_price": lookup_price, "calculate": calculate}

# -- direct route: schemas written by hand ------------------------------------
SCHEMAS = [
    {"type": "function", "function": {
        "name": "lookup_price",
        "description": "Look up the price of one unit of a shop item, in pounds (GBP).",
        "parameters": {"type": "object", "additionalProperties": False, "required": ["item"],
                       "properties": {"item": {"type": "string", "description": 'Catalogue item name, e.g. "olive oil"'}}}}},
    {"type": "function", "function": {
        "name": "calculate",
        "description": "Evaluate an arithmetic expression exactly.",
        "parameters": {"type": "object", "additionalProperties": False, "required": ["expression"],
                       "properties": {"expression": {"type": "string",
                                                     "description": 'Numbers and + - * / // ( ) only, e.g. "3 * 2.5 + 4"'}}}}},
]


class DirectTools:
    """Tools called as plain Python functions."""
    route = "direct"

    def __init__(self):
        self.schemas = SCHEMAS

    def call(self, name: str, arguments: dict) -> tuple[str, bool]:
        """Returns (content for the model, is_error)."""
        try:
            return json.dumps(FUNCTIONS[name](**arguments)), False
        except ToolError as e:
            return json.dumps({"error": str(e)}), True

    def close(self):
        pass


# -- parsing model replies ---------------------------------------------------
FINAL_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["answer"],
                "properties": {"answer": {"type": "number"}}}
_CALL = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


def _first_error(instance, schema) -> str | None:
    err = jsonschema.exceptions.best_match(jsonschema.Draft202012Validator(schema).iter_errors(instance))
    if err is None:
        return None
    where = "/".join(map(str, err.absolute_path))
    return f"{where + ': ' if where else ''}{err.message}"


def parse_reply(text: str, schemas: list[dict]) -> dict:
    """Classify a reply. Returns {"kind": "calls"|"final"|"malformed", ...}.
    calls: [{"name", "arguments"}]; final: "answer"; malformed: "type", "detail"."""
    params = {s["function"]["name"]: s["function"]["parameters"] for s in schemas}
    blocks = _CALL.findall(text)
    if "<tool_call>" in text and len(blocks) < text.count("<tool_call>"):
        return dict(kind="malformed", type="json_error", detail="a <tool_call> tag was opened but not closed")
    if blocks:
        calls = []
        for raw in blocks:
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as e:
                return dict(kind="malformed", type="json_error", detail=f"tool call is not valid JSON ({e.msg})")
            if not isinstance(obj, dict) or not isinstance(obj.get("name"), str):
                return dict(kind="malformed", type="json_error", detail='tool call must be {"name": ..., "arguments": {...}}')
            if obj["name"] not in params:
                return dict(kind="malformed", type="unknown_tool",
                            detail=f"no tool named {obj['name']!r}; tools are {', '.join(params)}")
            args = obj.get("arguments")
            if not isinstance(args, dict):
                return dict(kind="malformed", type="schema_error", detail=f"arguments for {obj['name']} must be a JSON object")
            err = _first_error(args, params[obj["name"]])
            if err:
                return dict(kind="malformed", type="schema_error", detail=f"{obj['name']} arguments: {err}")
            calls.append(dict(name=obj["name"], arguments=args))
        return dict(kind="calls", calls=calls)
    body = text.strip()
    m = _FENCE.match(body)
    body = m.group(1) if m else body
    try:
        obj = json.loads(body)
    except json.JSONDecodeError:
        return dict(kind="malformed", type="final_format_error",
                    detail='no tool call, and the reply is not the JSON object {"answer": <number>}')
    err = _first_error(obj, FINAL_SCHEMA)
    if err:
        return dict(kind="malformed", type="final_format_error", detail=f"final answer: {err}")
    return dict(kind="final", answer=float(obj["answer"]))


def repair_message(detail: str) -> str:
    return (f"Your last reply could not be used: {detail}. Reply again with either tool calls in "
            '<tool_call></tool_call> tags, or, if you know the answer, only the JSON object {"answer": <number>}.')
