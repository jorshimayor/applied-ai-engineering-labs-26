"""Week 6 checks that need no model: the reply parser on known-bad outputs, and the agent
loop driven by a scripted fake model over both tool routes.

    uv run python tests/test_week6.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from tool_agent import MCPTools, run_problems  # noqa: E402
from tool_tasks import CATALOGUE, make_problems  # noqa: E402
from tools import SCHEMAS, DirectTools, parse_reply  # noqa: E402

CALL = '<tool_call>\n{{"name": "{}", "arguments": {}}}\n</tool_call>'

PARSER_CASES = [   # (reply, expected kind, expected malformed type)
    (CALL.format("lookup_price", '{"item": "olive oil"}'), "calls", None),
    (CALL.format("lookup_price", '{"item": "a"}') + CALL.format("calculate", '{"expression": "1+1"}'), "calls", None),
    ('{"answer": 12.5}', "final", None),
    ('```json\n{"answer": 3}\n```', "final", None),
    ('<tool_call>\n{"name": "calculate", "arguments": {"expression": "1+1"}}', "malformed", "json_error"),
    (CALL.format("calculate", '{"expression": 1+1}'), "malformed", "json_error"),
    ('<tool_call>["calculate"]</tool_call>', "malformed", "json_error"),
    (CALL.format("add", '{"a": 1, "b": 2}'), "malformed", "unknown_tool"),
    (CALL.format("calculate", "{}"), "malformed", "schema_error"),
    (CALL.format("calculate", '{"expression": 5}'), "malformed", "schema_error"),
    (CALL.format("lookup_price", '{"item": "x", "qty": 2}'), "malformed", "schema_error"),
    (CALL.format("calculate", '"{\\"expression\\": \\"1+1\\"}"'), "malformed", "schema_error"),
    ("The total is 12.50 pounds.", "malformed", "final_format_error"),
    ('{"answer": "12.50"}', "malformed", "final_format_error"),
    ('{"answer": 12.5, "currency": "GBP"}', "malformed", "final_format_error"),
    ('{"answer": true}', "malformed", "final_format_error"),
    ("", "malformed", "final_format_error"),
]


def test_parser():
    for reply, kind, mtype in PARSER_CASES:
        r = parse_reply(reply, SCHEMAS)
        assert r["kind"] == kind and r.get("type") == mtype, (reply, r)
    print(f"parser: {len(PARSER_CASES)} cases classified correctly")


class ScriptedLLM:
    """Plays back a fixed list of replies per problem, in order, ignoring the prompt."""
    device = "scripted"

    def __init__(self, scripts: dict[str, list[str]]):
        self.scripts = {q: list(v) for q, v in scripts.items()}

    def generate(self, conversations, max_new_tokens, tools=None):
        assert tools, "tool schemas must reach the chat template"
        return [(self.scripts[c[1]["content"]].pop(0), 100, 20) for c in conversations]


def scripts_for(problems):
    """Problem 0: clean run. 1: a malformed call, then repaired. 2: four malformed replies (gives up).
    3: unknown item (tool error), then corrected. 4: prose final answer, then valid JSON."""
    out = {}
    for i, p in enumerate(problems):
        a, b = p["items"][0], p["items"][-1]
        good = [CALL.format("lookup_price", json.dumps({"item": a})) + CALL.format("lookup_price", json.dumps({"item": b})),
                CALL.format("calculate", json.dumps({"expression": "1+1"})),
                json.dumps({"answer": p["gold"]})]
        bad = CALL.format("calculate", "{}")
        out[p["question"]] = {0: good,
                              1: [bad] + good,
                              2: [bad, bad, bad, bad],
                              3: [CALL.format("lookup_price", json.dumps({"item": a + "s"}))] + good,
                              4: good[:2] + ["The answer is 5.", json.dumps({"answer": p["gold"]})]}[i]
    return out


def test_loop(backend):
    problems = make_problems("dev", 5)
    recs = {r["pid"]: r for r in run_problems(ScriptedLLM(scripts_for(problems)), problems, backend, batch_size=3)}
    r = recs[0]; assert r["correct"] and r["correct_fail_closed"] and r["tool_calls"] == 3 and not r["malformed"], r
    r = recs[1]; assert r["correct"] and not r["correct_fail_closed"] and r["repairs"] == 1 and r["first_malformed_turn"] == 1, r
    assert r["malformed"][0]["type"] == "schema_error"
    r = recs[2]; assert not r["correct"] and r["status"] == "repairs_exhausted" and r["repairs"] == 3, r
    r = recs[3]; assert r["correct"] and r["correct_fail_closed"] and r["tool_errors"] == 1, r
    assert "Did you mean" in r["messages"][3]["content"], r["messages"][3]
    r = recs[4]; assert r["correct"] and not r["correct_fail_closed"] and r["malformed"][0]["type"] == "final_format_error", r
    lookups = [m for m in recs[0]["messages"] if m["role"] == "tool" and m["name"] == "lookup_price"]
    assert json.loads(lookups[0]["content"])["price"] == float(CATALOGUE[problems[0]["items"][0]]["price"])
    print(f"loop ({backend.route}): clean, repaired, exhausted, tool error and final-format cases all behave")


if __name__ == "__main__":
    test_parser()
    test_loop(DirectTools())
    mcp = MCPTools()
    try:
        test_loop(mcp)
    finally:
        mcp.close()
    print("all Week 6 checks passed")
