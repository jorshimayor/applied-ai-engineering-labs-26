"""Week 6: a two-tool calling loop with schema validation and malformed-output repair.

Each turn the model either calls tools or gives a final JSON answer. Every reply is
validated (tools.parse_reply). A malformed reply is not executed: the model is told what
was wrong and asked again, up to `max_repairs` times per problem.

Problems run in step-synchronous batches: all unfinished conversations generate their
next turn together, which keeps a GPU busy.

Two tool backends, same loop:
  DirectTools  plain Python functions, hand-written JSON schemas (tools.py)
  MCPTools     the same functions behind an MCP server over stdio (mcp_server.py),
               with the schemas the server advertises
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from tool_tasks import is_correct  # noqa: E402
from tools import parse_reply, repair_message  # noqa: E402

SYSTEM = ("You are a shopping assistant. You do not know any prices: call lookup_price for every item, "
          "and use calculate for every arithmetic step. Money is in pounds, rounded to 2 decimal places. "
          'When you have the result, reply with only the JSON object {"answer": <number>} and nothing else.')


class MCPTools:
    """MCP client for mcp_server.py. Runs its own event loop on a background thread so the
    synchronous agent loop can call it."""
    route = "mcp"

    def __init__(self, server_script: Path = Path(__file__).parent / "mcp_server.py", errlog=None):
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, daemon=True).start()
        ready = concurrent.futures.Future()
        errlog = errlog or open(os.devnull, "w")   # the server logs every tool failure to stderr

        async def lifecycle():
            # anyio needs the connection opened and closed by the same task, so one task holds it
            self._closed = asyncio.Event()
            try:
                async with stdio_client(StdioServerParameters(command=sys.executable, args=[str(server_script)]),
                                        errlog=errlog) as (read, write):
                    async with ClientSession(read, write) as session:
                        self.session = session
                        await session.initialize()
                        listed = await session.list_tools()
                        ready.set_result([{"type": "function", "function": {
                            "name": t.name, "description": t.description, "parameters": t.input_schema}}
                            for t in listed.tools])
                        await self._closed.wait()
            except Exception as e:
                if not ready.done():
                    ready.set_exception(e)
                raise

        self._task = asyncio.run_coroutine_threadsafe(lifecycle(), self.loop)
        self.schemas = ready.result(timeout=60)

    def call(self, name: str, arguments: dict) -> tuple[str, bool]:
        result = asyncio.run_coroutine_threadsafe(self.session.call_tool(name, arguments), self.loop).result(timeout=60)
        text = "\n".join(c.text for c in result.content if getattr(c, "type", "") == "text")
        return (json.dumps({"error": text}) if result.is_error else text), bool(result.is_error)

    def close(self):
        self.loop.call_soon_threadsafe(self._closed.set)
        self._task.result(timeout=30)
        self.loop.call_soon_threadsafe(self.loop.stop)


def _assistant_turn(text: str, calls: list[dict]) -> dict:
    content = text.split("<tool_call>")[0].strip()
    return {"role": "assistant", "content": content,
            "tool_calls": [{"type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}} for c in calls]}


def run_problems(llm, problems: list[dict], backend, max_turns: int = 8, max_repairs: int = 3,
                 max_new_tokens: int = 256, batch_size: int = 16, on_done=None) -> list[dict]:
    """Run each problem to a final answer, a turn limit or a repair limit. Returns one record per problem."""
    states = [dict(p=p, messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": p["question"]}],
                   events=[], turns=0, repairs=0, tool_calls=0, tool_errors=0, tool_seconds=[], tokens_in=0,
                   tokens_out=0, seconds=0.0, pred=None, status=None) for p in problems]
    pending = list(states)
    while pending:
        batch, pending = pending[:batch_size], pending[batch_size:]
        active = batch
        while active:
            t0 = time.time()
            gens = llm.generate([s["messages"] for s in active], max_new_tokens, tools=backend.schemas)
            dt = (time.time() - t0) / len(active)
            for s, (text, n_in, n_out) in zip(active, gens):
                s["turns"] += 1; s["tokens_in"] += n_in; s["tokens_out"] += n_out; s["seconds"] += dt
                r = parse_reply(text, backend.schemas)
                if r["kind"] == "malformed":
                    s["events"].append(dict(turn=s["turns"], type=r["type"], detail=r["detail"], output=text))
                    if s["repairs"] >= max_repairs:
                        s["status"] = "repairs_exhausted"; continue
                    s["repairs"] += 1
                    s["messages"] += [{"role": "assistant", "content": text},
                                      {"role": "user", "content": repair_message(r["detail"])}]
                elif r["kind"] == "final":
                    s["pred"], s["status"] = r["answer"], "final"
                    s["messages"].append({"role": "assistant", "content": text})
                    continue
                else:
                    s["messages"].append(_assistant_turn(text, r["calls"]))
                    for c in r["calls"]:
                        t1 = time.perf_counter()
                        content, is_error = backend.call(c["name"], c["arguments"])
                        s["tool_seconds"].append(time.perf_counter() - t1)
                        s["tool_calls"] += 1; s["tool_errors"] += is_error
                        s["messages"].append({"role": "tool", "name": c["name"], "content": content})
                if s["turns"] >= max_turns:
                    s["status"] = "max_turns"
            active = [s for s in active if s["status"] is None]
        for s in batch:
            if on_done:
                on_done(record(s, backend.route))
    return [record(s, backend.route) for s in states]


def record(s: dict, route: str) -> dict:
    p, first = s["p"], (s["events"][0]["turn"] if s["events"] else None)
    correct = s["status"] == "final" and is_correct(s["pred"], p["gold"])
    return dict(id=f"{route}|{p['pid']}", route=route, pid=p["pid"], split=p["split"], template=p["template"],
                question=p["question"], gold=p["gold"], pred=s["pred"], status=s["status"], correct=correct,
                # fail-closed: the same run scored as if the first malformed reply had ended it
                correct_fail_closed=correct and not s["events"], first_malformed_turn=first,
                malformed=[{k: e[k] for k in ("turn", "type", "detail")} for e in s["events"]],
                malformed_outputs=[e["output"] for e in s["events"]], repairs=s["repairs"], turns=s["turns"],
                tool_calls=s["tool_calls"], tool_errors=s["tool_errors"], tool_seconds=s["tool_seconds"],
                tokens_in=s["tokens_in"], tokens_out=s["tokens_out"], seconds=s["seconds"], messages=s["messages"])
