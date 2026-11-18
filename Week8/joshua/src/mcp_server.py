"""Week 6: the same two tools behind an MCP server (stdio transport).

The server reuses the tool functions from tools.py unchanged. Its input schemas are not
written by hand: the MCP SDK generates them from the type hints and docstrings below.
A tool failure must be re-raised as the SDK's ToolError, or the client only sees
"Error executing tool <name>" and the model never learns what went wrong.

    python src/mcp_server.py        # speaks MCP over stdin/stdout; a client starts it
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

from pydantic import Field

sys.path.insert(0, str(Path(__file__).parent))
import tools  # noqa: E402
from mcp.server.mcpserver import MCPServer  # noqa: E402
from mcp.server.mcpserver.exceptions import ToolError  # noqa: E402

server = MCPServer("week6-shop")


@server.tool()
def lookup_price(item: Annotated[str, Field(description='Catalogue item name, e.g. "olive oil"')]) -> dict:
    """Look up the price of one unit of a shop item, in pounds (GBP)."""
    try:
        return tools.lookup_price(item)
    except tools.ToolError as e:
        raise ToolError(str(e)) from e


@server.tool()
def calculate(expression: Annotated[str, Field(description='Numbers and + - * / // ( ) only, e.g. "3 * 2.5 + 4"')]) -> dict:
    """Evaluate an arithmetic expression exactly."""
    try:
        return tools.calculate(expression)
    except tools.ToolError as e:
        raise ToolError(str(e)) from e


if __name__ == "__main__":
    server.run("stdio")
