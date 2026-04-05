#!/usr/bin/env python3
"""
mcp_client.py — Custom interactive MCP client for the GEO pipeline server.

Modes
-----
Interactive REPL (default):
    python mcp_client.py

One-shot commands:
    python mcp_client.py tools                          list all tools + schemas
    python mcp_client.py resources                      list all resources
    python mcp_client.py call <tool> [key=value ...]    call a tool directly
    python mcp_client.py resource <uri>                 read a resource

Examples:
    python mcp_client.py call run_pipeline query="Restaurants in Sfax"
    python mcp_client.py call run_pipeline query="Cafes in Tunis" top_n_brands=5
    python mcp_client.py call generate_prompts domain="Egyptian restaurants" languages=fr,ar
    python mcp_client.py call scrape_brands brands="Dar El Jeld,Le Golfe"
    python mcp_client.py call get_results
    python mcp_client.py call list_output_files
    python mcp_client.py resource geo://overview
    python mcp_client.py resource geo://models
    python mcp_client.py resource geo://last-run
"""

from __future__ import annotations

import asyncio
import json
import sys
import textwrap
from pathlib import Path
from typing import Any

# Force UTF-8 output on Windows (avoids cp1252 encoding errors)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

# ── MCP imports ──────────────────────────────────────────────────────────────
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
except ImportError:
    sys.exit("ERROR: 'mcp' package not found. Run: pip install mcp>=1.0.0")

# ── Config ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
SERVER_SCRIPT = str(ROOT / "mcp_server.py")

# Use whichever Python interpreter launched this client (inherits Anaconda env)
PYTHON = sys.executable

SEPARATOR = "─" * 60


# ══════════════════════════════════════════════════════════════════════════════
# Formatting helpers
# ══════════════════════════════════════════════════════════════════════════════

def _header(text: str) -> str:
    return f"\n\033[1;36m{text}\033[0m\n{SEPARATOR}"


def _ok(text: str) -> str:
    return f"\033[32m{text}\033[0m"


def _err(text: str) -> str:
    return f"\033[31m{text}\033[0m"


def _dim(text: str) -> str:
    return f"\033[2m{text}\033[0m"


def _bold(text: str) -> str:
    return f"\033[1m{text}\033[0m"


def _pretty_json(data: str | dict | list) -> str:
    """Pretty-print JSON with syntax-like coloring (best-effort)."""
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            return data
    return json.dumps(data, indent=2, ensure_ascii=False, default=str)


def _print_result(content_blocks: list) -> None:
    """Print tool result content blocks."""
    for block in content_blocks:
        if hasattr(block, "text"):
            text = block.text
            try:
                parsed = json.loads(text)
                print(_pretty_json(parsed))
            except json.JSONDecodeError:
                print(text)
        else:
            print(str(block))


# ══════════════════════════════════════════════════════════════════════════════
# Core client session factory
# ══════════════════════════════════════════════════════════════════════════════

def _server_params() -> StdioServerParameters:
    return StdioServerParameters(
        command=PYTHON,
        args=[SERVER_SCRIPT],
    )


# ══════════════════════════════════════════════════════════════════════════════
# One-shot command handlers
# ══════════════════════════════════════════════════════════════════════════════

async def cmd_list_tools() -> None:
    async with stdio_client(_server_params()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.list_tools()

    print(_header("Available Tools"))
    for i, tool in enumerate(result.tools, 1):
        print(f"\n  {_bold(f'{i}. {tool.name}')}")
        if tool.description:
            for line in textwrap.wrap(tool.description, width=70):
                print(f"     {_dim(line)}")
        schema = tool.inputSchema or {}
        props = schema.get("properties", {})
        required = schema.get("required", [])
        if props:
            print(f"     {_dim('Parameters:')}")
            for pname, pinfo in props.items():
                ptype = pinfo.get("type", "any")
                default = pinfo.get("default", "")
                req_mark = "*" if pname in required else " "
                default_str = f"  (default: {default})" if default != "" else ""
                print(f"       {req_mark} {pname}: {ptype}{_dim(default_str)}")
    print()


async def cmd_list_resources() -> None:
    async with stdio_client(_server_params()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.list_resources()

    print(_header("Available Resources"))
    for resource in result.resources:
        print(f"\n  {_bold(str(resource.uri))}")
        if resource.name:
            print(f"     name: {resource.name}")
        if resource.description:
            for line in textwrap.wrap(resource.description, width=70):
                print(f"     {_dim(line)}")
    print()


async def cmd_call_tool(tool_name: str, raw_args: list[str]) -> None:
    """Parse key=value pairs and call the named tool."""
    # Parse key=value args
    kwargs: dict[str, Any] = {}
    for item in raw_args:
        if "=" not in item:
            print(_err(f"Ignored malformed argument (expected key=value): {item}"))
            continue
        k, _, v = item.partition("=")
        kwargs[k.strip()] = v.strip()

    # Coerce numeric strings to int/float for known numeric params
    _INT_PARAMS = {"n_intents", "n_variants", "n_runs", "max_prompts_per_intent",
                   "top_n_brands", "max_reflection_loops"}
    for k in _INT_PARAMS:
        if k in kwargs:
            try:
                kwargs[k] = int(kwargs[k])
            except ValueError:
                pass

    print(_dim(f"\nCalling tool: {tool_name}"))
    if kwargs:
        print(_dim(f"Arguments:   {json.dumps(kwargs, indent=2)}"))
    print(SEPARATOR)

    async with stdio_client(_server_params()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments=kwargs)

    _print_result(result.content)


async def cmd_read_resource(uri: str) -> None:
    async with stdio_client(_server_params()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.read_resource(uri)

    print(_header(f"Resource: {uri}"))
    for block in result.contents:
        if hasattr(block, "text"):
            print(_pretty_json(block.text))
        else:
            print(str(block))
    print()


# ══════════════════════════════════════════════════════════════════════════════
# Interactive REPL
# ══════════════════════════════════════════════════════════════════════════════

_HELP = """
Commands:
  tools                        list all tools
  resources                    list all resources
  call <tool> [key=value ...]  call a tool with arguments
  resource <uri>               read a resource
  run <query>                  shortcut for: call run_pipeline query=<query>
  help, ?                      show this help
  quit, exit, q                exit
"""


async def _fetch_tool_schema(session: ClientSession) -> dict[str, Any]:
    result = await session.list_tools()
    return {t.name: t for t in result.tools}


async def _interactive_call(session: ClientSession, tool_name: str,
                            tools_cache: dict) -> None:
    tool = tools_cache.get(tool_name)
    if tool is None:
        print(_err(f"Unknown tool: '{tool_name}'"))
        print(_dim("Available: " + ", ".join(tools_cache.keys())))
        return

    schema = tool.inputSchema or {}
    props = schema.get("properties", {})
    required = schema.get("required", [])

    print(_dim(f"\n{tool.description or ''}"))

    if not props:
        print(_dim("(no parameters)"))
        kwargs: dict[str, Any] = {}
    else:
        print(_dim("Enter values (press Enter to use default, shown in brackets):\n"))
        kwargs = {}
        for pname, pinfo in props.items():
            ptype = pinfo.get("type", "any")
            default = pinfo.get("default", "")
            req = pname in required
            prompt = f"  {'*' if req else ' '} {pname} ({ptype})"
            if default != "":
                prompt += f" [{default}]"
            prompt += ": "
            try:
                val = input(prompt).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return
            if val == "":
                if default != "":
                    val = default
                elif req:
                    print(_err(f"  Parameter '{pname}' is required."))
                    return
                else:
                    continue  # omit optional with no default

            # Coerce types
            if ptype == "integer":
                try:
                    val = int(val)
                except ValueError:
                    print(_err(f"  '{pname}' must be an integer."))
                    return
            elif ptype == "number":
                try:
                    val = float(val)
                except ValueError:
                    print(_err(f"  '{pname}' must be a number."))
                    return
            elif ptype == "boolean":
                val = val.lower() in ("true", "1", "yes", "y")

            kwargs[pname] = val

    print(f"\n{SEPARATOR}")
    print(_dim(f"Calling {tool_name}..."))
    print(SEPARATOR)

    result = await session.call_tool(tool_name, arguments=kwargs)
    _print_result(result.content)
    print()


async def repl() -> None:
    print(_bold("\n  GEO Pipeline — MCP Client"))
    print(_dim("  Type 'help' for commands, 'quit' to exit\n"))

    async with stdio_client(_server_params()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools_cache = await _fetch_tool_schema(session)

            while True:
                try:
                    line = input("\033[1;32mgeo>\033[0m ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nBye.")
                    break

                if not line:
                    continue

                parts = line.split()
                cmd = parts[0].lower()

                # ── quit ──────────────────────────────────────────────────
                if cmd in ("quit", "exit", "q"):
                    print("Bye.")
                    break

                # ── help ──────────────────────────────────────────────────
                elif cmd in ("help", "?"):
                    print(_HELP)

                # ── tools ─────────────────────────────────────────────────
                elif cmd == "tools":
                    print(_header("Available Tools"))
                    for i, (name, tool) in enumerate(tools_cache.items(), 1):
                        desc = (tool.description or "").split("\n")[0][:60]
                        print(f"  {i:2}. {_bold(name):35} {_dim(desc)}")
                    print()

                # ── resources ─────────────────────────────────────────────
                elif cmd == "resources":
                    res = await session.list_resources()
                    print(_header("Available Resources"))
                    for r in res.resources:
                        print(f"  {_bold(str(r.uri))}")
                        if r.description:
                            print(f"     {_dim(r.description)}")
                    print()

                # ── resource <uri> ────────────────────────────────────────
                elif cmd == "resource":
                    if len(parts) < 2:
                        print(_err("Usage: resource <uri>"))
                        continue
                    uri = parts[1]
                    try:
                        res = await session.read_resource(uri)
                        print(_header(f"Resource: {uri}"))
                        for block in res.contents:
                            if hasattr(block, "text"):
                                print(_pretty_json(block.text))
                            else:
                                print(str(block))
                        print()
                    except Exception as exc:
                        print(_err(f"Error: {exc}"))

                # ── run <query> ───────────────────────────────────────────
                elif cmd == "run":
                    if len(parts) < 2:
                        print(_err("Usage: run <natural-language query>"))
                        continue
                    query = " ".join(parts[1:])
                    print(f"\n{SEPARATOR}")
                    print(_dim(f"Calling run_pipeline with query: {query}"))
                    print(SEPARATOR)
                    result = await session.call_tool(
                        "run_pipeline", arguments={"query": query}
                    )
                    _print_result(result.content)
                    print()

                # ── call <tool> [key=value ...] ───────────────────────────
                elif cmd == "call":
                    if len(parts) < 2:
                        print(_err("Usage: call <tool_name> [key=value ...]"))
                        continue
                    tool_name = parts[1]
                    inline_args = parts[2:]

                    if inline_args:
                        # Parse inline key=value args
                        kwargs: dict[str, Any] = {}
                        for item in inline_args:
                            if "=" not in item:
                                print(_err(f"Ignored: {item}  (expected key=value)"))
                                continue
                            k, _, v = item.partition("=")
                            kwargs[k.strip()] = v.strip()
                        # Coerce integers
                        _INT_PARAMS = {"n_intents", "n_variants", "n_runs",
                                       "max_prompts_per_intent", "top_n_brands",
                                       "max_reflection_loops"}
                        for k in _INT_PARAMS:
                            if k in kwargs:
                                try:
                                    kwargs[k] = int(kwargs[k])
                                except ValueError:
                                    pass
                        print(f"\n{SEPARATOR}")
                        print(_dim(f"Calling {tool_name}..."))
                        print(SEPARATOR)
                        try:
                            result = await session.call_tool(tool_name, arguments=kwargs)
                            _print_result(result.content)
                        except Exception as exc:
                            print(_err(f"Error: {exc}"))
                        print()
                    else:
                        # Interactive parameter prompting
                        await _interactive_call(session, tool_name, tools_cache)

                else:
                    print(_err(f"Unknown command: '{cmd}'"))
                    print(_dim("Type 'help' for available commands."))


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = sys.argv[1:]

    if not args:
        asyncio.run(repl())
        return

    cmd = args[0].lower()

    if cmd in ("tools", "list-tools"):
        asyncio.run(cmd_list_tools())

    elif cmd in ("resources", "list-resources"):
        asyncio.run(cmd_list_resources())

    elif cmd == "call":
        if len(args) < 2:
            print("Usage: python mcp_client.py call <tool_name> [key=value ...]")
            sys.exit(1)
        tool_name = args[1]
        asyncio.run(cmd_call_tool(tool_name, args[2:]))

    elif cmd == "resource":
        if len(args) < 2:
            print("Usage: python mcp_client.py resource <uri>")
            sys.exit(1)
        asyncio.run(cmd_read_resource(args[1]))

    elif cmd in ("help", "--help", "-h"):
        print(__doc__)

    else:
        print(f"Unknown command: '{cmd}'")
        print("Run 'python mcp_client.py help' for usage.")
        sys.exit(1)


if __name__ == "__main__":
    main()
