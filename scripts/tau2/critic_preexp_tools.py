# SPDX-License-Identifier: Apache-2.0
"""Reproduce the E8 WRITE inventory from pinned tau2 decorators without imports."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
from pathlib import Path


def derive_write_inventory(source: Path) -> dict:
    """Read all decorated tools in the two domains present in the audited pool."""
    domains = {}
    for domain in ("airline", "retail"):
        relative = Path("src/tau2/domains") / domain / "tools.py"
        content = (source / relative).read_bytes()
        tools = []
        for node in ast.walk(ast.parse(content)):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for decorator in node.decorator_list:
                if not (
                    isinstance(decorator, ast.Call)
                    and isinstance(decorator.func, ast.Name)
                    and decorator.func.id == "is_tool"
                ):
                    continue
                if len(decorator.args) != 1 or decorator.keywords:
                    raise ValueError("Unsupported is_tool annotation; inspect source")
                kind = decorator.args[0]
                if not (
                    isinstance(kind, ast.Attribute)
                    and isinstance(kind.value, ast.Name)
                    and kind.value.id == "ToolType"
                ):
                    raise ValueError("Unsupported ToolType annotation")
                tools.append(
                    {
                        "name": node.name,
                        "type": kind.attr,
                        "annotation_line": decorator.lineno,
                        "definition_line": node.lineno,
                    }
                )
        if not tools or len({tool["name"] for tool in tools}) != len(tools):
            raise ValueError("Missing or ambiguous domain tools")
        domains[domain] = {
            "source_file": relative.as_posix(),
            "source_sha256": hashlib.sha256(content).hexdigest(),
            "tools": sorted(tools, key=lambda tool: tool["name"]),
            "write_tools": sorted(
                tool["name"] for tool in tools if tool["type"] == "WRITE"
            ),
        }
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    return {
        "source_revision": revision,
        "derivation": "AST is_tool(ToolType.WRITE)",
        "domains": domains,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(
        json.dumps(derive_write_inventory(args.source), indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
