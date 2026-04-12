"""Tool Registry — define once, export to any format.

Each tool is defined once with:
- name, description, parameters (JSON Schema)
- handler function (async callable)

The ToolDispenser converts to whatever format the LLM provider needs:
- Anthropic: {"name": ..., "description": ..., "input_schema": ...}
- OpenAI:    {"type": "function", "function": {"name": ..., "parameters": ...}}
- MCP:       {"name": ..., "description": ..., "inputSchema": ...}
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable, Optional

logger = logging.getLogger("meeting_intelligence.tools")


@dataclass
class Tool:
    """A single tool definition — provider-agnostic."""
    name: str
    description: str
    parameters: dict  # JSON Schema
    handler: Callable[..., Awaitable[str]]  # async (args: dict) -> str
    required: list[str] = field(default_factory=list)


class ToolDispenser:
    """Holds tools, converts to any LLM provider format, executes by name."""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    @property
    def names(self) -> list[str]:
        return list(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)

    # --- Format Converters ---

    def for_anthropic(self) -> list[dict]:
        """Anthropic Messages API tool format."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": {
                    "type": "object",
                    "properties": t.parameters,
                    **({"required": t.required} if t.required else {}),
                },
            }
            for t in self._tools.values()
        ]

    def for_openai(self) -> list[dict]:
        """OpenAI function calling format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": {
                        "type": "object",
                        "properties": t.parameters,
                        **({"required": t.required} if t.required else {}),
                    },
                },
            }
            for t in self._tools.values()
        ]

    def for_mcp(self) -> list[dict]:
        """MCP tools/list format."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "inputSchema": {
                    "type": "object",
                    "properties": t.parameters,
                    **({"required": t.required} if t.required else {}),
                },
            }
            for t in self._tools.values()
        ]

    def for_provider(self, provider_name: str) -> list[dict]:
        """Auto-detect format based on provider name."""
        if "anthropic" in provider_name or "claude" in provider_name:
            return self.for_anthropic()
        elif "openai" in provider_name or "gpt" in provider_name:
            return self.for_openai()
        elif "mcp" in provider_name:
            return self.for_mcp()
        else:
            return self.for_anthropic()  # default

    # --- Execution ---

    async def execute(self, name: str, args: dict) -> str:
        """Execute a tool by name. Returns result string."""
        tool = self._tools.get(name)
        if not tool:
            return f"Unknown tool: {name}"

        try:
            result = await tool.handler(args)
            logger.info(f"Tool {name} executed: {str(result)[:80]}")
            return result
        except Exception as e:
            logger.error(f"Tool {name} failed: {e}")
            return f"Error: {e}"
