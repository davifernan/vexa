"""LLM Provider abstraction — subscription-first, API as fallback.

IMPORTANT: Subscription mode (Claude Max, Codex) is for INTERNAL USE ONLY.
If this becomes a premium product, switch to API mode with proper billing.

Provider hierarchy:
1. Claude Max (subscription) — claude CLI subprocess, $0
2. Codex (subscription) — ChatGPT Plus OAuth, $0
3. Anthropic API — pay-per-use, needs ANTHROPIC_API_KEY
4. OpenAI API — pay-per-use, needs OPENAI_API_KEY
"""

import asyncio
import json
import logging
import os
import shutil
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("meeting_intelligence.llm")


@dataclass
class LLMResponse:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0  # $0 for subscriptions


class LLMProvider(ABC):
    """Abstract LLM provider."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    def is_subscription(self) -> bool:
        return False

    @abstractmethod
    async def generate(
        self, system: str, prompt: str, max_tokens: int = 500, model: Optional[str] = None,
    ) -> LLMResponse: ...

    @abstractmethod
    async def generate_with_tools(
        self, system: str, messages: list, tools: list,
        max_tokens: int = 2000, model: Optional[str] = None,
    ) -> dict: ...


class ClaudeCliProvider(LLMProvider):
    """Claude via CLI subprocess — uses Max subscription, $0 cost.

    Calls `claude` CLI with --print flag. Requires claude to be installed
    and authenticated (Max subscription or Claude Code).
    """

    def __init__(self, default_model: str = "sonnet"):
        self._default_model = default_model
        self._available: Optional[bool] = None

    @property
    def name(self) -> str:
        return "claude-cli"

    @property
    def is_subscription(self) -> bool:
        return True

    def is_available(self) -> bool:
        if self._available is None:
            self._available = shutil.which("claude") is not None
        return self._available

    async def generate(
        self, system: str, prompt: str, max_tokens: int = 500, model: Optional[str] = None,
    ) -> LLMResponse:
        model = model or self._default_model
        full_prompt = f"{system}\n\n---\n\n{prompt}" if system else prompt

        cmd = [
            "claude", "--print",
            "--dangerously-skip-permissions",
            "--model", model,
            "-p", full_prompt,
        ]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)

        if proc.returncode != 0:
            raise RuntimeError(f"Claude CLI failed: {stderr.decode()[:200]}")

        text = stdout.decode().strip()
        return LLMResponse(text=text, model=f"claude-cli/{model}", cost=0.0)

    async def generate_with_tools(
        self, system: str, messages: list, tools: list,
        max_tokens: int = 2000, model: Optional[str] = None,
    ) -> dict:
        # CLI doesn't support tool calls natively — fall through to API
        raise NotImplementedError("Claude CLI does not support tool calls. Use API provider.")


class AnthropicApiProvider(LLMProvider):
    """Claude via Anthropic API — pay-per-use."""

    def __init__(self, api_key: str, default_model: str = "claude-sonnet-4-6"):
        import anthropic
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self._default_model = default_model

    @property
    def name(self) -> str:
        return "anthropic-api"

    async def generate(
        self, system: str, prompt: str, max_tokens: int = 500, model: Optional[str] = None,
    ) -> LLMResponse:
        model = model or self._default_model
        resp = await self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return LLMResponse(
            text=resp.content[0].text.strip(),
            model=model,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
        )

    async def generate_with_tools(
        self, system: str, messages: list, tools: list,
        max_tokens: int = 2000, model: Optional[str] = None,
    ) -> dict:
        model = model or self._default_model
        resp = await self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
            tools=tools,
        )
        return {
            "content": resp.content,
            "stop_reason": resp.stop_reason,
            "model": model,
            "usage": {"input": resp.usage.input_tokens, "output": resp.usage.output_tokens},
        }


class OpenAiApiProvider(LLMProvider):
    """OpenAI via API — pay-per-use."""

    def __init__(self, api_key: str, default_model: str = "gpt-4o-mini"):
        import openai
        self._client = openai.AsyncOpenAI(api_key=api_key)
        self._default_model = default_model

    @property
    def name(self) -> str:
        return "openai-api"

    async def generate(
        self, system: str, prompt: str, max_tokens: int = 500, model: Optional[str] = None,
    ) -> LLMResponse:
        model = model or self._default_model
        resp = await self._client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        )
        return LLMResponse(
            text=resp.choices[0].message.content.strip(),
            model=model,
            input_tokens=resp.usage.prompt_tokens if resp.usage else 0,
            output_tokens=resp.usage.completion_tokens if resp.usage else 0,
        )

    async def generate_with_tools(
        self, system: str, messages: list, tools: list,
        max_tokens: int = 2000, model: Optional[str] = None,
    ) -> dict:
        # Convert Anthropic tool format to OpenAI format
        openai_tools = []
        for t in tools:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {}),
                },
            })

        model = model or self._default_model
        resp = await self._client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "system", "content": system}] + messages,
            tools=openai_tools if openai_tools else None,
        )
        return {
            "content": resp.choices[0].message,
            "stop_reason": resp.choices[0].finish_reason,
            "model": model,
        }


def create_provider_stack() -> dict[str, LLMProvider]:
    """Create available providers in priority order.

    Returns dict with role-based keys:
    - "quick": fastest TTFT for Quick-Ack (subscription preferred)
    - "deep": most capable for Deep Agent (subscription preferred, needs tools)
    - "summarizer": cheap model for transcript summaries
    """
    providers: dict[str, LLMProvider] = {}

    # 1. Try Claude CLI (Max subscription, $0)
    cli = ClaudeCliProvider(default_model="sonnet")
    cli_available = cli.is_available()

    if cli_available:
        logger.info("Claude CLI available — using Max subscription ($0)")
        providers["quick"] = ClaudeCliProvider(default_model="haiku")
        providers["summarizer"] = ClaudeCliProvider(default_model="haiku")
        # Deep Agent needs tool calls — CLI doesn't support them
        # So we still need API for deep agent
    else:
        logger.info("Claude CLI not available — using API providers")

    # 2. API providers as fallback / for tool calls
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
    openai_key = os.getenv("OPENAI_API_KEY", "")

    if anthropic_key:
        api = AnthropicApiProvider(anthropic_key)
        # Deep Agent always uses API (needs tool calls)
        providers.setdefault("deep", api)
        providers.setdefault("quick", AnthropicApiProvider(anthropic_key, "claude-haiku-4-5"))
        providers.setdefault("summarizer", AnthropicApiProvider(anthropic_key, "claude-haiku-4-5"))
        logger.info("Anthropic API configured")

    if openai_key:
        oai = OpenAiApiProvider(openai_key, "gpt-4o-mini")
        providers.setdefault("quick", oai)  # gpt-4o-mini has fast TTFT
        providers.setdefault("summarizer", oai)
        logger.info("OpenAI API configured")

    if not providers:
        raise RuntimeError(
            "No LLM provider available. Install Claude CLI (Max subscription) "
            "or set ANTHROPIC_API_KEY / OPENAI_API_KEY."
        )

    logger.info(
        f"Provider stack: quick={providers.get('quick', {}).name}, "
        f"deep={providers.get('deep', {}).name}, "
        f"summarizer={providers.get('summarizer', {}).name}"
    )

    return providers
