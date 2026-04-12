"""Deep Agent — the curator of shared state and intelligent responder.

Runs periodically (every 2 min) and on-demand when triggered.
Uses the configured LLM provider with tools from ToolDispenser.
Tools are defined once and auto-converted to the right format.
"""

import asyncio
import json
import logging
import time
from typing import Optional

import config
from llm.provider import LLMProvider
from tools.registry import ToolDispenser
from state.shared_state import SharedState
from state.transcript_manager import TranscriptManager
from actions.queue import ActionQueue, ChatAction, SpeakAction
from stt.provider import TranscriptSegment

logger = logging.getLogger("meeting_intelligence.deep_agent")

SYSTEM_PROMPT = """Du bist der Deep Agent eines Meeting-Assistenten namens Nilo.

Deine Aufgabe:
1. Den Shared State aktualisieren (update_state Tool)
2. Auf Trigger reagieren (Fragen beantworten, Aufgaben erledigen)
3. Entscheiden WIE du reagierst (Chat, Sprache, Zeichnung, Hand heben)

Regeln:
- Aktualisiere den State mit neuen Erkenntnissen aus dem Transcript
- Beantworte Trigger-Anfragen praezise und kurz
- Nutze send_chat fuer Zitate, Listen, Links (liest man besser)
- Nutze bot_speak fuer kurze Antworten wenn jemand wartet
- Halte agent_instructions aktuell — was sollte der Watcher als naechstes beachten?
- Deutsch bitte, knapp und natuerlich"""


class DeepAgent:
    """The brain — curates shared state and responds intelligently.

    Uses ToolDispenser for provider-agnostic tool definitions.
    Works with Claude CLI + MCP ($0), Anthropic API, or OpenAI API.
    """

    def __init__(
        self,
        llm: LLMProvider,
        tool_dispenser: ToolDispenser,
        shared_state: SharedState,
        transcript_manager: TranscriptManager,
        action_queue: ActionQueue,
        platform: str,
        meeting_id: str,
    ):
        self.llm = llm
        self.tools = tool_dispenser
        self.state = shared_state
        self.transcript = transcript_manager
        self.action_queue = action_queue
        self.platform = platform
        self.meeting_id = meeting_id
        self._running = False
        self._busy = False

    async def periodic_loop(self) -> None:
        """Run analysis every N seconds."""
        self._running = True
        logger.info(f"Deep agent periodic loop started (interval={config.DEEP_AGENT_INTERVAL_S}s)")

        while self._running:
            await asyncio.sleep(config.DEEP_AGENT_INTERVAL_S)
            if not self._busy:
                await self.analyze()

    async def trigger(self, segment: TranscriptSegment, reason: str = "keyword") -> None:
        """Immediate analysis triggered by Watcher."""
        if self._busy:
            logger.info("Deep agent busy, queueing trigger as pending request")
            await self.state.add_pending_request({
                "from": "watcher",
                "type": "trigger",
                "reason": reason,
                "text": segment.text,
                "speaker": segment.speaker,
            })
            return

        logger.info(f"Deep agent triggered: {reason} — \"{segment.text[:60]}\"")
        await self.analyze(extra_context={
            "trigger_reason": reason,
            "trigger_text": segment.text,
            "trigger_speaker": segment.speaker,
        })

    async def analyze(self, extra_context: Optional[dict] = None) -> None:
        """Run a full analysis cycle."""
        self._busy = True

        try:
            state = await self.state.read()
            live_window = self.transcript.get_live_window()
            summaries = self.transcript.get_summaries_text()
            pending = await self.state.clear_pending_requests()

            user_message = self._build_prompt(state, live_window, summaries, pending, extra_context)
            messages = [{"role": "user", "content": user_message}]

            # Get tools in the right format for this provider
            provider_tools = self.tools.for_provider(self.llm.name)

            if self.llm.name == "claude-cli":
                # Claude CLI handles tool calls itself via MCP
                # We just get the final text output back
                response = await self.llm.generate_with_tools(
                    system=SYSTEM_PROMPT,
                    messages=messages,
                    tools=provider_tools,  # ignored for CLI, MCP server has them
                    max_tokens=2000,
                )
                # Log final output
                content = response.get("content", [])
                for block in (content if isinstance(content, list) else [content]):
                    text = block.get("text", "") if isinstance(block, dict) else str(block)
                    if text:
                        logger.info(f"Deep agent (CLI): {text[:100]}")

            else:
                # API providers — we handle the agentic loop manually
                for turn in range(10):
                    response = await self.llm.generate_with_tools(
                        system=SYSTEM_PROMPT,
                        messages=messages,
                        tools=provider_tools,
                        max_tokens=2000,
                    )

                    stop = response.get("stop_reason", "")

                    if stop == "tool_use":
                        # Anthropic format
                        tool_results = await self._handle_anthropic_tools(response)
                        messages.append({"role": "assistant", "content": response["content"]})
                        messages.append({"role": "user", "content": tool_results})

                    elif stop == "stop" and self._has_openai_tool_calls(response):
                        # OpenAI format
                        tool_results = await self._handle_openai_tools(response)
                        messages.append(response["content"])  # assistant message
                        messages.extend(tool_results)  # tool result messages

                    else:
                        # Final response, no more tool calls
                        content = response.get("content", [])
                        for block in (content if isinstance(content, list) else [content]):
                            text = getattr(block, "text", None) or (
                                block.get("text", "") if isinstance(block, dict) else str(block)
                            )
                            if text:
                                logger.info(f"Deep agent (API): {text[:100]}")
                        break

        except Exception as e:
            logger.error(f"Deep agent analysis failed: {e}")
        finally:
            self._busy = False

    async def stop(self) -> None:
        self._running = False

    def _build_prompt(self, state, live_window, summaries, pending, extra_context):
        parts = [
            f"## Aktueller Shared State\n```json\n{json.dumps(state, indent=2, ensure_ascii=False)}\n```",
            f"\n## Live Transcript (letzte 5 Min)\n{live_window}",
            f"\n## Bisherige Zusammenfassungen\n{summaries}",
        ]
        if pending:
            parts.append(f"\n## Offene Anfragen\n{json.dumps(pending, indent=2, ensure_ascii=False)}")
        if extra_context:
            parts.append(f"\n## Trigger\n{json.dumps(extra_context, indent=2, ensure_ascii=False)}")
        parts.append(
            "\n## Deine Aufgabe\n"
            "1. Aktualisiere den State (update_state Tool)\n"
            "2. Reagiere auf Trigger/Anfragen wenn vorhanden\n"
            "3. Entscheide ob Chat, Sprache, oder nichts"
        )
        return "\n".join(parts)

    # --- Anthropic tool call handling ---

    async def _handle_anthropic_tools(self, response) -> list:
        results = []
        for block in response.get("content", []):
            if not (hasattr(block, "type") and block.type == "tool_use"):
                continue
            result = await self.tools.execute(block.name, block.input)
            results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result,
            })
        return results

    # --- OpenAI tool call handling ---

    def _has_openai_tool_calls(self, response) -> bool:
        content = response.get("content")
        if hasattr(content, "tool_calls"):
            return bool(content.tool_calls)
        return False

    async def _handle_openai_tools(self, response) -> list:
        messages = []
        content = response["content"]
        for tc in content.tool_calls:
            args = json.loads(tc.function.arguments) if isinstance(tc.function.arguments, str) else tc.function.arguments
            result = await self.tools.execute(tc.function.name, args)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })
        return messages
