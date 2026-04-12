"""Deep Agent — the curator of shared state and intelligent responder.

Runs periodically (every 2 min) and on-demand when triggered.
Uses Sonnet with full meeting context. Can make multiple tool-call turns.
"""

import asyncio
import json
import logging
import time
from typing import Optional

import config
from llm.provider import LLMProvider
from state.shared_state import SharedState
from state.transcript_manager import TranscriptManager
from actions.queue import ActionQueue, ChatAction, SpeakAction, ScreenAction
from stt.provider import TranscriptSegment

logger = logging.getLogger("meeting_intelligence.deep_agent")

SYSTEM_PROMPT = """Du bist der Deep Agent eines Meeting-Assistenten namens Nilo.

Deine Aufgabe:
1. Den Shared State aktualisieren (context_summary, active_topics, action_items, agent_instructions)
2. Auf Trigger reagieren (Fragen beantworten, Aufgaben erledigen)
3. Entscheiden WIE du reagierst (Chat, Sprache, Zeichnung, Hand heben)

Regeln:
- Aktualisiere den State mit neuen Erkenntnissen aus dem Transcript
- Beantworte Trigger-Anfragen praezise und kurz
- Nutze Chat fuer Zitate, Listen, Links (liest man besser)
- Nutze Sprache fuer kurze Antworten wenn jemand wartet
- Hebe die Hand wenn du einen wichtigen Punkt hast aber nicht unterbrechen willst
- Halte agent_instructions aktuell — was sollte der Watcher als naechstes beachten?
- Deutsch bitte, knapp und natuerlich"""


TOOLS = [
    {
        "name": "update_state",
        "description": "Aktualisiere den Shared State mit neuen Erkenntnissen. Felder: context_summary, active_topics, resolved_topics, action_items, agent_instructions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "context_summary": {"type": "string", "description": "Aktualisierte Zusammenfassung des bisherigen Meetings"},
                "active_topics": {"type": "array", "items": {"type": "string"}, "description": "Aktuelle Themen"},
                "resolved_topics": {"type": "array", "items": {"type": "string"}, "description": "Abgeschlossene Themen"},
                "action_items": {"type": "array", "items": {"type": "object"}, "description": "Action Items mit assignee, task, status"},
                "agent_instructions": {"type": "string", "description": "Hinweise fuer den Watcher was als naechstes zu beachten ist"},
            },
        },
    },
    {
        "name": "search_transcript",
        "description": "Durchsuche das gesamte Meeting-Transcript nach einem Suchbegriff. Gibt passende Segmente mit Timestamps zurueck.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Suchbegriff"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_transcript_range",
        "description": "Hole das rohe Transcript fuer einen Zeitraum (in Sekunden seit Meeting-Start).",
        "input_schema": {
            "type": "object",
            "properties": {
                "from_seconds": {"type": "number", "description": "Startzeit in Sekunden"},
                "to_seconds": {"type": "number", "description": "Endzeit in Sekunden"},
            },
            "required": ["from_seconds", "to_seconds"],
        },
    },
    {
        "name": "send_chat",
        "description": "Sende eine Chat-Nachricht ins Meeting. Nutze fuer: Zitate, Listen, Links, leise Vorschlaege.",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Die Chat-Nachricht"},
            },
            "required": ["text"],
        },
    },
    {
        "name": "speak",
        "description": "Sage etwas im Meeting per TTS. Nutze fuer: kurze Antworten wenn jemand wartet.",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Was gesagt werden soll (kurz, max 2 Saetze)"},
            },
            "required": ["text"],
        },
    },
]


class DeepAgent:
    """The brain — curates shared state and responds intelligently."""

    def __init__(
        self,
        llm: LLMProvider,
        shared_state: SharedState,
        transcript_manager: TranscriptManager,
        action_queue: ActionQueue,
        platform: str,
        meeting_id: str,
    ):
        self.llm = llm
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
        """Run a full analysis cycle with agentic tool-call loop."""
        self._busy = True

        try:
            state = await self.state.read()
            live_window = self.transcript.get_live_window()
            summaries = self.transcript.get_summaries_text()
            pending = await self.state.clear_pending_requests()

            user_message = self._build_prompt(state, live_window, summaries, pending, extra_context)

            messages = [{"role": "user", "content": user_message}]

            # Agentic loop — multiple turns if tool calls needed
            for turn in range(10):  # max 10 turns
                response = await self.llm.generate_with_tools(
                    system=SYSTEM_PROMPT,
                    messages=messages,
                    tools=TOOLS,
                    max_tokens=2000,
                )

                # Process response
                if response.get("stop_reason") == "tool_use":
                    tool_results = await self._execute_tool_calls(response)
                    messages.append({"role": "assistant", "content": response["content"]})
                    messages.append({"role": "user", "content": tool_results})
                else:
                    # Final text response (if any)
                    content = response.get("content", [])
                    for block in (content if isinstance(content, list) else [content]):
                        text = getattr(block, "text", None) or (block if isinstance(block, str) else None)
                        if text:
                            logger.info(f"Deep agent final: {str(text)[:100]}")
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

    async def _execute_tool_calls(self, response) -> list:
        """Execute tool calls and return results."""
        results = []

        for block in response.content:
            if block.type != "tool_use":
                continue

            name = block.name
            args = block.input
            result = None

            try:
                if name == "update_state":
                    updated = await self.state.update(args)
                    result = f"State updated to v{updated['version']}"

                elif name == "search_transcript":
                    segments = await self.transcript.search(args["query"])
                    result = json.dumps(segments[:10], ensure_ascii=False)

                elif name == "get_transcript_range":
                    segments = await self.transcript.get_range(
                        args["from_seconds"], args["to_seconds"]
                    )
                    result = json.dumps(segments, ensure_ascii=False)

                elif name == "send_chat":
                    await self.action_queue.push(ChatAction(
                        text=args["text"],
                        platform=self.platform,
                        meeting_id=self.meeting_id,
                    ))
                    result = "Chat message queued"

                elif name == "speak":
                    await self.action_queue.push(SpeakAction(
                        text=args["text"],
                        platform=self.platform,
                        meeting_id=self.meeting_id,
                    ))
                    result = "Speech queued"

                else:
                    result = f"Unknown tool: {name}"

            except Exception as e:
                result = f"Tool error: {e}"
                logger.error(f"Tool {name} failed: {e}")

            results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": str(result),
            })

        return results
