"""Watcher agent — always-on transcript monitor with keyword trigger detection."""

import asyncio
import logging
import time
from typing import Optional

import config
from stt.provider import TranscriptSegment
from state.shared_state import SharedState
from state.transcript_manager import TranscriptManager
from state.monolog import Monolog, AgentMessage
from .quick_ack import QuickAck

logger = logging.getLogger("meeting_intelligence.watcher")


class RollingBuffer:
    """Keeps the last N seconds of transcript segments."""

    def __init__(self, window_seconds: int = 30):
        self.window = window_seconds
        self._segments: list[TranscriptSegment] = []

    def add(self, segment: TranscriptSegment) -> None:
        self._segments.append(segment)
        self._trim()

    def get_text(self) -> str:
        return " ".join(s.text for s in self._segments)

    def get_last_n(self, n: int = 5) -> list[TranscriptSegment]:
        return self._segments[-n:]

    def _trim(self) -> None:
        cutoff = time.time() - self.window
        self._segments = [s for s in self._segments if s.timestamp >= cutoff or True]
        # Keep max 100 segments regardless of time
        if len(self._segments) > 100:
            self._segments = self._segments[-100:]


class Watcher:
    """Always-on agent that monitors the transcript stream for triggers.

    Detection methods:
    1. Keyword matching (instant, free) — catches "Nilo", "Hey Nilo"
    2. Optional Mini-LLM classifier (periodic) — catches semantic triggers

    On trigger: fires Quick-Ack + Deep Agent in parallel (async, non-blocking).
    """

    def __init__(
        self,
        keywords: list[str],
        quick_ack: QuickAck,
        shared_state: SharedState,
        transcript_manager: TranscriptManager,
        monolog: Monolog,
        deep_agent_trigger_fn=None,
        classifier_llm=None,
    ):
        self.keywords = [kw.lower() for kw in keywords]
        self.quick_ack = quick_ack
        self.shared_state = shared_state
        self.transcript_manager = transcript_manager
        self.monolog = monolog
        self._deep_agent_trigger = deep_agent_trigger_fn
        self._classifier_llm = classifier_llm
        self.buffer = RollingBuffer(window_seconds=30)
        self._running = False
        self._last_classify_time = 0.0
        self._last_trigger_time = 0.0
        self._cooldown_seconds = 5.0  # Don't re-trigger within 5s
        self._classify_interval = config.WATCHER_CLASSIFY_INTERVAL_S

    async def on_segment(self, segment: TranscriptSegment) -> None:
        """Called for every transcript segment from STT. Main entry point."""
        self.buffer.add(segment)

        # Add to transcript manager (async, non-blocking)
        asyncio.create_task(self.transcript_manager.add_segment(segment))

        # Only process final segments for triggers
        if not segment.is_final:
            return

        # Cooldown check
        now = time.time()
        if (now - self._last_trigger_time) < self._cooldown_seconds:
            return

        # 1. Keyword match (instant, free)
        if self._keyword_match(segment.text):
            logger.info(f"Keyword trigger: \"{segment.text[:60]}\"")
            self._last_trigger_time = now
            await self._fire_trigger(segment, reason="keyword")
            return

        # 2. Periodic semantic classifier (Mini-LLM, every N seconds)
        if self._classifier_llm and (now - self._last_classify_time) >= self._classify_interval:
            self._last_classify_time = now
            asyncio.create_task(self._semantic_classify())

    async def _fire_trigger(self, segment: TranscriptSegment, reason: str) -> None:
        """Fire Quick-Ack + Deep Agent in parallel. Non-blocking."""
        task_id = self.monolog.new_task_id()

        # Log to monolog
        asyncio.create_task(self.monolog.publish(AgentMessage(
            from_agent="watcher",
            type="trigger",
            content=f"{reason}: \"{segment.text[:80]}\"",
            task_id=task_id,
        )))

        # Quick-Ack + Deep Agent in parallel
        tasks = [
            asyncio.create_task(
                self.quick_ack.respond(segment.text, channel="chat")
            ),
        ]

        if self._deep_agent_trigger:
            tasks.append(
                asyncio.create_task(
                    self._deep_agent_trigger(segment, reason=reason)
                )
            )

        for task in tasks:
            task.add_done_callback(self._task_done_callback)

    async def _semantic_classify(self) -> None:
        """Periodic semantic classification — catches triggers without keywords."""
        try:
            instructions = await self.shared_state.get_instructions()
            buffer_text = self.buffer.get_text()
            if not buffer_text.strip():
                return

            resp = await self._classifier_llm.generate(
                system=(
                    "Du bist ein Meeting-Monitor. Klassifiziere ob der Meeting-Assistent "
                    "reagieren sollte. Antworte NUR mit einem Wort: TRIGGER, OBSERVE, oder IGNORE.\n"
                    "TRIGGER: Jemand stellt eine Frage an die Runde oder braucht Hilfe.\n"
                    "OBSERVE: Interessante Information, merken fuer spaeter.\n"
                    "IGNORE: Normales Gespraech, nichts zu tun."
                ),
                prompt=(
                    f"Letzte 30 Sekunden:\n{buffer_text}\n\n"
                    f"Instruktionen: {instructions or 'keine'}\n\n"
                    f"Klassifizierung:"
                ),
                max_tokens=5,
            )

            result = resp.text.strip().upper()

            if "TRIGGER" in result:
                logger.info(f"Semantic trigger detected: {buffer_text[-60:]}")
                # Create a synthetic segment for the trigger
                last_segments = self.buffer.get_last_n(1)
                if last_segments:
                    self._last_trigger_time = time.time()
                    await self._fire_trigger(last_segments[0], reason="semantic")

            elif "OBSERVE" in result:
                await self.monolog.publish(AgentMessage(
                    from_agent="watcher",
                    type="observation",
                    content=buffer_text[-200:],
                ))

        except Exception as e:
            logger.error(f"Semantic classify failed: {e}")

    def _keyword_match(self, text: str) -> bool:
        text_lower = text.lower()
        return any(kw in text_lower for kw in self.keywords)

    @staticmethod
    def _task_done_callback(task: asyncio.Task) -> None:
        if task.exception():
            logger.error(f"Trigger task failed: {task.exception()}")
