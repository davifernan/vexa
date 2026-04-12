"""Watcher agent — always-on transcript monitor with keyword trigger detection."""

import asyncio
import logging
import time
from typing import Optional

import config
from stt.provider import TranscriptSegment
from state.shared_state import SharedState
from state.transcript_manager import TranscriptManager
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
        deep_agent_trigger_fn=None,
    ):
        self.keywords = [kw.lower() for kw in keywords]
        self.quick_ack = quick_ack
        self.shared_state = shared_state
        self.transcript_manager = transcript_manager
        self._deep_agent_trigger = deep_agent_trigger_fn
        self.buffer = RollingBuffer(window_seconds=30)
        self._running = False
        self._last_classify_time = 0.0
        self._last_trigger_time = 0.0
        self._cooldown_seconds = 5.0  # Don't re-trigger within 5s

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

        # 2. Optional periodic LLM classifier
        # (not implemented in MVP — add later if keyword matching isn't enough)

    async def _fire_trigger(self, segment: TranscriptSegment, reason: str) -> None:
        """Fire Quick-Ack + Deep Agent in parallel. Non-blocking."""
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

        # Don't await — fire and forget
        for task in tasks:
            task.add_done_callback(self._task_done_callback)

    def _keyword_match(self, text: str) -> bool:
        text_lower = text.lower()
        return any(kw in text_lower for kw in self.keywords)

    @staticmethod
    def _task_done_callback(task: asyncio.Task) -> None:
        if task.exception():
            logger.error(f"Trigger task failed: {task.exception()}")
