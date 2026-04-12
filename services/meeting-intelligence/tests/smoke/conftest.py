"""Test fixtures for smoke tests — headless meeting sessions."""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import pytest
import pytest_asyncio
import redis.asyncio as aioredis

from stt.provider import TranscriptSegment
from state.shared_state import SharedState
from state.transcript_manager import TranscriptManager
from actions.queue import ActionQueue, Action, ChatAction, SpeakAction
from agents.watcher import Watcher
from agents.quick_ack import QuickAck
from agents.deep_agent import DeepAgent
from llm.provider import create_provider_stack
from tools.meeting_tools import create_meeting_tools

logger = logging.getLogger("test")


class MockVexaClient:
    """Records actions instead of calling Vexa API."""

    async def startup(self): pass
    async def shutdown(self): pass

    async def speak(self, platform, meeting_id, text, provider="piper", voice="alloy"):
        logger.info(f"[MOCK BOT] speak: {text}")
        return True

    async def chat(self, platform, meeting_id, text):
        logger.info(f"[MOCK BOT] chat: {text}")
        return True

    async def stop_speaking(self, platform, meeting_id):
        return True

    async def screen_share(self, platform, meeting_id, url):
        logger.info(f"[MOCK BOT] screen: {url}")
        return True

    async def stop_screen_share(self, platform, meeting_id):
        return True


class ActionCollector:
    """Wraps ActionQueue to collect actions for assertions."""

    def __init__(self, action_queue: ActionQueue):
        self._queue = action_queue
        self.actions: list[Action] = []
        self._event = asyncio.Event()
        self._original_execute = action_queue._execute

        # Monkey-patch the execute method to collect actions
        async def collecting_execute(action: Action):
            self.actions.append(action)
            self._event.set()
            logger.info(f"[ACTION] {type(action).__name__}: {getattr(action, 'text', '?')[:60]}")

        action_queue._execute = collecting_execute

    async def wait_for_action(self, timeout: float = 5.0) -> Optional[Action]:
        """Wait for the next action to be executed."""
        self._event.clear()
        try:
            await asyncio.wait_for(self._event.wait(), timeout=timeout)
            return self.actions[-1] if self.actions else None
        except asyncio.TimeoutError:
            return None

    async def wait_for_actions(self, count: int, timeout: float = 10.0) -> list[Action]:
        """Wait until at least `count` actions have been collected."""
        deadline = time.time() + timeout
        while len(self.actions) < count:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            self._event.clear()
            try:
                await asyncio.wait_for(self._event.wait(), timeout=remaining)
            except asyncio.TimeoutError:
                break
        return self.actions

    def clear(self):
        self.actions.clear()
        self._event.clear()


class TestMeetingSession:
    """A headless meeting session for testing.

    Injects transcript segments directly — no Deepgram, no Google Meet, no Bot.
    Meeting Intelligence agents run for real with real LLM calls.
    Actions are collected instead of sent to Vexa API.
    """

    def __init__(self, meeting_id: str = "test-meeting-001", platform: str = "google_meet"):
        self.meeting_id = meeting_id
        self.platform = platform
        self._redis = None
        self._tasks: list[asyncio.Task] = []

    async def setup(self):
        """Initialize all components."""
        # Redis (use DB 1 for tests to avoid conflicts)
        self._redis = aioredis.from_url("redis://localhost:6379/1", decode_responses=False)
        await self._redis.flushdb()

        # Set env vars for MCP server
        os.environ["MEETING_ID"] = self.meeting_id
        os.environ["REDIS_URL"] = "redis://localhost:6379/1"

        # LLM providers
        self.llm_providers = create_provider_stack()

        # State
        self.shared_state = SharedState(self._redis, self.meeting_id)
        self.transcript_manager = TranscriptManager(
            self._redis, self.meeting_id, window_seconds=300,
        )

        # Actions (mock — collect instead of execute)
        self._mock_vexa = MockVexaClient()
        self.action_queue = ActionQueue(self._mock_vexa)
        self.collector = ActionCollector(self.action_queue)

        # Tools — defined once, auto-converted per provider
        self.tool_dispenser = create_meeting_tools(
            shared_state=self.shared_state,
            transcript_manager=self.transcript_manager,
            vexa_client=self._mock_vexa,
            platform=self.platform,
            native_meeting_id=self.meeting_id,
        )

        # Agents
        self.quick_ack = QuickAck(
            llm=self.llm_providers["quick"],
            action_queue=self.action_queue,
            platform=self.platform,
            meeting_id=self.meeting_id,
        )

        self.deep_agent = DeepAgent(
            llm=self.llm_providers["deep"],
            tool_dispenser=self.tool_dispenser,
            shared_state=self.shared_state,
            transcript_manager=self.transcript_manager,
            action_queue=self.action_queue,
            platform=self.platform,
            meeting_id=self.meeting_id,
        )

        self.watcher = Watcher(
            keywords=["Nilo", "Hey Nilo"],
            quick_ack=self.quick_ack,
            shared_state=self.shared_state,
            transcript_manager=self.transcript_manager,
            deep_agent_trigger_fn=self.deep_agent.trigger,
        )

        # Start action queue executor
        self._tasks.append(asyncio.create_task(self.action_queue.executor_loop()))

    async def teardown(self):
        """Cleanup."""
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

        if self._redis:
            await self._redis.flushdb()
            await self._redis.aclose()

    async def inject_segment(self, speaker: str, text: str,
                             timestamp: float = 0.0, is_final: bool = True):
        """Inject a transcript segment — simulates what Deepgram would produce."""
        segment = TranscriptSegment(
            text=text,
            speaker=speaker,
            is_final=is_final,
            timestamp=timestamp,
            confidence=0.95,
        )
        await self.watcher.on_segment(segment)

    async def inject_conversation(self, segments: list[tuple], delay: float = 0.1):
        """Inject multiple segments with small delays between them.

        Args:
            segments: List of (timestamp, speaker, text) tuples.
            delay: Seconds between segments.
        """
        for timestamp, speaker, text in segments:
            await self.inject_segment(speaker, text, timestamp=timestamp)
            await asyncio.sleep(delay)


@pytest_asyncio.fixture
async def session():
    """Create a test meeting session with all agents wired up."""
    s = TestMeetingSession()
    await s.setup()
    yield s
    await s.teardown()
