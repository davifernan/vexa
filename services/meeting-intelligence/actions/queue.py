"""Async action queue — receives actions from all agents, executes via Vexa API."""

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

from .vexa_client import VexaClient

logger = logging.getLogger("meeting_intelligence.actions")


# --- Action Types ---

@dataclass
class ChatAction:
    text: str
    platform: str = "google_meet"
    meeting_id: str = ""


@dataclass
class SpeakAction:
    text: str
    provider: str = "piper"
    voice: str = "alloy"
    platform: str = "google_meet"
    meeting_id: str = ""


@dataclass
class ScreenAction:
    url: str
    platform: str = "google_meet"
    meeting_id: str = ""


@dataclass
class StopScreenAction:
    platform: str = "google_meet"
    meeting_id: str = ""


@dataclass
class StopSpeakingAction:
    platform: str = "google_meet"
    meeting_id: str = ""


Action = ChatAction | SpeakAction | ScreenAction | StopScreenAction | StopSpeakingAction


class ActionQueue:
    """Async queue that executes bot actions via Vexa API.

    All agents push actions here. The executor loop runs them
    asynchronously — fire and forget, no blocking.
    """

    def __init__(self, vexa_client: VexaClient):
        self.vexa = vexa_client
        self._queue: asyncio.Queue[Action] = asyncio.Queue()
        self._running = False

    async def push(self, action: Action) -> None:
        """Push an action to the queue. Non-blocking."""
        await self._queue.put(action)

    async def executor_loop(self) -> None:
        """Main loop — takes actions from queue, executes async."""
        self._running = True
        logger.info("Action queue executor started")

        while self._running:
            try:
                action = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            # Fire and forget — don't block the queue
            asyncio.create_task(self._execute(action))

    async def stop(self) -> None:
        self._running = False

    async def _execute(self, action: Action) -> None:
        try:
            if isinstance(action, ChatAction):
                await self.vexa.chat(action.platform, action.meeting_id, action.text)

            elif isinstance(action, SpeakAction):
                await self.vexa.speak(
                    action.platform, action.meeting_id,
                    action.text, action.provider, action.voice,
                )

            elif isinstance(action, ScreenAction):
                await self.vexa.screen_share(
                    action.platform, action.meeting_id, action.url,
                )

            elif isinstance(action, StopScreenAction):
                await self.vexa.stop_screen_share(action.platform, action.meeting_id)

            elif isinstance(action, StopSpeakingAction):
                await self.vexa.stop_speaking(action.platform, action.meeting_id)

            else:
                logger.warning(f"Unknown action type: {type(action)}")

        except Exception as e:
            logger.error(f"Action execution failed: {type(action).__name__}: {e}")
