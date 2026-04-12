"""Shared State — Redis-backed meeting context.

Only the Deep Agent writes. All other agents read.
"""

import json
import logging
import time
from typing import Optional

logger = logging.getLogger("meeting_intelligence.state")


def default_state(meeting_id: str) -> dict:
    return {
        "version": 0,
        "updated_at": None,
        "meeting": {
            "id": meeting_id,
            "title": None,
            "started_at": time.time(),
            "participants": [],
        },
        "context_summary": "",
        "active_topics": [],
        "resolved_topics": [],
        "action_items": [],
        "agent_instructions": "",
        "pending_requests": [],
    }


class SharedState:
    """Redis-backed shared state for a meeting session."""

    def __init__(self, redis, meeting_id: str):
        self.redis = redis
        self.meeting_id = meeting_id
        self._key = f"meeting:{meeting_id}:intelligence_state"

    async def read(self) -> dict:
        raw = await self.redis.get(self._key)
        if raw:
            return json.loads(raw)
        return default_state(self.meeting_id)

    async def update(self, updates: dict) -> dict:
        """Merge updates into state. Only call from Deep Agent."""
        state = await self.read()
        state.update(updates)
        state["version"] = state.get("version", 0) + 1
        state["updated_at"] = time.time()
        await self.redis.set(self._key, json.dumps(state))
        logger.info(f"State updated to v{state['version']} for meeting {self.meeting_id}")
        return state

    async def add_pending_request(self, request: dict) -> None:
        """Add a request from Watcher or Quick-Ack for the Deep Agent to process."""
        state = await self.read()
        state.setdefault("pending_requests", []).append({
            **request,
            "created_at": time.time(),
        })
        await self.redis.set(self._key, json.dumps(state))

    async def clear_pending_requests(self) -> list:
        """Deep Agent: read and clear pending requests."""
        state = await self.read()
        requests = state.get("pending_requests", [])
        state["pending_requests"] = []
        await self.redis.set(self._key, json.dumps(state))
        return requests

    async def get_summary(self) -> str:
        """Quick read of context summary — for Quick-Ack and Watcher."""
        state = await self.read()
        return state.get("context_summary", "")

    async def get_instructions(self) -> str:
        """Quick read of agent instructions — for Watcher."""
        state = await self.read()
        return state.get("agent_instructions", "")

    async def cleanup(self) -> None:
        """Remove state after meeting ends."""
        await self.redis.delete(self._key)
