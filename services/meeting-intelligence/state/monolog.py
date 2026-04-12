"""Internal Monolog — async agent-to-agent communication via Redis Streams.

Inspired by A2A (Agent-to-Agent Protocol, Google/IBM/Linux Foundation).
Lightweight implementation over Redis Streams instead of HTTP.

Principles:
- Async-first: agents publish and subscribe without blocking
- Task-IDs: track long-running operations (Deep Agent analysis)
- Broadcast + direct: messages to all or to a specific agent
- Audit trail: every message persisted, queryable
"""

import json
import logging
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Optional, Callable, Awaitable

logger = logging.getLogger("meeting_intelligence.monolog")


@dataclass
class AgentMessage:
    """A single message in the internal monolog."""
    from_agent: str               # "watcher" | "quick_ack" | "deep"
    type: str                     # observation | trigger | instruction |
                                  # context_update | task_started | task_completed |
                                  # relay
    content: str                  # Free-form payload
    to_agent: Optional[str] = None  # None = broadcast, "deep" = direct
    task_id: Optional[str] = None   # For long-running operations
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: float = field(default_factory=time.time)

    def to_redis(self) -> dict:
        """Convert to Redis Stream entry (flat string values)."""
        return {
            "id": self.id,
            "from": self.from_agent,
            "to": self.to_agent or "",
            "type": self.type,
            "content": self.content,
            "task_id": self.task_id or "",
            "timestamp": str(self.timestamp),
        }

    @classmethod
    def from_redis(cls, data: dict) -> "AgentMessage":
        """Parse from Redis Stream entry."""
        # Handle both bytes and str keys
        def get(key):
            return (data.get(key) or data.get(key.encode(), b""))

        val = lambda k: get(k).decode() if isinstance(get(k), bytes) else str(get(k))

        return cls(
            id=val("id"),
            from_agent=val("from"),
            to_agent=val("to") or None,
            type=val("type"),
            content=val("content"),
            task_id=val("task_id") or None,
            timestamp=float(val("timestamp") or 0),
        )


class Monolog:
    """Agent-to-agent message bus over Redis Streams.

    Usage:
        monolog = Monolog(redis, "meeting-123")

        # Publish
        await monolog.publish(AgentMessage(
            from_agent="watcher", type="observation",
            content="Davi redet über Timeline"
        ))

        # Subscribe (async callback)
        monolog.subscribe("deep", on_message_callback)

        # Read recent messages
        messages = await monolog.recent(count=20)

        # Read messages for a specific agent
        messages = await monolog.for_agent("deep", count=10)
    """

    def __init__(self, redis, meeting_id: str):
        self.redis = redis
        self.meeting_id = meeting_id
        self._stream_key = f"meeting:{meeting_id}:monolog"
        self._subscribers: dict[str, list[Callable]] = {}
        self._listen_task = None

    async def publish(self, message: AgentMessage) -> str:
        """Publish a message to the monolog. Returns the Redis stream ID."""
        entry = message.to_redis()
        stream_id = await self.redis.xadd(self._stream_key, entry)

        # Cap the stream at 500 messages
        await self.redis.xtrim(self._stream_key, maxlen=500)

        logger.debug(
            f"[Monolog] {message.from_agent} → {message.to_agent or 'all'}: "
            f"{message.type}: {message.content[:60]}"
        )

        # Notify in-process subscribers
        for agent_id, callbacks in self._subscribers.items():
            if message.to_agent and message.to_agent != agent_id:
                continue  # Direct message not for this subscriber
            for cb in callbacks:
                try:
                    await cb(message)
                except Exception as e:
                    logger.error(f"Monolog subscriber error ({agent_id}): {e}")

        return stream_id

    def subscribe(self, agent_id: str, callback: Callable[[AgentMessage], Awaitable[None]]) -> None:
        """Register a callback for messages directed to or broadcast."""
        self._subscribers.setdefault(agent_id, []).append(callback)

    async def recent(self, count: int = 20) -> list[AgentMessage]:
        """Get the most recent messages."""
        entries = await self.redis.xrevrange(self._stream_key, count=count)
        messages = []
        for stream_id, data in entries:
            try:
                messages.append(AgentMessage.from_redis(data))
            except Exception:
                continue
        messages.reverse()  # Chronological order
        return messages

    async def for_agent(self, agent_id: str, count: int = 10) -> list[AgentMessage]:
        """Get recent messages directed to a specific agent (or broadcast)."""
        all_messages = await self.recent(count=count * 3)  # Over-fetch for filtering
        filtered = [
            m for m in all_messages
            if m.to_agent is None or m.to_agent == agent_id
        ]
        return filtered[:count]

    async def by_task(self, task_id: str) -> list[AgentMessage]:
        """Get all messages for a specific task."""
        all_messages = await self.recent(count=100)
        return [m for m in all_messages if m.task_id == task_id]

    async def cleanup(self) -> None:
        """Remove the monolog stream."""
        await self.redis.delete(self._stream_key)

    def new_task_id(self) -> str:
        """Generate a new task ID for tracking long-running operations."""
        return f"task-{uuid.uuid4().hex[:8]}"
