"""Transcript manager — live window, summaries, and archive."""

import json
import logging
import time
from dataclasses import asdict
from typing import Optional

from stt.provider import TranscriptSegment

logger = logging.getLogger("meeting_intelligence.transcript")


class TranscriptManager:
    """Manages transcript data with three tiers:
    - Live Window: last N seconds, raw segments (always in prompt)
    - Summaries: per 5-min block, compressed (always in prompt)
    - Archive: full transcript in Redis (searchable via tools)
    """

    def __init__(self, redis, meeting_id: str, window_seconds: int = 300):
        self.redis = redis
        self.meeting_id = meeting_id
        self.window_seconds = window_seconds
        self._archive_key = f"meeting:{meeting_id}:transcript"
        self._live_buffer: list[dict] = []
        self.summaries: list[dict] = []
        self._last_summary_time: float = time.time()
        self._segment_count = 0

    async def add_segment(self, segment: TranscriptSegment) -> None:
        """Add a new transcript segment to all tiers."""
        entry = {
            "id": self._segment_count,
            "text": segment.text,
            "speaker": segment.speaker,
            "timestamp": segment.timestamp,
            "duration": segment.duration,
            "is_final": segment.is_final,
            "language": segment.language,
            "time_received": time.time(),
        }
        self._segment_count += 1

        # Live buffer
        self._live_buffer.append(entry)
        self._trim_live_buffer()

        # Archive (only final segments)
        if segment.is_final:
            await self.redis.rpush(self._archive_key, json.dumps(entry))

    def get_live_window(self) -> str:
        """Get the live window as formatted text for LLM context."""
        if not self._live_buffer:
            return "(keine aktuellen Segmente)"

        lines = []
        for seg in self._live_buffer:
            speaker = seg.get("speaker", "Unknown")
            text = seg["text"]
            ts = seg.get("timestamp", 0)
            mins = int(ts // 60)
            secs = int(ts % 60)
            lines.append(f"[{mins:02d}:{secs:02d}] {speaker}: {text}")
        return "\n".join(lines)

    def get_summaries_text(self) -> str:
        """Get all summaries as formatted text for LLM context."""
        if not self.summaries:
            return "(noch keine Zusammenfassungen)"

        lines = []
        for s in self.summaries:
            lines.append(f"[{s['time_range']}] {s['summary']}")
        return "\n".join(lines)

    async def search(self, query: str, time_range: Optional[tuple] = None) -> list[dict]:
        """Search the full archive by keyword. Tool for Deep Agent."""
        results = []
        all_raw = await self.redis.lrange(self._archive_key, 0, -1)
        query_lower = query.lower()

        for raw in all_raw:
            seg = json.loads(raw)
            if query_lower in seg.get("text", "").lower():
                if time_range:
                    ts = seg.get("timestamp", 0)
                    if not (time_range[0] <= ts <= time_range[1]):
                        continue
                results.append(seg)

        return results

    async def get_range(self, from_ts: float, to_ts: float) -> list[dict]:
        """Get raw segments in a time range. Tool for Deep Agent."""
        results = []
        all_raw = await self.redis.lrange(self._archive_key, 0, -1)

        for raw in all_raw:
            seg = json.loads(raw)
            ts = seg.get("timestamp", 0)
            if from_ts <= ts <= to_ts:
                results.append(seg)

        return results

    def should_summarize(self) -> bool:
        """Check if it's time to compress old segments."""
        return (time.time() - self._last_summary_time) >= self.window_seconds

    async def create_summary(self, summarize_fn) -> Optional[dict]:
        """Compress the oldest segments outside the live window.

        Args:
            summarize_fn: async callable(text) -> summary string
        """
        now = time.time()
        cutoff = now - self.window_seconds
        old_segments = [s for s in self._live_buffer if s["time_received"] < cutoff]

        if not old_segments:
            return None

        # Build text to summarize
        text = "\n".join(
            f"{s.get('speaker', '?')}: {s['text']}" for s in old_segments
        )

        summary_text = await summarize_fn(text)
        first_ts = old_segments[0].get("timestamp", 0)
        last_ts = old_segments[-1].get("timestamp", 0)

        summary = {
            "time_range": f"{int(first_ts // 60):02d}:{int(first_ts % 60):02d}-{int(last_ts // 60):02d}:{int(last_ts % 60):02d}",
            "summary": summary_text,
            "segment_count": len(old_segments),
            "created_at": now,
        }

        self.summaries.append(summary)
        self._last_summary_time = now

        return summary

    def _trim_live_buffer(self) -> None:
        """Remove segments older than the live window."""
        if not self._live_buffer:
            return
        cutoff = time.time() - self.window_seconds
        self._live_buffer = [
            s for s in self._live_buffer if s["time_received"] >= cutoff
        ]
