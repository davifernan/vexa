"""Meeting Intelligence Service — the brain behind the Vexa bot.

Listens to transcript streams, detects triggers, responds via Vexa API.
All agents run asynchronously and in parallel.
"""

import asyncio
import json
import logging
import sys
from contextlib import asynccontextmanager

import redis.asyncio as aioredis
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import config
from llm.provider import create_provider_stack
from stt.deepgram_provider import DeepgramProvider
from state.shared_state import SharedState
from state.transcript_manager import TranscriptManager
from actions.queue import ActionQueue
from actions.vexa_client import VexaClient
from agents.watcher import Watcher
from agents.quick_ack import QuickAck
from agents.deep_agent import DeepAgent
from tools.meeting_tools import create_meeting_tools
from state.monolog import Monolog

# Logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("meeting_intelligence")

# --- Active meeting sessions ---
active_sessions: dict[str, "MeetingSession"] = {}


class MeetingSession:
    """All agents and state for one active meeting."""

    def __init__(self, meeting_id: str, platform: str, redis_client,
                 vexa_client: VexaClient, llm_providers: dict):
        self.meeting_id = meeting_id
        self.platform = platform
        self.redis = redis_client
        self._llm_providers = llm_providers

        # State
        self.shared_state = SharedState(redis_client, meeting_id)
        self.transcript_manager = TranscriptManager(
            redis_client, meeting_id, window_seconds=config.LIVE_WINDOW_SECONDS,
        )
        self.monolog = Monolog(redis_client, meeting_id)

        # Actions
        self.action_queue = ActionQueue(vexa_client)

        # Tools — defined once, converted to any provider format
        self.tool_dispenser = create_meeting_tools(
            shared_state=self.shared_state,
            transcript_manager=self.transcript_manager,
            vexa_client=vexa_client,
            platform=platform,
            native_meeting_id=meeting_id,
        )

        # Agents — each gets the right LLM provider for their role
        self.quick_ack = QuickAck(
            llm=llm_providers["quick"],
            action_queue=self.action_queue,
            platform=platform,
            meeting_id=meeting_id,
        )
        self.deep_agent = DeepAgent(
            llm=llm_providers["deep"],
            tool_dispenser=self.tool_dispenser,
            shared_state=self.shared_state,
            transcript_manager=self.transcript_manager,
            action_queue=self.action_queue,
            platform=platform,
            meeting_id=meeting_id,
        )
        self.watcher = Watcher(
            keywords=config.TRIGGER_KEYWORDS,
            quick_ack=self.quick_ack,
            shared_state=self.shared_state,
            transcript_manager=self.transcript_manager,
            monolog=self.monolog,
            deep_agent_trigger_fn=self.deep_agent.trigger,
            classifier_llm=llm_providers.get("quick"),  # Haiku for semantic classify
        )

        # STT
        self.stt = None
        if config.STT_PROVIDER == "deepgram" and config.DEEPGRAM_API_KEY:
            self.stt = DeepgramProvider(
                api_key=config.DEEPGRAM_API_KEY,
                model=config.DEEPGRAM_MODEL,
                language=config.DEEPGRAM_LANGUAGE,
            )
            self.stt.on_transcript(self._on_transcript_sync)

        self._tasks: list[asyncio.Task] = []

    def _on_transcript_sync(self, segment):
        """Sync callback from Deepgram — bridge to async watcher."""
        asyncio.create_task(self.watcher.on_segment(segment))

    async def start(self) -> None:
        """Start all async loops."""
        logger.info(f"Starting session for meeting {self.meeting_id} ({self.platform})")

        # Connect STT
        if self.stt:
            await self.stt.connect()

        # Start async loops
        self._tasks = [
            asyncio.create_task(self.action_queue.executor_loop()),
            asyncio.create_task(self.deep_agent.periodic_loop()),
            asyncio.create_task(self._transcript_summarizer_loop()),
        ]

        # Also listen to Redis for transcript segments from Vexa bot
        # (when STT runs inside the bot, segments come via Redis)
        self._tasks.append(
            asyncio.create_task(self._redis_transcript_listener())
        )

        logger.info(f"Session started: {len(self._tasks)} async tasks running")

    async def stop(self) -> None:
        """Stop all loops and cleanup."""
        logger.info(f"Stopping session for meeting {self.meeting_id}")

        await self.deep_agent.stop()
        await self.action_queue.stop()

        if self.stt:
            await self.stt.disconnect()

        for task in self._tasks:
            task.cancel()

        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []

        logger.info(f"Session stopped for meeting {self.meeting_id}")

    async def _redis_transcript_listener(self) -> None:
        """Listen to Redis stream for transcript segments published by the Vexa bot.

        This is the primary input when STT runs inside the bot (batch mode)
        or as a secondary input alongside Deepgram.
        """
        stream_key = "transcription_segments"
        last_id = "$"  # Only new messages

        logger.info(f"Listening to Redis stream '{stream_key}' for meeting {self.meeting_id}")

        while True:
            try:
                results = await self.redis.xread(
                    {stream_key: last_id}, count=10, block=2000,
                )

                for stream_name, messages in results:
                    for msg_id, data in messages:
                        last_id = msg_id
                        await self._handle_redis_segment(data)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Redis listener error: {e}")
                await asyncio.sleep(1)

    async def _handle_redis_segment(self, data: dict) -> None:
        """Parse a Redis stream segment and feed to watcher."""
        from stt.provider import TranscriptSegment

        # Redis stream data comes as {b'key': b'value'}
        text = data.get(b"text", data.get("text", b""))
        if isinstance(text, bytes):
            text = text.decode()

        if not text.strip():
            return

        speaker = data.get(b"speaker", data.get("speaker", b""))
        if isinstance(speaker, bytes):
            speaker = speaker.decode()

        segment = TranscriptSegment(
            text=text,
            speaker=speaker or None,
            is_final=True,  # Redis segments from Vexa are always final
            timestamp=float(data.get(b"start", data.get("start", 0))),
        )

        await self.watcher.on_segment(segment)

    async def _transcript_summarizer_loop(self) -> None:
        """Periodically compress old transcript segments."""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30s
                if self.transcript_manager.should_summarize():
                    await self.transcript_manager.create_summary(
                        self._summarize_text
                    )
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Summarizer error: {e}")

    async def _summarize_text(self, text: str) -> str:
        """Use the cheapest available LLM to summarize a block of transcript."""
        # Uses whatever provider is configured for summarization
        # (Claude CLI/Max = $0, or Haiku API, or GPT-4o-mini)
        summarizer = self._llm_providers.get("summarizer")
        if not summarizer:
            return text[:200] + "..."
        resp = await summarizer.generate(
            system="Fasse das folgende Meeting-Transkript in 2-3 Saetzen zusammen. Deutsch, sachlich.",
            prompt=text,
            max_tokens=200,
        )
        return resp.text

    async def send_audio(self, chunk: bytes) -> None:
        """Send audio chunk to STT provider (if direct audio mode)."""
        if self.stt:
            await self.stt.send_audio(chunk)


# --- FastAPI App ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    app.state.redis = aioredis.from_url(config.REDIS_URL, decode_responses=False)
    app.state.vexa = VexaClient(config.VEXA_API_URL, config.VEXA_API_KEY)
    await app.state.vexa.startup()
    app.state.llm_providers = create_provider_stack()
    logger.info("Meeting Intelligence Service started")
    yield
    # Shutdown
    for session in active_sessions.values():
        await session.stop()
    await app.state.vexa.shutdown()
    await app.state.redis.aclose()
    logger.info("Meeting Intelligence Service stopped")


app = FastAPI(title="Meeting Intelligence Service", lifespan=lifespan)


# --- API Endpoints ---

class StartSessionRequest(BaseModel):
    meeting_id: str
    platform: str = "google_meet"


class StopSessionRequest(BaseModel):
    meeting_id: str


@app.post("/sessions/start")
async def start_session(req: StartSessionRequest):
    if req.meeting_id in active_sessions:
        raise HTTPException(400, f"Session already active for meeting {req.meeting_id}")

    session = MeetingSession(
        meeting_id=req.meeting_id,
        platform=req.platform,
        redis_client=app.state.redis,
        vexa_client=app.state.vexa,
        llm_providers=app.state.llm_providers,
    )
    await session.start()
    active_sessions[req.meeting_id] = session

    return {"status": "started", "meeting_id": req.meeting_id}


@app.post("/sessions/stop")
async def stop_session(req: StopSessionRequest):
    session = active_sessions.pop(req.meeting_id, None)
    if not session:
        raise HTTPException(404, f"No active session for meeting {req.meeting_id}")

    await session.stop()
    return {"status": "stopped", "meeting_id": req.meeting_id}


@app.get("/sessions")
async def list_sessions():
    return {
        "active": [
            {"meeting_id": mid, "platform": s.platform}
            for mid, s in active_sessions.items()
        ]
    }


@app.get("/sessions/{meeting_id}/state")
async def get_session_state(meeting_id: str):
    session = active_sessions.get(meeting_id)
    if not session:
        raise HTTPException(404, f"No active session for meeting {meeting_id}")
    return await session.shared_state.read()


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "active_sessions": len(active_sessions),
        "stt_provider": config.STT_PROVIDER,
        "deep_agent_model": config.DEEP_AGENT_MODEL,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=config.SERVICE_PORT)
