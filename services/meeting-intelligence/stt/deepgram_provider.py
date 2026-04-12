"""Deepgram Nova-3 realtime STT provider via WebSocket."""

import asyncio
import json
import logging
import time
from urllib.parse import urlencode

import websockets

from . import provider as base

logger = logging.getLogger("meeting_intelligence.stt.deepgram")


class DeepgramProvider(base.RealtimeSTTProvider):
    """Streams audio to Deepgram Nova-3 via WebSocket, emits transcript segments."""

    def __init__(self, api_key: str, model: str = "nova-3", language: str = "multi"):
        self._api_key = api_key
        self._model = model
        self._language = language
        self._ws = None
        self._listen_task: asyncio.Task | None = None
        self._transcript_cb: base.TranscriptCallback | None = None
        self._speaker_cb: base.SpeakerCallback | None = None
        self._connected = False

    @property
    def name(self) -> str:
        return f"deepgram-{self._model}"

    @property
    def supports_diarization(self) -> bool:
        return True

    @property
    def supports_interim_results(self) -> bool:
        return True

    async def connect(self) -> None:
        params = {
            "model": self._model,
            "language": self._language,
            "interim_results": "true",
            "utterance_end_ms": "1500",
            "diarize": "true",
            "smart_format": "true",
            "encoding": "linear16",
            "sample_rate": "16000",
            "channels": "1",
            "punctuate": "true",
        }
        url = f"wss://api.deepgram.com/v1/listen?{urlencode(params)}"
        headers = {"Authorization": f"Token {self._api_key}"}

        logger.info(f"Connecting to Deepgram ({self._model}, lang={self._language})")
        self._ws = await websockets.connect(url, additional_headers=headers)
        self._connected = True
        self._listen_task = asyncio.create_task(self._listen_loop())
        logger.info("Deepgram connected")

    async def send_audio(self, chunk: bytes) -> None:
        if self._ws and self._connected:
            try:
                await self._ws.send(chunk)
            except websockets.exceptions.ConnectionClosed:
                logger.warning("Deepgram WebSocket closed, cannot send audio")
                self._connected = False

    async def disconnect(self) -> None:
        self._connected = False
        if self._ws:
            # Send close signal to Deepgram
            try:
                await self._ws.send(json.dumps({"type": "CloseStream"}))
            except Exception:
                pass
            await self._ws.close()
            self._ws = None
        if self._listen_task:
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass
        logger.info("Deepgram disconnected")

    async def _listen_loop(self) -> None:
        """Read messages from Deepgram WebSocket and dispatch callbacks."""
        try:
            async for raw in self._ws:
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                msg_type = msg.get("type", "")

                if msg_type == "Results":
                    await self._handle_results(msg)
                elif msg_type == "UtteranceEnd":
                    pass  # Could be used for turn detection
                elif msg_type == "Metadata":
                    logger.debug(f"Deepgram metadata: {msg}")
                elif msg_type == "Error":
                    logger.error(f"Deepgram error: {msg}")

        except websockets.exceptions.ConnectionClosed as e:
            logger.warning(f"Deepgram connection closed: {e}")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Deepgram listen error: {e}")

    async def _handle_results(self, msg: dict) -> None:
        """Parse a Deepgram Results message into TranscriptSegments."""
        channel = msg.get("channel", {})
        alternatives = channel.get("alternatives", [])
        if not alternatives:
            return

        alt = alternatives[0]
        text = alt.get("transcript", "").strip()
        if not text:
            return

        is_final = msg.get("is_final", False)
        start_time = msg.get("start", 0.0)
        duration = msg.get("duration", 0.0)
        confidence = alt.get("confidence", 0.0)

        # Speaker diarization — extract from words
        speaker = None
        words_data = alt.get("words", [])
        if words_data:
            # Most common speaker in this segment
            speakers = [w.get("speaker") for w in words_data if w.get("speaker") is not None]
            if speakers:
                speaker = f"speaker_{max(set(speakers), key=speakers.count)}"

        words = [
            {
                "word": w.get("word", ""),
                "start": w.get("start", 0.0),
                "end": w.get("end", 0.0),
                "confidence": w.get("confidence", 0.0),
                "speaker": w.get("speaker"),
            }
            for w in words_data
        ]

        segment = base.TranscriptSegment(
            text=text,
            speaker=speaker,
            is_final=is_final,
            confidence=confidence,
            timestamp=start_time,
            duration=duration,
            words=words,
            language=channel.get("detected_language"),
        )

        if self._transcript_cb:
            try:
                self._transcript_cb(segment)
            except Exception as e:
                logger.error(f"Transcript callback error: {e}")
