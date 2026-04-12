"""Abstract base for realtime STT providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class TranscriptSegment:
    """A single transcript segment from the STT provider."""
    text: str
    speaker: Optional[str] = None
    is_final: bool = False
    confidence: float = 0.0
    timestamp: float = 0.0
    duration: float = 0.0
    words: list = field(default_factory=list)
    language: Optional[str] = None


@dataclass
class SpeakerEvent:
    """Speaker change event."""
    speaker: str
    event_type: str  # "joined" | "left" | "speaking" | "silent"
    timestamp: float = 0.0


TranscriptCallback = Callable[[TranscriptSegment], None]
SpeakerCallback = Callable[[SpeakerEvent], None]


class RealtimeSTTProvider(ABC):
    """Abstract base class for realtime speech-to-text providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    def supports_diarization(self) -> bool:
        return False

    @property
    def supports_interim_results(self) -> bool:
        return False

    @abstractmethod
    async def connect(self) -> None:
        """Open connection to the STT service."""
        ...

    @abstractmethod
    async def send_audio(self, chunk: bytes) -> None:
        """Send an audio chunk for transcription."""
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Close the connection cleanly."""
        ...

    def on_transcript(self, callback: TranscriptCallback) -> None:
        """Register callback for transcript segments."""
        self._transcript_cb = callback

    def on_speaker_change(self, callback: SpeakerCallback) -> None:
        """Register callback for speaker events."""
        self._speaker_cb = callback
