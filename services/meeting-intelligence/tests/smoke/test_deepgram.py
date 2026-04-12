"""Test Deepgram STT provider with a real audio file."""

import asyncio
import os
from pathlib import Path

import pytest

# Test audio from Vexa's transcription service tests
TEST_AUDIO = Path(__file__).parent.parent.parent.parent / "transcription-service" / "tests" / "test_audio.wav"


@pytest.mark.asyncio
async def test_deepgram_transcribes_audio():
    """Send a real WAV file to Deepgram and verify we get text back."""
    api_key = os.getenv("DEEPGRAM_API_KEY", "")
    if not api_key:
        pytest.skip("DEEPGRAM_API_KEY not set")

    if not TEST_AUDIO.exists():
        pytest.skip(f"Test audio not found: {TEST_AUDIO}")

    from stt.deepgram_provider import DeepgramProvider
    from stt.provider import TranscriptSegment

    # Collect transcripts
    segments: list[TranscriptSegment] = []

    def on_transcript(seg: TranscriptSegment):
        segments.append(seg)
        print(f"  [{seg.speaker or '?'}] {'FINAL' if seg.is_final else 'interim'}: {seg.text}")

    provider = DeepgramProvider(api_key=api_key, model="nova-3", language="en")
    provider.on_transcript(on_transcript)

    await provider.connect()

    # Read audio and send in chunks (simulate real-time)
    audio_data = TEST_AUDIO.read_bytes()
    # Skip WAV header (44 bytes)
    pcm_data = audio_data[44:]

    # Send in 3200-byte chunks (100ms at 16kHz 16-bit mono)
    chunk_size = 3200
    for i in range(0, len(pcm_data), chunk_size):
        chunk = pcm_data[i:i + chunk_size]
        await provider.send_audio(chunk)
        await asyncio.sleep(0.05)  # Simulate real-time pacing

    # Wait for final transcripts
    await asyncio.sleep(3.0)
    await provider.disconnect()

    # Assertions
    assert len(segments) > 0, "No transcripts received from Deepgram"

    final_segments = [s for s in segments if s.is_final]
    assert len(final_segments) > 0, "No final transcripts received"

    full_text = " ".join(s.text for s in final_segments)
    print(f"\nFull transcript: {full_text}")
    assert len(full_text) > 10, f"Transcript too short: {full_text}"

    print(f"\nDeepgram test PASSED: {len(final_segments)} final segments, {len(full_text)} chars")
