"""Deepgram Whisper-API Proxy — accepts Whisper API format, transcribes via Deepgram.

The Vexa Bot sends audio as multipart POST to /v1/audio/transcriptions (Whisper format).
This proxy accepts that format and forwards to Deepgram's REST API instead.
No bot code changes needed — just point TRANSCRIPTION_SERVICE_URL here.

Usage:
    DEEPGRAM_API_KEY=xxx python3 -m stt.deepgram_whisper_proxy

Then set in Vexa .env:
    TRANSCRIPTION_SERVICE_URL=http://localhost:8201/v1/audio/transcriptions
"""

import io
import json
import logging
import os
import sys
import time

import httpx
from fastapi import FastAPI, File, Form, UploadFile, Header
from fastapi.responses import JSONResponse
import uvicorn

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger("deepgram_proxy")

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
DEEPGRAM_MODEL = os.getenv("DEEPGRAM_MODEL", "nova-3")
PORT = int(os.getenv("DEEPGRAM_PROXY_PORT", "8201"))

app = FastAPI(title="Deepgram Whisper-API Proxy")


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form("whisper-1"),
    response_format: str = Form("verbose_json"),
    language: str = Form(None),
    timestamp_granularities: str = Form(None),
    prompt: str = Form(None),
    max_speech_duration_s: float = Form(None),
    min_silence_duration_ms: int = Form(None),
):
    """Accept Whisper API format, transcribe via Deepgram, return Whisper-compatible response."""
    if not DEEPGRAM_API_KEY:
        return JSONResponse({"error": "DEEPGRAM_API_KEY not set"}, status_code=500)

    start = time.time()
    audio_bytes = await file.read()

    # Build Deepgram REST API URL
    params = {
        "model": DEEPGRAM_MODEL,
        "smart_format": "true",
        "punctuate": "true",
        "utterances": "true",
    }
    if language:
        params["language"] = language
    else:
        params["detect_language"] = "true"

    if timestamp_granularities and "word" in timestamp_granularities:
        # Deepgram always returns word timestamps, no extra param needed
        pass

    url = "https://api.deepgram.com/v1/listen?" + "&".join(f"{k}={v}" for k, v in params.items())

    # Determine content type
    content_type = file.content_type or "audio/wav"
    if file.filename and file.filename.endswith(".wav"):
        content_type = "audio/wav"

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                url,
                content=audio_bytes,
                headers={
                    "Authorization": f"Token {DEEPGRAM_API_KEY}",
                    "Content-Type": content_type,
                },
            )

        if resp.status_code != 200:
            logger.error(f"Deepgram error {resp.status_code}: {resp.text[:200]}")
            return JSONResponse({"error": resp.text}, status_code=resp.status_code)

        dg_result = resp.json()

        # Convert Deepgram response to Whisper-compatible format
        whisper_response = _convert_to_whisper_format(dg_result)

        elapsed = time.time() - start
        logger.info(
            f"Transcribed {len(audio_bytes)} bytes in {elapsed:.2f}s: "
            f"\"{whisper_response.get('text', '')[:60]}\""
        )

        return JSONResponse(whisper_response)

    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


def _convert_to_whisper_format(dg: dict) -> dict:
    """Convert Deepgram response to Whisper verbose_json format."""
    results = dg.get("results", {})
    channels = results.get("channels", [{}])

    if not channels:
        return {"text": "", "language": "unknown", "duration": 0, "segments": []}

    channel = channels[0]
    alternatives = channel.get("alternatives", [{}])

    if not alternatives:
        return {"text": "", "language": "unknown", "duration": 0, "segments": []}

    alt = alternatives[0]
    full_text = alt.get("transcript", "")

    # Build segments from Deepgram paragraphs or utterances
    segments = []
    dg_words = alt.get("words", [])

    if dg_words:
        # Group words into segments by punctuation
        current_segment = {"start": dg_words[0]["start"], "text": "", "words": []}

        for w in dg_words:
            current_segment["text"] += (" " if current_segment["text"] else "") + w["word"]
            current_segment["words"].append({
                "word": w["word"],
                "start": w["start"],
                "end": w["end"],
                "probability": w.get("confidence", 0.9),
            })
            current_segment["end"] = w["end"]

            # Split on sentence-ending punctuation
            if w["word"].rstrip().endswith((".", "!", "?")):
                segments.append({
                    "start": current_segment["start"],
                    "end": current_segment["end"],
                    "text": current_segment["text"].strip(),
                    "words": current_segment["words"],
                })
                if w != dg_words[-1]:
                    next_idx = dg_words.index(w) + 1
                    if next_idx < len(dg_words):
                        current_segment = {"start": dg_words[next_idx]["start"], "text": "", "words": []}

        # Don't forget the last segment
        if current_segment["text"].strip():
            segments.append({
                "start": current_segment["start"],
                "end": current_segment.get("end", 0),
                "text": current_segment["text"].strip(),
                "words": current_segment["words"],
            })

    # Detect language
    detected_lang = "unknown"
    lang_prob = 0.0
    detected = results.get("metadata", {}).get("detected_language")
    if detected:
        detected_lang = detected
        lang_prob = 0.95
    elif channel.get("detected_language"):
        detected_lang = channel["detected_language"]
        lang_prob = channel.get("language_confidence", 0.9)

    duration = results.get("metadata", {}).get("duration", 0)

    return {
        "text": full_text,
        "language": detected_lang,
        "language_probability": lang_prob,
        "duration": duration,
        "segments": segments,
    }


@app.get("/health")
async def health():
    return {"status": "ok", "provider": "deepgram", "model": DEEPGRAM_MODEL}


if __name__ == "__main__":
    if not DEEPGRAM_API_KEY:
        print("ERROR: DEEPGRAM_API_KEY not set")
        sys.exit(1)
    print(f"Starting Deepgram Whisper-API Proxy on port {PORT}")
    print(f"Model: {DEEPGRAM_MODEL}")
    print(f"Set TRANSCRIPTION_SERVICE_URL=http://localhost:{PORT}/v1/audio/transcriptions")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
