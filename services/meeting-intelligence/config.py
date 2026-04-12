"""Central configuration — all settings from environment variables."""

import os

# --- STT ---
STT_PROVIDER = os.getenv("STT_PROVIDER", "deepgram")  # deepgram | whisper_batch
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
DEEPGRAM_MODEL = os.getenv("DEEPGRAM_MODEL", "nova-3")
DEEPGRAM_LANGUAGE = os.getenv("DEEPGRAM_LANGUAGE", "multi")

# --- LLM ---
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Quick-Ack model (fastest TTFT)
QUICK_ACK_PROVIDER = os.getenv("QUICK_ACK_PROVIDER", "anthropic")  # anthropic | openai
QUICK_ACK_MODEL = os.getenv("QUICK_ACK_MODEL", "claude-haiku-4-5")

# Deep Agent model
DEEP_AGENT_MODEL = os.getenv("DEEP_AGENT_MODEL", "claude-sonnet-4-6")
DEEP_AGENT_INTERVAL_S = int(os.getenv("DEEP_AGENT_INTERVAL_S", "120"))  # 2 min

# --- Vexa ---
VEXA_API_URL = os.getenv("VEXA_API_URL", "http://localhost:8056")
VEXA_API_KEY = os.getenv("VEXA_API_KEY", "")

# --- Redis ---
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# --- Watcher ---
TRIGGER_KEYWORDS = os.getenv(
    "TRIGGER_KEYWORDS", "Nilo,Hey Nilo,Nilo kannst du"
).split(",")
WATCHER_CLASSIFY_INTERVAL_S = int(os.getenv("WATCHER_CLASSIFY_INTERVAL_S", "10"))

# --- Transcript ---
LIVE_WINDOW_SECONDS = int(os.getenv("LIVE_WINDOW_SECONDS", "300"))  # 5 min
SUMMARY_BLOCK_SECONDS = int(os.getenv("SUMMARY_BLOCK_SECONDS", "300"))  # 5 min

# --- Service ---
SERVICE_PORT = int(os.getenv("MEETING_INTELLIGENCE_PORT", "8200"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
