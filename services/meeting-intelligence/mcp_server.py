"""Meeting Intelligence MCP Server — 7 focused tools for the Deep Agent.

Exposes only the tools the Deep Agent needs:
- update_state: Write to shared state (Redis)
- search_transcript: Search meeting transcript
- get_transcript_range: Get raw transcript for a time window
- bot_speak: Make the bot speak via TTS
- send_chat: Send a chat message in the meeting
- bot_screen_share: Share screen content
- stop_screen_share: Stop screen sharing

Runs as a stdio MCP server (spawned by Claude CLI via --mcp-config).
"""

import asyncio
import json
import os
import sys
import logging

logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
logger = logging.getLogger("mcp_meeting")

import redis
import httpx

# --- Config from environment ---
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/1")
MEETING_ID = os.getenv("MEETING_ID", "")
VEXA_API_URL = os.getenv("VEXA_API_URL", "http://localhost:8056")
VEXA_API_KEY = os.getenv("VEXA_API_KEY", "")
PLATFORM = os.getenv("MEETING_PLATFORM", "google_meet")
NATIVE_MEETING_ID = os.getenv("NATIVE_MEETING_ID", "")

# --- Redis client (sync for simplicity in MCP stdio server) ---
_redis = None
def get_redis():
    global _redis
    if _redis is None:
        _redis = redis.from_url(REDIS_URL, decode_responses=True)
    return _redis


# --- Tool implementations ---

def update_state(args: dict) -> str:
    """Update the shared meeting intelligence state."""
    r = get_redis()
    key = f"meeting:{MEETING_ID}:intelligence_state"
    raw = r.get(key)
    state = json.loads(raw) if raw else {
        "version": 0, "meeting": {"id": MEETING_ID}, "context_summary": "",
        "active_topics": [], "resolved_topics": [], "action_items": [],
        "agent_instructions": "", "pending_requests": [],
    }

    for field in ["context_summary", "active_topics", "resolved_topics",
                   "action_items", "agent_instructions"]:
        if field in args:
            state[field] = args[field]

    state["version"] = state.get("version", 0) + 1
    import time
    state["updated_at"] = time.time()
    r.set(key, json.dumps(state, ensure_ascii=False))
    return f"State updated to v{state['version']}"


def search_transcript(args: dict) -> str:
    """Search the full transcript archive by keyword."""
    r = get_redis()
    query = args.get("query", "").lower()
    archive_key = f"meeting:{MEETING_ID}:transcript"
    all_segments = r.lrange(archive_key, 0, -1)

    results = []
    for raw in all_segments:
        seg = json.loads(raw)
        if query in seg.get("text", "").lower():
            results.append(seg)

    if not results:
        return f"Keine Treffer fuer '{args.get('query', '')}'"

    lines = []
    for s in results[:10]:
        ts = s.get("timestamp", 0)
        speaker = s.get("speaker", "?")
        lines.append(f"[{int(ts//60):02d}:{int(ts%60):02d}] {speaker}: {s['text']}")
    return "\n".join(lines)


def get_transcript_range(args: dict) -> str:
    """Get raw transcript segments for a time range."""
    r = get_redis()
    from_s = float(args.get("from_seconds", 0))
    to_s = float(args.get("to_seconds", 9999))
    archive_key = f"meeting:{MEETING_ID}:transcript"
    all_segments = r.lrange(archive_key, 0, -1)

    results = []
    for raw in all_segments:
        seg = json.loads(raw)
        ts = seg.get("timestamp", 0)
        if from_s <= ts <= to_s:
            results.append(seg)

    if not results:
        return f"Kein Transcript im Zeitraum {from_s}-{to_s}s"

    lines = []
    for s in results:
        ts = s.get("timestamp", 0)
        speaker = s.get("speaker", "?")
        lines.append(f"[{int(ts//60):02d}:{int(ts%60):02d}] {speaker}: {s['text']}")
    return "\n".join(lines)


def bot_speak(args: dict) -> str:
    """Make the bot speak in the meeting via TTS."""
    text = args.get("text", "")
    if not text:
        return "Error: no text provided"
    try:
        resp = httpx.post(
            f"{VEXA_API_URL}/bots/{PLATFORM}/{NATIVE_MEETING_ID}/speak",
            headers={"X-API-Key": VEXA_API_KEY, "Content-Type": "application/json"},
            json={"text": text, "provider": "piper", "voice": "alloy"},
            timeout=10,
        )
        return f"Bot spricht: '{text[:50]}...' (HTTP {resp.status_code})"
    except Exception as e:
        return f"Speak failed: {e}"


def send_chat(args: dict) -> str:
    """Send a chat message in the meeting."""
    text = args.get("text", "")
    if not text:
        return "Error: no text provided"
    try:
        resp = httpx.post(
            f"{VEXA_API_URL}/bots/{PLATFORM}/{NATIVE_MEETING_ID}/chat",
            headers={"X-API-Key": VEXA_API_KEY, "Content-Type": "application/json"},
            json={"text": text},
            timeout=10,
        )
        return f"Chat gesendet: '{text[:50]}...' (HTTP {resp.status_code})"
    except Exception as e:
        return f"Chat failed: {e}"


def bot_screen_share(args: dict) -> str:
    """Share content on the bot's screen."""
    url = args.get("url", "")
    if not url:
        return "Error: no url provided"
    try:
        resp = httpx.post(
            f"{VEXA_API_URL}/bots/{PLATFORM}/{NATIVE_MEETING_ID}/screen",
            headers={"X-API-Key": VEXA_API_KEY, "Content-Type": "application/json"},
            json={"url": url},
            timeout=10,
        )
        return f"Screen shared: {url} (HTTP {resp.status_code})"
    except Exception as e:
        return f"Screen share failed: {e}"


def stop_screen_share(args: dict) -> str:
    """Stop screen sharing."""
    try:
        resp = httpx.delete(
            f"{VEXA_API_URL}/bots/{PLATFORM}/{NATIVE_MEETING_ID}/screen",
            headers={"X-API-Key": VEXA_API_KEY},
            timeout=10,
        )
        return f"Screen share stopped (HTTP {resp.status_code})"
    except Exception as e:
        return f"Stop screen share failed: {e}"


# --- MCP Protocol (stdio JSON-RPC) ---

TOOLS = [
    {
        "name": "update_state",
        "description": "Aktualisiere den Meeting-Kontext. Felder: context_summary (string), active_topics (list), resolved_topics (list), action_items (list of {assignee, task, status}), agent_instructions (string).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "context_summary": {"type": "string"},
                "active_topics": {"type": "array", "items": {"type": "string"}},
                "resolved_topics": {"type": "array", "items": {"type": "string"}},
                "action_items": {"type": "array", "items": {"type": "object"}},
                "agent_instructions": {"type": "string"},
            },
        },
    },
    {
        "name": "search_transcript",
        "description": "Durchsuche das Meeting-Transcript nach einem Suchbegriff. Gibt passende Segmente mit Timestamps und Sprecher zurueck.",
        "inputSchema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "get_transcript_range",
        "description": "Hole das Transcript fuer einen bestimmten Zeitraum (Sekunden seit Meeting-Start).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "from_seconds": {"type": "number"},
                "to_seconds": {"type": "number"},
            },
            "required": ["from_seconds", "to_seconds"],
        },
    },
    {
        "name": "bot_speak",
        "description": "Sage etwas im Meeting per Sprachausgabe (TTS). Nutze fuer kurze Antworten wenn jemand wartet. Max 2 Saetze.",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
    {
        "name": "send_chat",
        "description": "Sende eine Chat-Nachricht ins Meeting. Nutze fuer Zitate, Listen, Links, oder leise Vorschlaege die man besser liest als hoert.",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
    {
        "name": "bot_screen_share",
        "description": "Zeige eine URL oder ein Bild auf dem Bot-Bildschirm im Meeting. Teilnehmer sehen es als Bot-Kamera.",
        "inputSchema": {
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
        },
    },
    {
        "name": "stop_screen_share",
        "description": "Beende das Screen Sharing des Bots.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]

TOOL_HANDLERS = {
    "update_state": update_state,
    "search_transcript": search_transcript,
    "get_transcript_range": get_transcript_range,
    "bot_speak": bot_speak,
    "send_chat": send_chat,
    "bot_screen_share": bot_screen_share,
    "stop_screen_share": stop_screen_share,
}


def handle_request(request: dict) -> dict:
    """Handle a JSON-RPC request."""
    method = request.get("method", "")
    req_id = request.get("id")

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "meeting-intelligence", "version": "0.1.0"},
            },
        }

    elif method == "notifications/initialized":
        return None  # No response for notifications

    elif method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"tools": TOOLS},
        }

    elif method == "tools/call":
        tool_name = request.get("params", {}).get("name", "")
        tool_args = request.get("params", {}).get("arguments", {})
        handler = TOOL_HANDLERS.get(tool_name)

        if not handler:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"content": [{"type": "text", "text": f"Unknown tool: {tool_name}"}], "isError": True},
            }

        try:
            result = handler(tool_args)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"content": [{"type": "text", "text": result}]},
            }
        except Exception as e:
            logger.error(f"Tool {tool_name} error: {e}")
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"content": [{"type": "text", "text": f"Error: {e}"}], "isError": True},
            }

    else:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Unknown method: {method}"},
        }


def main():
    """Run MCP server over stdio."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            continue

        response = handle_request(request)
        if response is not None:
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
