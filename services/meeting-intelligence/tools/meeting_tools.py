"""Meeting Intelligence Tools — defined once, used by any provider.

Each tool has a handler that gets injected with the actual services
(shared_state, transcript_manager, vexa_client) at runtime.
"""

from .registry import Tool, ToolDispenser


def create_meeting_tools(
    shared_state,
    transcript_manager,
    vexa_client,
    platform: str,
    native_meeting_id: str,
) -> ToolDispenser:
    """Create a ToolDispenser with all 7 meeting tools wired to real services."""

    dispenser = ToolDispenser()

    # --- 1. update_state ---
    async def _update_state(args: dict) -> str:
        updated = await shared_state.update(args)
        return f"State updated to v{updated['version']}"

    dispenser.register(Tool(
        name="update_state",
        description=(
            "Aktualisiere den Meeting-Kontext. Felder: context_summary (string), "
            "active_topics (list), resolved_topics (list), "
            "action_items (list of {{assignee, task, status}}), "
            "agent_instructions (string — Hinweise fuer den Watcher)."
        ),
        parameters={
            "context_summary": {"type": "string", "description": "Aktualisierte Zusammenfassung des bisherigen Meetings"},
            "active_topics": {"type": "array", "items": {"type": "string"}, "description": "Aktuelle Diskussionsthemen"},
            "resolved_topics": {"type": "array", "items": {"type": "string"}, "description": "Abgeschlossene Themen"},
            "action_items": {"type": "array", "items": {"type": "object"}, "description": "Action Items mit assignee, task, status"},
            "agent_instructions": {"type": "string", "description": "Hinweise fuer den Watcher was als naechstes zu beachten ist"},
        },
        handler=_update_state,
    ))

    # --- 2. search_transcript ---
    async def _search_transcript(args: dict) -> str:
        results = await transcript_manager.search(args.get("query", ""))
        if not results:
            return f"Keine Treffer fuer '{args.get('query', '')}'"
        lines = []
        for s in results[:10]:
            ts = s.get("timestamp", 0)
            speaker = s.get("speaker", "?")
            lines.append(f"[{int(ts//60):02d}:{int(ts%60):02d}] {speaker}: {s['text']}")
        return "\n".join(lines)

    dispenser.register(Tool(
        name="search_transcript",
        description="Durchsuche das Meeting-Transcript nach einem Suchbegriff. Gibt passende Segmente mit Timestamps und Sprecher zurueck.",
        parameters={
            "query": {"type": "string", "description": "Suchbegriff"},
        },
        required=["query"],
        handler=_search_transcript,
    ))

    # --- 3. get_transcript_range ---
    async def _get_transcript_range(args: dict) -> str:
        results = await transcript_manager.get_range(
            float(args.get("from_seconds", 0)),
            float(args.get("to_seconds", 9999)),
        )
        if not results:
            return "Kein Transcript in diesem Zeitraum"
        lines = []
        for s in results:
            ts = s.get("timestamp", 0)
            speaker = s.get("speaker", "?")
            lines.append(f"[{int(ts//60):02d}:{int(ts%60):02d}] {speaker}: {s['text']}")
        return "\n".join(lines)

    dispenser.register(Tool(
        name="get_transcript_range",
        description="Hole das Transcript fuer einen bestimmten Zeitraum (Sekunden seit Meeting-Start).",
        parameters={
            "from_seconds": {"type": "number", "description": "Startzeit in Sekunden"},
            "to_seconds": {"type": "number", "description": "Endzeit in Sekunden"},
        },
        required=["from_seconds", "to_seconds"],
        handler=_get_transcript_range,
    ))

    # --- 4. bot_speak ---
    async def _bot_speak(args: dict) -> str:
        text = args.get("text", "")
        if not text:
            return "Error: kein Text angegeben"
        ok = await vexa_client.speak(platform, native_meeting_id, text)
        return f"Bot spricht: '{text[:50]}'" if ok else "Speak fehlgeschlagen"

    dispenser.register(Tool(
        name="bot_speak",
        description="Sage etwas im Meeting per Sprachausgabe (TTS). Nutze fuer kurze Antworten wenn jemand wartet. Max 2 Saetze.",
        parameters={
            "text": {"type": "string", "description": "Was gesagt werden soll"},
        },
        required=["text"],
        handler=_bot_speak,
    ))

    # --- 5. send_chat ---
    async def _send_chat(args: dict) -> str:
        text = args.get("text", "")
        if not text:
            return "Error: kein Text angegeben"
        ok = await vexa_client.chat(platform, native_meeting_id, text)
        return f"Chat gesendet: '{text[:50]}'" if ok else "Chat fehlgeschlagen"

    dispenser.register(Tool(
        name="send_chat",
        description="Sende eine Chat-Nachricht ins Meeting. Nutze fuer Zitate, Listen, Links, oder leise Vorschlaege.",
        parameters={
            "text": {"type": "string", "description": "Die Chat-Nachricht"},
        },
        required=["text"],
        handler=_send_chat,
    ))

    # --- 6. bot_screen_share ---
    async def _bot_screen_share(args: dict) -> str:
        url = args.get("url", "")
        if not url:
            return "Error: keine URL angegeben"
        ok = await vexa_client.screen_share(platform, native_meeting_id, url)
        return f"Screen shared: {url}" if ok else "Screen share fehlgeschlagen"

    dispenser.register(Tool(
        name="bot_screen_share",
        description="Zeige eine URL oder ein Bild auf dem Bot-Bildschirm. Teilnehmer sehen es als Bot-Kamera.",
        parameters={
            "url": {"type": "string", "description": "URL zum Anzeigen"},
        },
        required=["url"],
        handler=_bot_screen_share,
    ))

    # --- 7. stop_screen_share ---
    async def _stop_screen_share(args: dict) -> str:
        ok = await vexa_client.stop_screen_share(platform, native_meeting_id)
        return "Screen share beendet" if ok else "Stop fehlgeschlagen"

    dispenser.register(Tool(
        name="stop_screen_share",
        description="Beende das Screen Sharing des Bots.",
        parameters={},
        handler=_stop_screen_share,
    ))

    return dispenser
