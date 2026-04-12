"""Output channel selector — decides how the agent responds.

Chat, Speak, or Hand-raise + Chat — based on context.
"""

import re


def suggest_channel(trigger_text: str, response_text: str, context: dict = None) -> str:
    """Suggest the best output channel based on context.

    Returns: "chat" | "speak" | "hand_then_chat"
    """
    context = context or {}
    trigger_lower = trigger_text.lower()
    response_lower = response_text.lower()

    # 1. Response contains structured data → Chat (liest man besser)
    if _has_structured_content(response_text):
        return "chat"

    # 2. Response is a quote/citation → Chat
    if '"' in response_text or "'" in response_text or "sagte" in response_lower:
        return "chat"

    # 3. Direct question with someone waiting → Speak
    if _is_direct_question(trigger_lower):
        return "speak"

    # 4. Agent initiates (not triggered) → Hand + Chat (don't interrupt)
    if context.get("agent_initiated", False):
        return "hand_then_chat"

    # 5. Short confirmation → Speak
    if len(response_text.split()) <= 8:
        return "speak"

    # 6. Default → Chat (safer, doesn't interrupt)
    return "chat"


def _has_structured_content(text: str) -> bool:
    """Check if text contains lists, links, numbers that read better."""
    indicators = [
        r"\d+\.",          # Numbered list
        r"^-\s",           # Bullet point
        r"https?://",      # URL
        r"\d{2}:\d{2}",    # Timestamp
        r"\|",             # Table
    ]
    return any(re.search(p, text, re.MULTILINE) for p in indicators)


def _is_direct_question(text: str) -> bool:
    """Check if someone asked the agent directly."""
    direct_patterns = [
        r"nilo.*was\b",
        r"nilo.*kannst\b",
        r"nilo.*fass\b",
        r"nilo.*sag\b",
        r"nilo.*mach\b",
        r"hey nilo",
        r"nilo\?$",
    ]
    return any(re.search(p, text) for p in direct_patterns)
