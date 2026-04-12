"""Judge LLM — evaluates agent responses when keyword assertions fail.

Two-step evaluation:
1. Hard check (keywords, regex) — free, instant
2. If hard check fails → Judge LLM decides PASS/FAIL — cheap, ~$0.001

This avoids false FAILs when the agent answers correctly
but uses different words than expected.
"""

import logging
from typing import Optional

logger = logging.getLogger("meeting_intelligence.judge")

# Will be set by test setup
_judge_llm = None


def set_judge_llm(llm) -> None:
    """Set the LLM provider used for judging. Call once in conftest."""
    global _judge_llm
    _judge_llm = llm


async def judge_response(
    question: str,
    response: str,
    expected_keywords: list[str],
    context: str = "",
) -> dict:
    """Two-step evaluation of an agent response.

    Returns: {"passed": bool, "method": "keyword"|"judge", "detail": str}
    """
    # Step 1: Keyword match (free, instant)
    response_lower = response.lower()
    matched = [kw for kw in expected_keywords if kw.lower() in response_lower]

    if matched:
        return {
            "passed": True,
            "method": "keyword",
            "detail": f"Keywords matched: {matched}",
        }

    # Step 2: Judge LLM (cheap, ~$0.001)
    if not _judge_llm:
        return {
            "passed": False,
            "method": "keyword_only",
            "detail": f"No keywords matched ({expected_keywords}), no Judge LLM configured",
        }

    try:
        verdict = await _judge_llm.generate(
            system=(
                "Du bist ein Evaluator. Bewerte ob ein Meeting-Assistent "
                "eine Frage korrekt beantwortet hat. "
                "Antworte NUR mit PASS oder FAIL und einem kurzen Grund."
            ),
            prompt=(
                f"Frage/Trigger: \"{question}\"\n"
                f"Antwort des Assistenten: \"{response}\"\n"
                f"{f'Meeting-Kontext: {context}' if context else ''}\n"
                f"Erwartete Themen: {expected_keywords}\n\n"
                f"Hat der Assistent die Frage inhaltlich korrekt beantwortet, "
                f"auch wenn er andere Worte benutzt hat?\n"
                f"Bewertung:"
            ),
            max_tokens=30,
        )

        text = verdict.text.strip().upper()
        passed = "PASS" in text

        return {
            "passed": passed,
            "method": "judge",
            "detail": verdict.text.strip(),
        }

    except Exception as e:
        logger.error(f"Judge LLM failed: {e}")
        return {
            "passed": False,
            "method": "judge_error",
            "detail": f"Judge failed: {e}",
        }


async def judge_state_field(
    field_name: str,
    field_value: str,
    expected_content: str,
) -> dict:
    """Judge whether a state field contains expected information."""
    if not field_value:
        return {"passed": False, "method": "empty", "detail": f"{field_name} is empty"}

    # Step 1: Simple contains check
    if expected_content.lower() in field_value.lower():
        return {"passed": True, "method": "contains", "detail": "Direct match"}

    # Step 2: Judge LLM
    if not _judge_llm:
        return {"passed": False, "method": "no_judge", "detail": "No match, no judge"}

    try:
        verdict = await _judge_llm.generate(
            system="Bewerte ob ein Feld den erwarteten Inhalt enthaelt. Antworte NUR PASS oder FAIL.",
            prompt=(
                f"Feld '{field_name}': \"{field_value}\"\n"
                f"Erwartet: \"{expected_content}\"\n"
                f"Enthaelt das Feld die erwartete Information (auch umformuliert)?"
            ),
            max_tokens=15,
        )
        passed = "PASS" in verdict.text.upper()
        return {"passed": passed, "method": "judge", "detail": verdict.text.strip()}

    except Exception as e:
        return {"passed": False, "method": "judge_error", "detail": str(e)}
