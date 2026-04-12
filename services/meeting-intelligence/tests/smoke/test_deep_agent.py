"""Smoke tests for Deep Agent — state updates, tool calls, responses."""

import asyncio

import pytest

from actions.queue import ChatAction, SpeakAction


MEETING_CONTEXT = [
    (0,   "Davi", "Ok, lasst uns anfangen. Thema heute ist die API Migration."),
    (15,  "Max",  "Meine Sorge ist dass die v2 API breaking changes hat und wir alle Clients updaten muessen."),
    (30,  "Lisa", "Ich schlage Feature Flags vor, dann koennen wir schrittweise migrieren."),
    (45,  "Davi", "Gute Idee Lisa. Max, kannst du einen POC machen bis Freitag?"),
    (60,  "Max",  "Ja, mach ich."),
]


@pytest.mark.asyncio
async def test_deep_agent_updates_state_on_trigger(session):
    """When triggered, Deep Agent should update the shared state."""
    # Feed context
    await session.inject_conversation(MEETING_CONTEXT)

    # Trigger Nilo
    await session.inject_segment(
        speaker="Davi",
        text="Nilo, was hat Max zu den Breaking Changes gesagt?",
        timestamp=90.0,
    )

    # Wait for Deep Agent to complete (longer timeout — it does real LLM calls)
    await session.collector.wait_for_actions(count=2, timeout=30.0)

    # Check state was updated
    state = await session.shared_state.read()
    assert state["version"] > 0, "State was never updated"
    assert state["context_summary"], "context_summary is empty"


@pytest.mark.asyncio
async def test_deep_agent_responds_to_question(session):
    """Deep Agent should respond when asked a question about the meeting."""
    # Feed context
    await session.inject_conversation(MEETING_CONTEXT)

    # Ask Nilo a question
    await session.inject_segment(
        speaker="Davi",
        text="Nilo, fass mal zusammen was wir besprochen haben",
        timestamp=90.0,
    )

    # Wait for response actions (Quick-Ack + Deep Agent response)
    actions = await session.collector.wait_for_actions(count=2, timeout=30.0)

    # Should have at least Quick-Ack
    assert len(actions) >= 1, "No actions at all"

    # Check that actions contain meaningful content
    texts = [a.text for a in actions if hasattr(a, "text")]
    assert len(texts) > 0, "No text actions at all"

    # At least one action should be a natural response (Quick-Ack or Deep Agent)
    all_text = " ".join(texts).lower()
    assert len(all_text) > 10, f"Response too short: {texts}"

    # Check state was updated with meeting context
    state = await session.shared_state.read()
    if state["version"] > 0 and state.get("context_summary"):
        # Deep Agent ran and updated state — meeting content should be there
        summary = state["context_summary"].lower()
        has_content = any(
            kw in summary for kw in ["api", "migration", "breaking", "feature", "max", "lisa"]
        )
        assert has_content, f"State summary doesn't reference meeting: {state['context_summary']}"


@pytest.mark.asyncio
async def test_deep_agent_periodic_without_trigger(session):
    """Deep Agent periodic loop should update state even without triggers."""
    # Feed some context
    await session.inject_conversation(MEETING_CONTEXT[:3], delay=0.05)

    # Manually run one analysis cycle (don't wait for periodic timer)
    await session.deep_agent.analyze()

    state = await session.shared_state.read()
    assert state["version"] > 0, "State not updated after periodic analysis"
    assert state["context_summary"], "No context summary generated"
