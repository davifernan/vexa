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

    # Check that at least one action contains meaningful content
    texts = [a.text for a in actions if hasattr(a, "text")]
    all_text = " ".join(texts).lower()

    # The response should reference something from the meeting
    assert any(
        keyword in all_text
        for keyword in ["api", "migration", "breaking", "feature flag", "max", "lisa", "poc"]
    ), f"Response doesn't reference meeting content: {texts}"


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
