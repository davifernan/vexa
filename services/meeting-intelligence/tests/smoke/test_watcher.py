"""Smoke tests for Watcher agent — keyword detection and trigger flow."""

import asyncio

import pytest

from actions.queue import ChatAction, SpeakAction


@pytest.mark.asyncio
async def test_keyword_nilo_triggers_quick_ack(session):
    """When someone says 'Nilo', Quick-Ack must respond with a chat message."""
    await session.inject_segment(
        speaker="Davi",
        text="Nilo, was hat Max vorhin gesagt?",
        timestamp=90.0,
    )

    action = await session.collector.wait_for_action(timeout=30.0)
    assert action is not None, "No Quick-Ack response within 30s"
    assert isinstance(action, ChatAction), f"Expected ChatAction, got {type(action).__name__}"
    assert len(action.text) > 0, "Quick-Ack text is empty"
    assert len(action.text.split()) <= 20, f"Quick-Ack too long: {action.text}"


@pytest.mark.asyncio
async def test_keyword_hey_nilo_triggers(session):
    """'Hey Nilo' should also trigger."""
    await session.inject_segment(
        speaker="Lisa",
        text="Hey Nilo, kannst du das zusammenfassen?",
        timestamp=120.0,
    )

    action = await session.collector.wait_for_action(timeout=10.0)
    assert action is not None, "No Quick-Ack for 'Hey Nilo'"
    assert isinstance(action, ChatAction)


@pytest.mark.asyncio
async def test_no_trigger_on_normal_speech(session):
    """Normal conversation without keywords must NOT trigger anything."""
    await session.inject_segment(
        speaker="Max",
        text="Ich denke wir sollten die API Migration priorisieren.",
        timestamp=30.0,
    )

    action = await session.collector.wait_for_action(timeout=3.0)
    assert action is None, f"False trigger! Got: {action}"


@pytest.mark.asyncio
async def test_no_trigger_on_interim_segment(session):
    """Interim (non-final) segments should not trigger anything."""
    await session.inject_segment(
        speaker="Davi",
        text="Nilo, was",
        timestamp=90.0,
        is_final=False,  # Not final!
    )

    action = await session.collector.wait_for_action(timeout=3.0)
    assert action is None, "Interim segment triggered an action!"


@pytest.mark.asyncio
async def test_cooldown_prevents_double_trigger(session):
    """Two triggers within 5s — only the first should fire."""
    await session.inject_segment(
        speaker="Davi",
        text="Nilo, erster Trigger",
        timestamp=90.0,
    )

    first = await session.collector.wait_for_action(timeout=10.0)
    assert first is not None, "First trigger didn't fire"

    session.collector.clear()

    # Second trigger 2 seconds later (within cooldown)
    await session.inject_segment(
        speaker="Davi",
        text="Nilo, zweiter Trigger",
        timestamp=92.0,
    )

    second = await session.collector.wait_for_action(timeout=3.0)
    assert second is None, "Second trigger fired within cooldown!"
