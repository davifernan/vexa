"""Smoke tests for Action Queue — async execution, no blocking."""

import asyncio

import pytest

from actions.queue import ChatAction, SpeakAction, ScreenAction


@pytest.mark.asyncio
async def test_chat_action_executes(session):
    """ChatAction should be collected by the mock."""
    await session.action_queue.push(ChatAction(
        text="Test message",
        platform="google_meet",
        meeting_id="test-001",
    ))

    action = await session.collector.wait_for_action(timeout=2.0)
    assert action is not None
    assert isinstance(action, ChatAction)
    assert action.text == "Test message"


@pytest.mark.asyncio
async def test_multiple_actions_execute_async(session):
    """Multiple actions pushed rapidly should all execute."""
    for i in range(5):
        await session.action_queue.push(ChatAction(
            text=f"Message {i}",
            platform="google_meet",
            meeting_id="test-001",
        ))

    actions = await session.collector.wait_for_actions(count=5, timeout=5.0)
    assert len(actions) == 5, f"Expected 5 actions, got {len(actions)}"


@pytest.mark.asyncio
async def test_speak_action_executes(session):
    """SpeakAction should work."""
    await session.action_queue.push(SpeakAction(
        text="Hallo, ich bin Nilo",
        platform="google_meet",
        meeting_id="test-001",
    ))

    action = await session.collector.wait_for_action(timeout=2.0)
    assert isinstance(action, SpeakAction)
    assert action.text == "Hallo, ich bin Nilo"
