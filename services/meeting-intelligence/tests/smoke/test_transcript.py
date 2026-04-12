"""Smoke tests for Transcript Manager — buffer, archive, search."""

import pytest

from stt.provider import TranscriptSegment
from state.transcript_manager import TranscriptManager


@pytest.mark.asyncio
async def test_segments_added_to_live_window(session):
    """Segments should appear in the live window text."""
    await session.inject_segment("Max", "Die API hat Breaking Changes.", timestamp=10.0)
    await session.inject_segment("Lisa", "Feature Flags loesen das.", timestamp=20.0)

    window = session.transcript_manager.get_live_window()
    assert "Breaking Changes" in window
    assert "Feature Flags" in window
    assert "Max" in window
    assert "Lisa" in window


@pytest.mark.asyncio
async def test_search_transcript_finds_segments(session):
    """search_transcript should find matching segments by keyword."""
    await session.inject_segment("Max", "Meine Sorge ist dass die API breaking changes hat.", timestamp=30.0)
    await session.inject_segment("Lisa", "Ich schlage Feature Flags vor.", timestamp=45.0)
    await session.inject_segment("Davi", "Wie sieht es mit der Timeline aus?", timestamp=60.0)

    results = await session.transcript_manager.search("breaking changes")
    assert len(results) >= 1, "search_transcript didn't find 'breaking changes'"
    assert "breaking changes" in results[0]["text"].lower()


@pytest.mark.asyncio
async def test_search_transcript_no_results(session):
    """Search for something that wasn't said should return empty."""
    await session.inject_segment("Max", "Alles gut soweit.", timestamp=10.0)

    results = await session.transcript_manager.search("kubernetes deployment")
    assert len(results) == 0


@pytest.mark.asyncio
async def test_get_range_returns_correct_window(session):
    """get_range should return only segments within the time window."""
    await session.inject_segment("Max", "Erstes Thema.", timestamp=10.0)
    await session.inject_segment("Lisa", "Zweites Thema.", timestamp=60.0)
    await session.inject_segment("Davi", "Drittes Thema.", timestamp=120.0)

    results = await session.transcript_manager.get_range(50.0, 130.0)
    texts = [r["text"] for r in results]
    assert "Zweites Thema." in texts
    assert "Drittes Thema." in texts
    assert "Erstes Thema." not in texts
