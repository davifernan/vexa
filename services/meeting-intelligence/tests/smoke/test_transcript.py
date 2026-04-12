"""Smoke tests for Transcript Manager — buffer, archive, search."""

import asyncio

import pytest

from stt.provider import TranscriptSegment


@pytest.mark.asyncio
async def test_segments_added_to_live_window(session):
    """Segments should appear in the live window text."""
    seg1 = TranscriptSegment(text="Die API hat Breaking Changes.", speaker="Max", timestamp=10.0, is_final=True)
    seg2 = TranscriptSegment(text="Feature Flags loesen das.", speaker="Lisa", timestamp=20.0, is_final=True)

    await session.transcript_manager.add_segment(seg1)
    await session.transcript_manager.add_segment(seg2)

    window = session.transcript_manager.get_live_window()
    assert "Breaking Changes" in window
    assert "Feature Flags" in window
    assert "Max" in window
    assert "Lisa" in window


@pytest.mark.asyncio
async def test_search_transcript_finds_segments(session):
    """search_transcript should find matching segments by keyword."""
    await session.transcript_manager.add_segment(
        TranscriptSegment(text="Meine Sorge ist dass die API breaking changes hat.", speaker="Max", timestamp=30.0, is_final=True))
    await session.transcript_manager.add_segment(
        TranscriptSegment(text="Ich schlage Feature Flags vor.", speaker="Lisa", timestamp=45.0, is_final=True))
    await session.transcript_manager.add_segment(
        TranscriptSegment(text="Wie sieht es mit der Timeline aus?", speaker="Davi", timestamp=60.0, is_final=True))

    results = await session.transcript_manager.search("breaking changes")
    assert len(results) >= 1, "search_transcript didn't find 'breaking changes'"
    assert "breaking changes" in results[0]["text"].lower()


@pytest.mark.asyncio
async def test_search_transcript_no_results(session):
    """Search for something that wasn't said should return empty."""
    await session.transcript_manager.add_segment(
        TranscriptSegment(text="Alles gut soweit.", speaker="Max", timestamp=10.0, is_final=True))

    results = await session.transcript_manager.search("kubernetes deployment")
    assert len(results) == 0


@pytest.mark.asyncio
async def test_get_range_returns_correct_window(session):
    """get_range should return only segments within the time window."""
    await session.transcript_manager.add_segment(
        TranscriptSegment(text="Erstes Thema.", speaker="Max", timestamp=10.0, is_final=True))
    await session.transcript_manager.add_segment(
        TranscriptSegment(text="Zweites Thema.", speaker="Lisa", timestamp=60.0, is_final=True))
    await session.transcript_manager.add_segment(
        TranscriptSegment(text="Drittes Thema.", speaker="Davi", timestamp=120.0, is_final=True))

    results = await session.transcript_manager.get_range(50.0, 130.0)
    texts = [r["text"] for r in results]
    assert "Zweites Thema." in texts
    assert "Drittes Thema." in texts
    assert "Erstes Thema." not in texts
