"""Async HTTP client for Vexa Bot REST API."""

import logging

import httpx

logger = logging.getLogger("meeting_intelligence.vexa_client")


class VexaClient:
    """Talks to the Vexa API Gateway to control the bot."""

    def __init__(self, base_url: str, api_key: str):
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._http: httpx.AsyncClient | None = None

    async def startup(self) -> None:
        self._http = httpx.AsyncClient(
            base_url=self._base_url,
            headers={"X-API-Key": self._api_key, "Content-Type": "application/json"},
            timeout=10.0,
        )

    async def shutdown(self) -> None:
        if self._http:
            await self._http.aclose()

    async def speak(self, platform: str, meeting_id: str, text: str,
                    provider: str = "piper", voice: str = "alloy") -> bool:
        """Make the bot speak in the meeting via TTS."""
        try:
            resp = await self._http.post(
                f"/bots/{platform}/{meeting_id}/speak",
                json={"text": text, "provider": provider, "voice": voice},
            )
            if resp.status_code in (200, 202):
                logger.info(f"Bot speaking: {text[:50]}...")
                return True
            logger.warning(f"Speak failed: {resp.status_code} {resp.text}")
            return False
        except Exception as e:
            logger.error(f"Speak error: {e}")
            return False

    async def chat(self, platform: str, meeting_id: str, text: str) -> bool:
        """Send a chat message in the meeting."""
        try:
            resp = await self._http.post(
                f"/bots/{platform}/{meeting_id}/chat",
                json={"text": text},
            )
            if resp.status_code in (200, 202):
                logger.info(f"Bot chat: {text[:50]}...")
                return True
            logger.warning(f"Chat failed: {resp.status_code} {resp.text}")
            return False
        except Exception as e:
            logger.error(f"Chat error: {e}")
            return False

    async def stop_speaking(self, platform: str, meeting_id: str) -> bool:
        """Interrupt bot speech (barge-in)."""
        try:
            resp = await self._http.delete(f"/bots/{platform}/{meeting_id}/speak")
            return resp.status_code in (200, 202)
        except Exception as e:
            logger.error(f"Stop speaking error: {e}")
            return False

    async def screen_share(self, platform: str, meeting_id: str,
                           content_url: str) -> bool:
        """Share screen content (image/URL on bot camera)."""
        try:
            resp = await self._http.post(
                f"/bots/{platform}/{meeting_id}/screen",
                json={"url": content_url},
            )
            return resp.status_code in (200, 202)
        except Exception as e:
            logger.error(f"Screen share error: {e}")
            return False

    async def stop_screen_share(self, platform: str, meeting_id: str) -> bool:
        try:
            resp = await self._http.delete(f"/bots/{platform}/{meeting_id}/screen")
            return resp.status_code in (200, 202)
        except Exception as e:
            logger.error(f"Stop screen share error: {e}")
            return False
