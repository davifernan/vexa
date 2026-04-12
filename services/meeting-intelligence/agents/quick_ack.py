"""Quick-Ack agent — instant natural confirmation via fastest-TTFT LLM."""

import logging

from llm.provider import LLMProvider
from actions.queue import ActionQueue, ChatAction, SpeakAction

logger = logging.getLogger("meeting_intelligence.quick_ack")


class QuickAck:
    """Generates a 1-sentence natural acknowledgment as fast as possible.

    Needs almost no context — just the trigger sentence.
    Uses the provider with the fastest Time-to-First-Token.
    Provider is injected — can be Claude CLI (Max, $0), Codex ($0), or API.
    """

    def __init__(self, llm: LLMProvider, action_queue: ActionQueue,
                 platform: str, meeting_id: str):
        self.llm = llm
        self.action_queue = action_queue
        self.platform = platform
        self.meeting_id = meeting_id

    async def respond(self, trigger_text: str, channel: str = "chat") -> None:
        """Generate and send a quick acknowledgment.

        Args:
            trigger_text: What the user said (the trigger).
            channel: "chat" or "speak" — how to respond.
        """
        try:
            resp = await self.llm.generate(
                system=(
                    "Du bist Nilo, ein freundlicher Meeting-Assistent. "
                    "Generiere eine kurze natuerliche Bestaetigung (max 10 Woerter, Deutsch). "
                    "Keine Emojis. Kein Smalltalk. Nur die Bestaetigung."
                ),
                prompt=f"Der Teilnehmer sagte: \"{trigger_text}\"",
                max_tokens=30,
            )

            if channel == "speak":
                action = SpeakAction(
                    text=resp.text,
                    platform=self.platform,
                    meeting_id=self.meeting_id,
                )
            else:
                action = ChatAction(
                    text=resp.text,
                    platform=self.platform,
                    meeting_id=self.meeting_id,
                )

            await self.action_queue.push(action)
            logger.info(f"Quick-ack ({channel}, via {self.llm.name}): {resp.text}")

        except Exception as e:
            logger.error(f"Quick-ack failed: {e}")
