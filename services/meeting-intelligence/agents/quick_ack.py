"""Quick-Ack agent — instant natural confirmation via fastest-TTFT LLM."""

import logging

import anthropic
import openai

import config
from actions.queue import ActionQueue, ChatAction, SpeakAction

logger = logging.getLogger("meeting_intelligence.quick_ack")


class QuickAck:
    """Generates a 1-sentence natural acknowledgment as fast as possible.

    Needs almost no context — just the trigger sentence.
    Uses the model with the fastest Time-to-First-Token.
    """

    def __init__(self, action_queue: ActionQueue, platform: str, meeting_id: str):
        self.action_queue = action_queue
        self.platform = platform
        self.meeting_id = meeting_id

        if config.QUICK_ACK_PROVIDER == "anthropic":
            self._anthropic = anthropic.AsyncAnthropic(api_key=config.ANTHROPIC_API_KEY)
        else:
            self._openai = openai.AsyncOpenAI(api_key=config.OPENAI_API_KEY)

    async def respond(self, trigger_text: str, channel: str = "chat") -> None:
        """Generate and send a quick acknowledgment.

        Args:
            trigger_text: What the user said (the trigger).
            channel: "chat" or "speak" — how to respond.
        """
        try:
            response = await self._generate(trigger_text)

            if channel == "speak":
                action = SpeakAction(
                    text=response,
                    platform=self.platform,
                    meeting_id=self.meeting_id,
                )
            else:
                action = ChatAction(
                    text=response,
                    platform=self.platform,
                    meeting_id=self.meeting_id,
                )

            await self.action_queue.push(action)
            logger.info(f"Quick-ack ({channel}): {response}")

        except Exception as e:
            logger.error(f"Quick-ack failed: {e}")

    async def _generate(self, trigger_text: str) -> str:
        """Call the fastest LLM for a 1-sentence response."""
        system = (
            "Du bist Nilo, ein freundlicher Meeting-Assistent. "
            "Generiere eine kurze natuerliche Bestaetigung (max 10 Woerter, Deutsch). "
            "Keine Emojis. Kein Smalltalk. Nur die Bestaetigung."
        )
        prompt = f"Der Teilnehmer sagte: \"{trigger_text}\""

        if config.QUICK_ACK_PROVIDER == "anthropic":
            resp = await self._anthropic.messages.create(
                model=config.QUICK_ACK_MODEL,
                max_tokens=30,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text.strip()
        else:
            resp = await self._openai.chat.completions.create(
                model=config.QUICK_ACK_MODEL,
                max_tokens=30,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            )
            return resp.choices[0].message.content.strip()
