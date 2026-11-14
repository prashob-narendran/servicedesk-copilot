"""Sub-task 4: one-line ticket summary for the agent, written by the LLM from redacted text."""
from __future__ import annotations

_SYSTEM = (
    "You summarise customer-support tickets for service-desk agents. Write ONE sentence of at most {n} words "
    "stating the problem or request, who or what is affected, and any urgency. Keep masks such as [PERSON_1] "
    "exactly as written; never invent details. Output only the sentence."
)


class Summariser:
    def __init__(self, llm):
        self.llm = llm

    @property
    def model_name(self) -> str:
        return getattr(self.llm, "model", "llm")

    def summarise(self, subject: str, body: str, max_words: int = 25) -> str:
        text = f"Subject: {subject}\n\n{body}" if subject else body
        out = self.llm.complete(_SYSTEM.format(n=max_words), text, max_tokens=max_words * 3 + 20)
        out = " ".join(out.split()).strip().strip('"')
        words = out.split()
        if len(words) > max_words:
            out = " ".join(words[:max_words]).rstrip(",;:") + "…"
        return out
