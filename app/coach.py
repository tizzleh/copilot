from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List

from openai import OpenAI

from .config import CONFIG
from .kb import KBStore

LOG_PATH = Path("logs/interaction-log.jsonl")


class CoachEngine:
    def __init__(self, kb: KBStore, prompt_path: str = "coach_prompt.txt") -> None:
        self.kb = kb
        self.prompt = Path(prompt_path).read_text(encoding="utf-8")
        self.client = None

    def _log(self, event: dict) -> None:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        event["ts"] = time.time()
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")

    def _maybe_client(self):
        if self.client is None:
            try:
                self.client = OpenAI()
            except Exception:
                self.client = False
        return self.client

    def _fallback_card(self, transcript: str, hits: List[dict]) -> dict:
        citation_lines = []
        for h in hits:
            citation_lines.append(
                f"• {h.get('source','unknown')} #{h.get('chunk_id')} — {h.get('location','')}"
            )
        if not citation_lines:
            citation_lines = ["No supporting authority found in your library."]

        bullets = [
            "Likely issue: answer the question asked; keep it short.",
            f"Outline: summarize key point in 2 sentences from recent statements: {transcript[:200]}...",
            "If unsure: ask for clarification respectfully.",
            *citation_lines,
        ][:6]
        return {
            "issue": bullets[0],
            "bullets": bullets,
            "citations": hits,
        }

    def generate_card(self, transcript: str) -> dict:
        hits = self.kb.search(transcript) if transcript.strip() else []
        client = self._maybe_client()
        if not client:
            card = self._fallback_card(transcript, hits)
            self._log({"type": "coach_card", "mode": "fallback", "card": card})
            return card

        context = "\n\n".join(
            [f"Chunk {h['chunk_id']} ({h['source']} {h.get('location','')}):\n{h['text']}" for h in hits]
        )
        objectives = "\n".join(CONFIG.objectives)
        system_prompt = self.prompt
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Case: {CONFIG.case_name}\nObjectives:\n{objectives}\n"
                    f"Recent transcript:\n{transcript}\n\nKB Context:\n{context or 'None'}\n"
                    "Respond with max 6 bullets and include chunk ids for every citation."
                ),
            },
        ]

        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.2,
            )
            content = resp.choices[0].message.content
            card = {
                "issue": content.split("\n")[0] if content else "",
                "bullets": content.split("\n"),
                "citations": hits,
            }
            self._log({"type": "coach_card", "mode": "llm", "card": card, "prompt": messages})
            return card
        except Exception:
            card = self._fallback_card(transcript, hits)
            self._log({"type": "coach_card", "mode": "fallback", "card": card})
            return card
