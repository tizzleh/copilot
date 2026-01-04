from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, List


@dataclass
class TranscriptLine:
    text: str
    ts: float


class TranscriptBuffer:
    def __init__(self, max_entries: int = 5000):
        self.lines: Deque[TranscriptLine] = deque(maxlen=max_entries)

    def add_line(self, text: str) -> None:
        self.lines.append(TranscriptLine(text=text, ts=time.time()))

    def window_text(self, seconds: int) -> str:
        cutoff = time.time() - seconds
        recent = [line.text for line in self.lines if line.ts >= cutoff]
        return " \n".join(recent)

    def all_text(self) -> str:
        return " \n".join(line.text for line in self.lines)
