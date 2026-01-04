from __future__ import annotations

import yaml
from pathlib import Path
from pydantic import BaseModel, Field


class AppConfig(BaseModel):
    case_name: str = "Untitled Hearing"
    objectives: list[str] = Field(default_factory=list)
    citation_style: str = "NM Bluebook"
    mode: str = "manual"
    window_seconds: int = 20
    auto_interval_seconds: int = 4

    @classmethod
    def load(cls, path: str | Path = "config.yaml") -> "AppConfig":
        config_path = Path(path)
        if not config_path.exists():
            return cls()
        with config_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)


CONFIG = AppConfig.load()
