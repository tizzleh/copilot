from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request

from .coach import CoachEngine, LOG_PATH
from .config import CONFIG
from .kb import KBStore
from .transcript import TranscriptBuffer

app = FastAPI(title="Hearing Copilot")

kb_store = KBStore()
coach = CoachEngine(kb_store)
transcript_buffer = TranscriptBuffer()

app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")


class ConnectionManager:
    def __init__(self):
        self.active: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active:
            self.active.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in list(self.active):
            try:
                await connection.send_text(json.dumps(message))
            except Exception:
                self.disconnect(connection)


manager = ConnectionManager()

state = {
    "mode": CONFIG.mode,
    "window_seconds": CONFIG.window_seconds,
    "auto_interval_seconds": CONFIG.auto_interval_seconds,
    "paused": False,
    "do_not_listen": False,
}


def log_event(event: dict):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    event["ts"] = asyncio.get_event_loop().time()
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "case_name": CONFIG.case_name,
            "mode": state["mode"],
        },
    )


@app.get("/health")
async def health():
    return {"status": "ok", "mode": state}


async def auto_loop():
    while True:
        await asyncio.sleep(state["auto_interval_seconds"])
        if state.get("paused") or state.get("do_not_listen") or state.get("mode") != "auto":
            continue
        text = transcript_buffer.window_text(state["window_seconds"])
        if not text.strip():
            continue
        card = coach.generate_card(text)
        await manager.broadcast({"type": "coach_card", "card": card})


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(auto_loop())


@app.websocket("/ws/client")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        await websocket.send_text(
            json.dumps({"type": "state", "state": state, "case_name": CONFIG.case_name})
        )
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            msg_type = message.get("type")

            if msg_type == "transcript":
                text = message.get("text", "")
                if state.get("paused") or state.get("do_not_listen"):
                    log_event({"type": "transcript_ignored", "text": text, "state": state})
                    continue
                transcript_buffer.add_line(text)
                log_event({"type": "transcript", "text": text})
                await manager.broadcast({"type": "transcript", "text": text})

            elif msg_type == "manual_push":
                window = int(message.get("window_seconds", state["window_seconds"]))
                text = transcript_buffer.window_text(window)
                card = coach.generate_card(text)
                await manager.broadcast({"type": "coach_card", "card": card})

            elif msg_type == "set_mode":
                state["mode"] = message.get("mode", state["mode"])
                state["window_seconds"] = int(
                    message.get("window_seconds", state["window_seconds"])
                )
                state["auto_interval_seconds"] = int(
                    message.get("auto_interval_seconds", state["auto_interval_seconds"])
                )
                await manager.broadcast({"type": "state", "state": state})

            elif msg_type == "toggle_pause":
                state["paused"] = bool(message.get("paused"))
                await manager.broadcast({"type": "state", "state": state})

            elif msg_type == "set_do_not_listen":
                state["do_not_listen"] = bool(message.get("do_not_listen"))
                await manager.broadcast({"type": "state", "state": state})

            elif msg_type == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        manager.disconnect(websocket)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
