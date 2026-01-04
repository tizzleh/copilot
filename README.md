# Hearing Copilot (local-first)

A lightweight FastAPI web app that pairs streaming transcripts with concise "coach cards" for a pro se litigant during a live remote hearing. It keeps knowledge-base data and logs fully local.

## Features
- Two-column UI: transcript feed (left) and coach cards (right) with a big pause toggle and "do not listen" control.
- Hotkeys: **Ctrl+Enter** sends the last transcript window; **Ctrl+P** pauses/resumes listening.
- Modes: manual push or auto (rolling window every few seconds, rate-limited).
- Local FAISS vector store seeded from `hearing_pack.md` and PDFs in `./sources/`.
- Local JSONL logging for transcripts and model outputs (`logs/interaction-log.jsonl`).
- Guardrails: concise (≤6 bullets) responses; citations only when present in KB, otherwise "No supporting authority found in your library."

## Setup
1. Install dependencies (Python 3.11+):
   ```bash
   pip install -r requirements.txt
   ```
2. Add your OpenAI API key to the environment if you want live LLM responses/embeddings:
   ```bash
   export OPENAI_API_KEY=sk-...
   ```
3. Prepare your local knowledge base:
   - Place `hearing_pack.md` at repo root and PDFs inside `./sources/`.
   - Run the ingestion script to build `data/kb_index.faiss` and metadata:
     ```bash
     python scripts/ingest_kb.py
     ```

## Running the app
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
Open http://localhost:8000 to view the UI.

### Controls
- **Pause Listening** button and **Do not listen** checkbox stop transcript intake (ignored at the server and excluded from the buffer).
- **Send last window (Ctrl+Enter)** pushes the latest transcript window to the coach.
- **Auto mode** checkbox enables the rolling 3–5s window pushes (interval from `config.yaml`).
- Manual transcript entry box lets you paste/type lines that would normally arrive from the transcriber feed.

### Configuration
Edit `config.yaml` to set `case_name`, `objectives`, `citation_style`, default `mode`, `window_seconds`, and `auto_interval_seconds`.

## Notes
- When OpenAI is unavailable, the app falls back to a deterministic, offline coach response while still honoring KB citations and logging.
- All logs stay in `logs/interaction-log.jsonl` (JSONL with timestamps).
