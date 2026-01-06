# Mock Hearing Workflow (YouTube replay)

Use this guide to dry-run Hearing Copilot against a recorded hearing (e.g., on YouTube) before you attend your own.

## 1) Fork and set up locally
1. Fork this repo on GitHub and clone your fork.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. (Optional) Export your OpenAI key for live LLM/embedding support:
   ```bash
   export OPENAI_API_KEY=sk-...
   ```

## 2) Prepare a mock hearing pack
1. Pick a public hearing video you can watch end-to-end.
2. Copy `hearing_pack_template.md` to `hearing_pack.md` in the repo root and fill it out with that hearing's facts, motions, and authorities.
3. Add any PDFs that support your notes to `./sources/` with clear filenames referenced inside `hearing_pack.md`.

## 3) Build the knowledge base
Run the ingestion script so the coach can cite your materials:
```bash
python scripts/ingest_kb.py
```
This writes `data/kb_index.faiss` and metadata files locally.

## 4) Start the app
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
Open http://localhost:8000 in your browser.

## 5) Replay the hearing as a transcript feed
- Play the YouTube video and type/paste short snippets into the "Manual transcript" box every few seconds, or enable Auto mode to push rolling windows.
- Use **Ctrl+Enter** to send the last window or **Ctrl+P** (Pause) when the video is paused.
- Keep the snippets concise and chronological; the coach cards will refresh after each push.

## 6) Compare to the video
- Note where the coach cards match or miss issues raised by the judge or opposing counsel.
- Update `hearing_pack.md` with stronger citations or rebuttals, rerun `python scripts/ingest_kb.py`, and replay sections as needed.

## 7) Resetting between trials
- Replace `hearing_pack.md` with a new copy from the template for each new mock hearing.
- Clear or version your `logs/interaction-log.jsonl` if you want a fresh log for the next run.

