# 🎬 Video Watcher

Streamlit app that turns any video (YouTube/Vimeo/TikTok/etc. URL, or local
upload) into a **timestamped, citation-grounded summary** — or answers a
custom question about it — using transcript text *and* sampled keyframes.

Built for one-click deploy to **Streamlit Community Cloud** so you can use
it from any machine.

## Features

- **Two modes**: structured summary, or ask-a-question Q&A
- **Quality presets**: Fast / Balanced / Thorough (controls keyframe count + output length)
- **Model picker**: Claude Sonnet 4 (default), Haiku 4.5 (cheap+fast), or Opus 4 (deepest)
- **Two Whisper providers**: Groq `whisper-large-v3-turbo` (fastest) or OpenAI `whisper-1`
- **Native captions first** when the source provides them (free, instant, accurate)
- **Section focus**: process only `[start, end]` of a long video
- **Keyframe gallery** with clickable timestamps that jump into the source YouTube video
- **Export**: summary (`.md`), transcript (`.txt`, timestamped `.txt`, `.srt`)
- **Debug tab**: stage timings + exact settings used

## Local dev

```powershell
cd "C:\Users\kevin\OneDrive - City of Brentwood\Documents\COWORK_MASTER\tools\video_watcher"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Option A: .env (loaded automatically)
copy .env.example .env

# Option B: Streamlit secrets (same format as Cloud)
copy .streamlit\secrets.toml.example .streamlit\secrets.toml

streamlit run app.py
```

On the first run you can also paste keys into the sidebar "API keys" expander.

## Deploy to Streamlit Community Cloud

1. **Push this folder to a GitHub repo** (the `.gitignore` already excludes
   `.env` and `.streamlit/secrets.toml`).
2. Go to <https://share.streamlit.io> → **New app** → pick your repo,
   branch, and `app.py`.
3. Open **Settings → Secrets** and paste:

   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   GROQ_API_KEY      = "gsk_..."
   OPENAI_API_KEY    = "sk-..."   # optional if you only use Groq Whisper
   ```

4. Deploy. The app picks up `packages.txt` (installs `ffmpeg` via apt) and
   `.streamlit/config.toml` (1 GB upload cap, dark theme).

That's it — the URL Streamlit gives you works from any computer.

## Notes & gotchas

- **YouTube + cloud IPs**: yt-dlp occasionally trips bot-detection from
  shared cloud egress. If a particular URL fails on Cloud but works locally,
  download it locally and use the upload tab instead.
- **Whisper 25 MB cap**: we transcode to 32 kbps mono mp3, which fits roughly
  100 minutes of audio. For longer videos, prefer URLs with native captions
  or use the section start/end controls.
- **Cost shape**: native captions are free; Groq Whisper is ~$0.04/hr; Claude
  Sonnet 4 with 12 keyframes + a 30-min transcript is roughly $0.05–$0.15
  per summary. The Debug tab shows you per-stage timings so you can tune.

## File layout

```
video_watcher/
├── app.py                            # Streamlit app
├── requirements.txt                  # pip deps
├── packages.txt                      # apt deps (ffmpeg) — used by Streamlit Cloud
├── .streamlit/
│   ├── config.toml                   # theme + upload cap
│   └── secrets.toml.example          # secret keys template
├── .env.example                      # local-dev secret keys template
├── .gitignore
└── README.md
```
