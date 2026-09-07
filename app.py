"""
Video Watcher — Streamlit app that summarizes / answers questions about videos.

Pipeline:
    URL or file upload
      -> yt-dlp download (+ native captions if available)
      -> ffmpeg keyframe extraction
      -> transcript: native captions first, then Whisper (Groq or OpenAI)
      -> Claude (multimodal): timestamped summary or answer to a user question

Designed to deploy to Streamlit Community Cloud:
  - ffmpeg from packages.txt (system apt)
  - secrets from .streamlit/secrets.toml or the Cloud "Secrets" pane
  - imageio-ffmpeg as a fallback when system ffmpeg is missing (local dev)
"""

from __future__ import annotations

import base64
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import streamlit as st
from dotenv import load_dotenv

import anthropic
import yt_dlp
import webvtt
from openai import OpenAI

# Groq is optional — we only construct the client if a key is present.
try:
    from groq import Groq  # type: ignore
except ImportError:  # pragma: no cover
    Groq = None  # type: ignore


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# override=True: a local .env should always win over a stale OS-level env var
# (e.g. a system OPENAI_API_KEY set for other tools) — otherwise editing .env
# silently has no effect for as long as that variable is already set.
load_dotenv(override=True)

# Model menus. Display label -> API model id.
CLAUDE_MODELS = {
    "Claude Sonnet 5 (balanced — recommended)": "claude-sonnet-5",
    "Claude Haiku 4.5 (fast & cheap)": "claude-haiku-4-5-20251001",
    "Claude Opus 4.8 (deepest analysis)": "claude-opus-4-8",
}

# Whisper providers. Groq's whisper-large-v3-turbo is ~10x faster than OpenAI for the same quality tier.
WHISPER_PROVIDERS = {
    "Groq · whisper-large-v3-turbo (fastest)": ("groq", "whisper-large-v3-turbo"),
    "Groq · whisper-large-v3 (most accurate)": ("groq", "whisper-large-v3"),
    "OpenAI · whisper-1": ("openai", "whisper-1"),
}

# Quality presets drive keyframe count + Claude max_tokens. Sliders override.
QUALITY_PRESETS = {
    "Fast":     {"frames": 6,  "max_tokens": 1500},
    "Balanced": {"frames": 12, "max_tokens": 2500},
    "Thorough": {"frames": 20, "max_tokens": 4000},
}

WHISPER_BITRATE = "32k"   # mono mp3; keeps us comfortably under the 25 MB upload cap
WHISPER_MAX_BYTES = 24 * 1024 * 1024


def _resolve_ffmpeg() -> str:
    """Prefer system ffmpeg (apt-installed on Streamlit Cloud); fall back to imageio bundle."""
    on_path = shutil.which("ffmpeg")
    if on_path:
        return on_path
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


FFMPEG = _resolve_ffmpeg()


# ---------------------------------------------------------------------------
# Secrets helper — st.secrets first, env var fallback
# ---------------------------------------------------------------------------

def secret(name: str, default: str = "") -> str:
    try:
        if name in st.secrets:  # type: ignore[operator]
            return str(st.secrets[name])
    except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):  # type: ignore[attr-defined]
        pass
    except Exception:
        pass
    return os.getenv(name, default)


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def run(cmd: list[str]) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({Path(cmd[0]).name}):\n{proc.stderr or proc.stdout}")
    return proc


def fmt_ts(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def parse_ts(text: str) -> float | None:
    """Accept '90', '1:30', '01:23:45', or '1m30s' style; return seconds or None."""
    text = (text or "").strip()
    if not text:
        return None
    if text.replace(".", "", 1).isdigit():
        return float(text)
    if ":" in text:
        parts = [float(p) for p in text.split(":")]
        if len(parts) == 2:
            return parts[0] * 60 + parts[1]
        if len(parts) == 3:
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
    m = re.fullmatch(r"(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?", text)
    if m and any(m.groups()):
        h, mi, s = (int(x) if x else 0 for x in m.groups())
        return h * 3600 + mi * 60 + s
    return None


def ffprobe_duration(path: Path) -> float:
    proc = subprocess.run([FFMPEG, "-i", str(path)], capture_output=True, text=True)
    m = re.search(r"Duration:\s+(\d+):(\d+):(\d+\.\d+)", proc.stderr)
    if not m:
        return 0.0
    h, mm, s = m.groups()
    return int(h) * 3600 + int(mm) * 60 + float(s)


def youtube_id(url: str) -> str | None:
    """Extract the YouTube video id so we can link timestamped citations back to the source."""
    m = re.search(r"(?:v=|youtu\.be/|/shorts/|/embed/)([A-Za-z0-9_-]{11})", url)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Download (URL path)
# ---------------------------------------------------------------------------

@dataclass
class Downloaded:
    video_path: Path
    sub_path: Path | None
    info: dict


def _cookie_file(workdir: Path) -> str | None:
    """If a YT_COOKIES secret is set (Netscape format), write it to a temp file."""
    raw = secret("YT_COOKIES")
    if not raw:
        return None
    p = workdir / "cookies.txt"
    p.write_text(raw, encoding="utf-8")
    return str(p)


def download_video(url: str, workdir: Path, prefer_low_res: bool = True) -> Downloaded:
    outtmpl = str(workdir / "video.%(ext)s")
    # Each "/" is a fallback. YouTube usually serves separate video+audio streams;
    # bv*+ba handles that. `b` and `best` cover combined-stream single files.
    if prefer_low_res:
        fmt = "bv*[height<=720]+ba/b[height<=720]/bv*+ba/best[height<=720]/b/best"
    else:
        fmt = "bv*+ba/b/best"
    base_opts: dict = {
        "outtmpl": outtmpl,
        "format": fmt,
        "merge_output_format": "mp4",
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["en", "en-US", "en-GB"],
        "subtitlesformat": "vtt",
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        # Looks more like a real browser; reduces 403s on shared cloud egress.
        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/127.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        },
    }
    cookie_path = _cookie_file(workdir)
    if cookie_path:
        base_opts["cookiefile"] = cookie_path

    # Probe metadata with each client until one returns a non-empty format list.
    # `process=False` skips format selection so we see the raw manifest. (Plain
    # `download=False` still applies the format string and would fail the same way.)
    client_attempts = [None, ["mweb"], ["android"], ["ios"], ["tv_embedded"], ["web"]]

    info = None
    probe_log: list[str] = []
    chosen_clients: list[str] | None = None

    for clients in client_attempts:
        probe_opts = dict(base_opts)
        probe_opts.pop("format", None)  # ensure no format constraint during probe
        if clients is not None:
            probe_opts["extractor_args"] = {"youtube": {"player_client": clients}}
        label = "default" if clients is None else "+".join(clients)
        try:
            with yt_dlp.YoutubeDL(probe_opts) as probe:
                meta = probe.extract_info(url, download=False, process=False)
            formats = meta.get("formats") or []
            probe_log.append(f"{label}: {len(formats)} formats")
            if formats:
                info = meta
                chosen_clients = clients
                break
        except Exception as e:
            probe_log.append(f"{label}: {type(e).__name__}: {e}")
            continue

    if info is None:
        hint = (
            "\n\nNo client returned any downloadable formats. Likely causes:\n"
            "• Video is members-only, age-restricted, region-locked, or a live stream.\n"
            "• Your cookies don't include access to this video (try a different YouTube account).\n"
            "• Or: download the video locally and use the **Upload file** tab.\n\n"
            "Probe log:\n  " + "\n  ".join(probe_log)
        )
        raise RuntimeError(f"yt-dlp could not access this video.{hint}")

    # Build a download-time format string. Start with the user preference, then add
    # a guaranteed-resolvable last resort: the highest-numbered *real* format_id we
    # just saw. Storyboard formats ("sb0", "sb1"...) are picture-grid previews, not
    # video — they sort last in YouTube's format list and always fail if picked.
    fmt_ids = [
        f.get("format_id") for f in info["formats"]
        if f.get("format_id")
        and not str(f.get("format_id", "")).startswith("sb")
        and (f.get("format_note") or "").lower() != "storyboard"
        and f.get("ext") != "mhtml"
    ]
    last_resort = fmt_ids[-1] if fmt_ids else "best"
    chained_fmt = f"{fmt}/{last_resort}"

    dl_opts = dict(base_opts)
    if chosen_clients is not None:
        dl_opts["extractor_args"] = {"youtube": {"player_client": chosen_clients}}
    dl_opts["format"] = chained_fmt

    try:
        with yt_dlp.YoutubeDL(dl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
    except Exception as e:
        raise RuntimeError(
            f"yt-dlp download failed despite finding {len(info.get('formats', []))} formats.\n"
            f"Tried format string: {chained_fmt!r}\n"
            f"Error: {e}\n\n"
            f"Fallback: download the video locally and use the **Upload file** tab."
        ) from e

    video_path = None
    for p in workdir.iterdir():
        if p.stem == "video" and p.suffix.lower() in {".mp4", ".mkv", ".webm"}:
            video_path = p
            break
    if video_path is None:
        rd = info.get("requested_downloads") or []
        if rd and rd[0].get("filepath"):
            video_path = Path(rd[0]["filepath"])
    if video_path is None:
        raise RuntimeError("yt-dlp finished but no video file was produced.")

    sub_path = next(iter(workdir.glob("video*.vtt")), None)
    return Downloaded(video_path=video_path, sub_path=sub_path, info=info)


# ---------------------------------------------------------------------------
# Clip a section (when user supplies start/end)
# ---------------------------------------------------------------------------

def clip_section(src: Path, workdir: Path, start: float | None, end: float | None) -> Path:
    if start is None and end is None:
        return src
    out = workdir / "clip.mp4"
    cmd = [FFMPEG, "-y"]
    if start is not None:
        cmd += ["-ss", f"{start:.2f}"]
    cmd += ["-i", str(src)]
    if end is not None:
        # -to here is end timestamp relative to the seek point if -ss came before -i;
        # so use -t (duration) for correctness across both branches.
        duration = end - (start or 0)
        cmd += ["-t", f"{duration:.2f}"]
    cmd += ["-c:v", "libx264", "-c:a", "aac", "-preset", "veryfast", str(out)]
    run(cmd)
    return out


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def extract_keyframes(video_path: Path, duration: float, n_frames: int, workdir: Path) -> list[tuple[float, Path]]:
    if duration <= 0 or n_frames < 1:
        return []
    n = min(n_frames, max(1, int(duration)))
    step = duration / (n + 1)
    timestamps = [step * (i + 1) for i in range(n)]

    frames_dir = workdir / "frames"
    frames_dir.mkdir(exist_ok=True)

    out: list[tuple[float, Path]] = []
    for i, ts in enumerate(timestamps):
        outpath = frames_dir / f"frame_{i:03d}.jpg"
        run([
            FFMPEG, "-y", "-ss", f"{ts:.2f}", "-i", str(video_path),
            "-frames:v", "1", "-q:v", "4", "-vf", "scale=768:-2",
            str(outpath),
        ])
        out.append((ts, outpath))
    return out


# ---------------------------------------------------------------------------
# Transcript
# ---------------------------------------------------------------------------

def transcript_from_vtt(vtt_path: Path) -> str:
    lines: list[str] = []
    for cap in webvtt.read(str(vtt_path)):
        ts = cap.start.split(".")[0]
        text = cap.text.replace("\n", " ").strip()
        if text:
            lines.append(f"[{ts}] {text}")
    # Dedup adjacent identical lines (YouTube auto-captions repeat partials).
    deduped: list[str] = []
    for line in lines:
        body = line.split("] ", 1)[-1]
        if not deduped or deduped[-1].split("] ", 1)[-1] != body:
            deduped.append(line)
    return "\n".join(deduped)


def transcribe_audio(video_path: Path, workdir: Path, provider: str, model: str, language: str | None) -> str:
    """Extract audio and send to Groq or OpenAI Whisper. Returns timestamped transcript text."""
    audio_path = workdir / "audio.mp3"
    run([
        FFMPEG, "-y", "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", "16000",
        "-b:a", WHISPER_BITRATE,
        str(audio_path),
    ])

    size = audio_path.stat().st_size
    if size > WHISPER_MAX_BYTES:
        raise RuntimeError(
            f"Extracted audio is {size/1e6:.1f} MB — over the 25 MB Whisper cap. "
            f"Use the section start/end controls to focus on part of the video, "
            f"or supply a video that has native captions."
        )

    if provider == "groq":
        if Groq is None:
            raise RuntimeError("groq package not installed.")
        key = secret("GROQ_API_KEY")
        if not key:
            raise RuntimeError("GROQ_API_KEY missing — add it to secrets or pick OpenAI.")
        client = Groq(api_key=key)
        with audio_path.open("rb") as f:
            result = client.audio.transcriptions.create(
                file=(audio_path.name, f.read()),
                model=model,
                response_format="verbose_json",
                language=language or None,
                timestamp_granularities=["segment"],
            )
        segments = getattr(result, "segments", None) or []
    else:
        key = secret("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY missing — add it to secrets or pick Groq.")
        client = OpenAI(api_key=key)
        with audio_path.open("rb") as f:
            result = client.audio.transcriptions.create(
                model=model,
                file=f,
                response_format="verbose_json",
                language=language or None,
            )
        segments = getattr(result, "segments", None) or []

    lines: list[str] = []
    for seg in segments:
        start = seg["start"] if isinstance(seg, dict) else seg.start
        text = (seg["text"] if isinstance(seg, dict) else seg.text).strip()
        if text:
            lines.append(f"[{fmt_ts(start)}] {text}")
    if not lines:
        return getattr(result, "text", "") or ""
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Claude summary / Q&A
# ---------------------------------------------------------------------------

SUMMARY_PROMPT = """\
You are summarizing a video titled: {title!r}{author} (duration {duration}).

Below are {n_frames} keyframes sampled across the video (each labeled with its timestamp), followed by the transcript with timestamps.

Produce a structured summary in markdown with these sections:
1. **TL;DR** — 2–3 sentences.
2. **Timestamped outline** — bullet list with `[HH:MM:SS]` anchors for each major topic or section change.
3. **Key visual moments** — bullets referencing the keyframe timestamps; describe what is actually visible (slides, demos, diagrams, people, scene changes).
4. **Notable quotes or takeaways** — 3–6 short bullets, each with a timestamp.

Cite timestamps for every claim. Be concrete and specific."""

QA_PROMPT = """\
You are answering a question about a video titled: {title!r}{author} (duration {duration}).

Below are {n_frames} keyframes sampled across the video (each labeled with its timestamp), followed by the transcript with timestamps.

The user's question:
{question}

Answer in markdown. Ground your answer in the transcript and keyframes — cite `[HH:MM:SS]` timestamps for each claim. If the video does not contain the answer, say so plainly and describe what *is* covered."""


def claude_call(
    transcript: str,
    keyframes: list[tuple[float, Path]],
    title: str,
    author: str,
    duration: float,
    model: str,
    max_tokens: int,
    question: str | None,
) -> str:
    key = secret("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY missing — add it to secrets.")

    client = anthropic.Anthropic(api_key=key)
    template = QA_PROMPT if question else SUMMARY_PROMPT
    header = template.format(
        title=title,
        author=f" by {author}" if author else "",
        duration=fmt_ts(duration),
        n_frames=len(keyframes),
        question=question or "",
    )

    content: list[dict] = [{"type": "text", "text": header}]
    for ts, path in keyframes:
        b64 = base64.standard_b64encode(path.read_bytes()).decode("ascii")
        content.append({"type": "text", "text": f"Keyframe @ [{fmt_ts(ts)}]:"})
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
        })
    content.append({"type": "text", "text": f"\nTranscript:\n\n{transcript}"})

    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": content}],
    )
    return "".join(b.text for b in resp.content if b.type == "text")


# ---------------------------------------------------------------------------
# Transcript export formats
# ---------------------------------------------------------------------------

def transcript_to_srt(transcript: str) -> str:
    """Convert our '[HH:MM:SS] text' lines into a rough SRT."""
    lines = [ln for ln in transcript.splitlines() if ln.strip().startswith("[")]
    out: list[str] = []
    for i, ln in enumerate(lines, 1):
        m = re.match(r"\[(\d\d:\d\d:\d\d)\]\s*(.*)", ln)
        if not m:
            continue
        start = m.group(1)
        # End at next line's start, or +5s for the last.
        if i < len(lines):
            nxt = re.match(r"\[(\d\d:\d\d:\d\d)\]", lines[i])
            end = nxt.group(1) if nxt else start
        else:
            end = start
        out.append(f"{i}\n{start},000 --> {end},000\n{m.group(2)}\n")
    return "\n".join(out)


def transcript_plain(transcript: str) -> str:
    return "\n".join(re.sub(r"^\[\d\d:\d\d:\d\d\]\s*", "", ln) for ln in transcript.splitlines())


def markdown_to_plain(md: str) -> str:
    """Strip common markdown syntax so the summary can be downloaded as clean plain text."""
    text = md
    text = re.sub(r"```[^\n]*\n?", "", text)              # code fence markers
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)  # headers
    text = re.sub(r"\*\*\*(.+?)\*\*\*", r"\1", text)       # bold+italic
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)           # bold
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"\1", text)  # italic
    text = re.sub(r"__(.+?)__", r"\1", text)               # bold (underscore)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", text)  # links
    text = re.sub(r"^\s*[-*+]\s+", "\u2022 ", text, flags=re.MULTILINE)  # bullets
    text = re.sub(r"\n{3,}", "\n\n", text)                 # collapse blank runs
    return text.strip()


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Video Watcher", page_icon="🎬", layout="wide")

# A bit of CSS polish — tighter padding, better cards, monospace transcript.
st.markdown(
    """
    <style>
      .block-container { padding-top: 2rem; padding-bottom: 4rem; max-width: 1200px; }
      h1 { font-weight: 700; letter-spacing: -0.02em; }
      .stTabs [data-baseweb="tab-list"] button { font-weight: 600; }
      .vw-meta { color: #9aa0b4; font-size: 0.9rem; }
      .vw-card {
        background: #161A23; border: 1px solid #232838; border-radius: 12px;
        padding: 1rem 1.25rem; margin-bottom: 0.75rem;
      }
      .vw-kpi { display: flex; gap: 1.5rem; margin: 0.25rem 0 1rem 0; }
      .vw-kpi div { background: #161A23; border: 1px solid #232838;
        border-radius: 10px; padding: 0.5rem 0.9rem; min-width: 110px; }
      .vw-kpi small { color: #9aa0b4; }
      pre, code { font-size: 0.85rem !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🎬 Video Watcher")
st.markdown(
    '<div class="vw-meta">Paste a URL or upload a file. Get a timestamped summary '
    "grounded in the transcript <i>and</i> the actual frames.</div>",
    unsafe_allow_html=True,
)

# ----- Sidebar: controls --------------------------------------------------------
with st.sidebar:
    st.markdown("### Output")
    quality = st.select_slider("Quality preset", options=list(QUALITY_PRESETS), value="Balanced")
    claude_label = st.selectbox("Claude model", list(CLAUDE_MODELS), index=0)

    mode = st.radio(
        "Mode",
        ["Structured summary", "Ask a question"],
        captions=["TL;DR + outline + key moments", "Custom Q&A grounded in the video"],
    )
    question = ""
    if mode == "Ask a question":
        question = st.text_area(
            "Your question",
            placeholder="e.g. What were the three main objections to the proposal?",
            height=80,
        )

    st.markdown("### Transcript")
    whisper_label = st.selectbox("Whisper provider", list(WHISPER_PROVIDERS), index=0)
    prefer_captions = st.toggle("Prefer native captions (free, faster)", value=True)
    language_hint = st.text_input("Language hint (optional)", value="", placeholder="en, es, fr…", max_chars=5)

    with st.expander("Advanced"):
        n_frames_override = st.slider("Keyframes", 4, 40, QUALITY_PRESETS[quality]["frames"])
        st.caption("More frames = better visual grounding, but slower + more tokens.")
        start_str = st.text_input("Start time", placeholder="0:00 or 1m30s")
        end_str = st.text_input("End time", placeholder="3:00 or empty")
        low_res = st.toggle("Cap download at 720p", value=True,
                            help="Faster downloads. Disable for high-detail visual analysis.")

    with st.expander("API keys"):
        st.caption(
            "Pulled from secrets/env automatically — configured keys are never shown here. "
            "Only fill a field below to override it for this session."
        )

        def _key_row(label: str, name: str) -> None:
            configured = "✓ configured" if secret(name) else "not set"
            val = st.text_input(
                label, value="", type="password",
                placeholder=configured, key=f"override_{name}",
            )
            if val:
                os.environ[name] = val

        _key_row("Anthropic", "ANTHROPIC_API_KEY")
        _key_row("OpenAI", "OPENAI_API_KEY")
        _key_row("Groq", "GROQ_API_KEY")

claude_model = CLAUDE_MODELS[claude_label]
whisper_provider, whisper_model = WHISPER_PROVIDERS[whisper_label]
preset = QUALITY_PRESETS[quality]
max_tokens = preset["max_tokens"]

# ----- Main input row ----------------------------------------------------------
tab_url, tab_file = st.tabs(["🔗 From URL", "⬆️ Upload file"])

with tab_url:
    url = st.text_input(
        "Video URL",
        placeholder="https://www.youtube.com/watch?v=…",
        label_visibility="collapsed",
    )
    go_url = st.button("Analyze video", type="primary", use_container_width=True, disabled=not url)

with tab_file:
    uploaded = st.file_uploader(
        "Drop a video (.mp4, .mov, .mkv, .webm)",
        type=["mp4", "mov", "mkv", "webm"],
        label_visibility="collapsed",
    )
    go_file = st.button("Analyze upload", type="primary", use_container_width=True,
                        disabled=uploaded is None, key="go_file")


# ----- Result cache ---------------------------------------------------------
# Streamlit reruns the whole script on every widget interaction, including
# download-button clicks. Without caching, that rerun would fall through to
# "no button was just clicked" and the results (and this section's downloads)
# would vanish — forcing a full re-download / re-transcribe / re-summarize to
# get them back. Stashing the finished pipeline output in session_state and
# rendering it unconditionally on every rerun fixes that.
if "vw_result" not in st.session_state:
    st.session_state.vw_result = None
if "vw_workdir" not in st.session_state:
    st.session_state.vw_workdir = None


def _cleanup_workdir() -> None:
    """Remove the previous run's temp dir (keyframe jpgs, audio, video) before starting a new one."""
    old = st.session_state.get("vw_workdir")
    if old and Path(old).exists():
        shutil.rmtree(old, ignore_errors=True)
    st.session_state.vw_workdir = None


# ----- Pipeline: compute only (no rendering) --------------------------------
def compute_pipeline(
    video_path: Path,
    workdir: Path,
    sub_path: Path | None,
    title: str,
    author: str,
    source_url: str | None,
    chapters: list[dict] | None,
) -> dict | None:
    if not secret("ANTHROPIC_API_KEY"):
        st.error("Anthropic API key required. Add it in the sidebar or secrets.")
        return None

    start = parse_ts(start_str)
    end = parse_ts(end_str)
    yt_id = youtube_id(source_url or "") if source_url else None

    timings: dict[str, float] = {}

    # Optional section clip.
    if start is not None or end is not None:
        t0 = time.time()
        with st.status("Clipping section…", expanded=False):
            video_path = clip_section(video_path, workdir, start, end)
            sub_path = None  # captions no longer aligned to clipped media
        timings["clip"] = time.time() - t0

    duration = ffprobe_duration(video_path)

    progress = st.progress(0.0, text="Starting…")

    # 1. Keyframes
    progress.progress(0.15, text="Extracting keyframes…")
    t0 = time.time()
    keyframes = extract_keyframes(video_path, duration, n_frames_override, workdir)
    timings["frames"] = time.time() - t0

    # 2. Transcript
    progress.progress(0.40, text="Building transcript…")
    transcript = ""
    transcript_source = "none"
    t0 = time.time()
    if prefer_captions and sub_path and sub_path.exists():
        try:
            transcript = transcript_from_vtt(sub_path)
            transcript_source = "native captions"
        except Exception as e:
            st.warning(f"Native-caption parse failed, falling back to Whisper: {e}")
    if not transcript.strip():
        try:
            transcript = transcribe_audio(
                video_path, workdir, whisper_provider, whisper_model,
                language_hint or None,
            )
            transcript_source = f"{whisper_provider} · {whisper_model}"
        except Exception as e:
            st.error(f"Transcription failed: {e}")
            return None
    timings["transcript"] = time.time() - t0

    # 3. Claude
    progress.progress(0.75, text=f"Calling {claude_label.split(' (')[0]}…")
    t0 = time.time()
    try:
        result_md = claude_call(
            transcript=transcript,
            keyframes=keyframes,
            title=title,
            author=author,
            duration=duration,
            model=claude_model,
            max_tokens=max_tokens,
            question=question.strip() if mode == "Ask a question" else None,
        )
    except Exception as e:
        st.error(f"Claude call failed: {e}")
        return None
    timings["claude"] = time.time() - t0
    progress.progress(1.0, text="Done.")
    progress.empty()

    return {
        "title": title,
        "src_label": "URL" if source_url else "Upload",
        "claude_label": claude_label,
        "duration": duration,
        "result_md": result_md,
        "transcript": transcript,
        "transcript_source": transcript_source,
        "keyframes": [(ts, str(p)) for ts, p in keyframes],
        "chapters": chapters,
        "yt_id": yt_id,
        "timings": timings,
        "settings": {
            "quality": quality,
            "claude_model": claude_model,
            "whisper": f"{whisper_provider}/{whisper_model}",
            "keyframes": len(keyframes),
            "language_hint": language_hint or None,
            "section": [start, end],
            "low_res": low_res,
        },
    }


# ----- Rendering: draws from cached state, runs on every rerun --------------
def render_results(result: dict) -> None:
    title = result["title"]
    out_name = re.sub(r"[^A-Za-z0-9_-]+", "_", title)[:60] or "summary"

    kpi = f"""
        <div class="vw-kpi">
          <div><small>Title</small><br/><b>{title[:60]}</b></div>
          <div><small>Duration</small><br/><b>{fmt_ts(result["duration"])}</b></div>
          <div><small>Source</small><br/><b>{result["src_label"]}</b></div>
          <div><small>Model</small><br/><b>{result["claude_label"].split(' (')[0]}</b></div>
        </div>
    """
    st.markdown(kpi, unsafe_allow_html=True)

    tab_sum, tab_tx, tab_kf, tab_ch, tab_dbg = st.tabs(
        ["📝 Summary", "🗒️ Transcript", "🖼️ Keyframes", "📑 Chapters", "⚙️ Debug"]
    )

    with tab_sum:
        st.markdown(result["result_md"])
        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "Download summary (.md)",
                data=result["result_md"].encode("utf-8"),
                file_name=f"{out_name}.md",
                mime="text/markdown",
                use_container_width=True,
                key="dl_summary_md",
            )
        with c2:
            st.download_button(
                "Download summary (.txt)",
                data=markdown_to_plain(result["result_md"]).encode("utf-8"),
                file_name=f"{out_name}.txt",
                mime="text/plain",
                use_container_width=True,
                key="dl_summary_txt",
            )

    with tab_tx:
        transcript = result["transcript"]
        st.caption(f"Source: **{result['transcript_source']}** · {len(transcript.splitlines())} segments")
        st.text_area("Transcript", value=transcript, height=400, label_visibility="collapsed")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.download_button("Download .txt", transcript_plain(transcript).encode(),
                               file_name=f"{out_name}.txt", mime="text/plain",
                               use_container_width=True, key="dl_tx_plain")
        with c2:
            st.download_button("Download timestamped .txt", transcript.encode(),
                               file_name=f"{out_name}_ts.txt", mime="text/plain",
                               use_container_width=True, key="dl_tx_ts")
        with c3:
            st.download_button("Download .srt", transcript_to_srt(transcript).encode(),
                               file_name=f"{out_name}.srt", mime="application/x-subrip",
                               use_container_width=True, key="dl_tx_srt")

    with tab_kf:
        keyframes = result["keyframes"]
        if not keyframes:
            st.info("No keyframes extracted.")
        else:
            cols = st.columns(4)
            for i, (ts, p) in enumerate(keyframes):
                with cols[i % 4]:
                    caption = fmt_ts(ts)
                    if result["yt_id"]:
                        caption = f"[{caption}](https://youtu.be/{result['yt_id']}?t={int(ts)})"
                    if Path(p).exists():
                        st.image(p, use_container_width=True)
                    st.markdown(caption)

    with tab_ch:
        chapters = result["chapters"]
        if chapters:
            for ch in chapters:
                s = fmt_ts(ch.get("start_time", 0))
                e = fmt_ts(ch.get("end_time", 0))
                st.markdown(f"- **[{s} – {e}]** {ch.get('title', '')}")
        else:
            st.info("No chapters metadata available for this video.")

    with tab_dbg:
        st.markdown("**Stage timings (seconds)**")
        st.json({k: round(v, 2) for k, v in result["timings"].items()})
        st.markdown("**Settings used**")
        st.json(result["settings"])


# ----- Dispatch ----------------------------------------------------------------
# The pipeline only runs when Analyze is actually clicked. Any other rerun
# (download-button clicks, sidebar tweaks, etc.) just re-renders the cached
# result below — it never re-downloads, re-transcribes, or re-calls Claude.

if go_url and url:
    _cleanup_workdir()
    workdir = Path(tempfile.mkdtemp(prefix="vw_"))
    st.session_state.vw_workdir = str(workdir)
    try:
        with st.status("Downloading with yt-dlp…", expanded=False):
            dl = download_video(url, workdir, prefer_low_res=low_res)
        info = dl.info or {}
        st.session_state.vw_result = compute_pipeline(
            video_path=dl.video_path,
            workdir=workdir,
            sub_path=dl.sub_path,
            title=info.get("title") or url,
            author=info.get("uploader") or info.get("channel") or "",
            source_url=url,
            chapters=info.get("chapters") or [],
        )
    except Exception as e:
        st.exception(e)

if go_file and uploaded is not None:
    _cleanup_workdir()
    workdir = Path(tempfile.mkdtemp(prefix="vw_"))
    st.session_state.vw_workdir = str(workdir)
    suffix = Path(uploaded.name).suffix or ".mp4"
    video_path = workdir / f"upload{suffix}"
    video_path.write_bytes(uploaded.getbuffer())
    try:
        st.session_state.vw_result = compute_pipeline(
            video_path=video_path,
            workdir=workdir,
            sub_path=None,
            title=uploaded.name,
            author="",
            source_url=None,
            chapters=None,
        )
    except Exception as e:
        st.exception(e)

if st.session_state.vw_result:
    render_results(st.session_state.vw_result)
