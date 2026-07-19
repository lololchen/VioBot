"""URL audio acquisition: YouTube/SoundCloud/etc. via yt-dlp; Spotify via
title matching (D-017).

Streamlit-free on purpose (like eval_harness/config_io) so the CLI can reuse
it later; the GUI calls it through `gui.pipeline_cache.fetch_url_bytes`.
Module level imports stdlib only -- yt-dlp is imported lazily inside
`fetch_audio` (guarded-import pattern, cf. transcriber/soundsim) so
`classify_url`/`spotify_title` work without the `[url]` extra installed.

Spotify links are never downloaded (streams are DRM-protected): the public
oEmbed endpoint (no auth) resolves the track title, and yt-dlp's `ytsearch1:`
fetches the best YouTube match for that title. The match may be a cover or
live version -- `FetchedAudio.resolved_url` surfaces what was actually
fetched so the user can verify.

Determinism boundary: this is input *acquisition*. Downloaded bytes enter the
pipeline through the same sha256-digest flow as uploads; the pipeline stays
byte-deterministic per digest, but re-fetching a URL later may yield
different bytes (platforms re-encode).

Downloads land only in a TemporaryDirectory (%TEMP%, outside the
OneDrive-synced repo) and are returned as bytes -- no persistent artifact is
written (D-016 GUI-write restrictions).
"""
from __future__ import annotations

import json
import re
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

DEFAULT_MAX_DURATION_S = 900.0  # 15 min; bounds download size/session memory

_SPOTIFY_OEMBED = "https://open.spotify.com/oembed?url="

# Hostname -> source kind. `/intl-xx/` Spotify paths share the same hostname,
# so hostname matching alone covers them.
_HOST_PATTERNS: "list[tuple[str, re.Pattern[str]]]" = [
    ("youtube", re.compile(r"(^|\.)(youtube\.com|youtu\.be)$")),
    ("soundcloud", re.compile(r"(^|\.)soundcloud\.com$")),
    ("spotify", re.compile(r"(^|\.)spotify\.com$")),
]


class UrlFetchError(RuntimeError):
    """User-displayable fetch failure; the GUI shows str(e) verbatim."""


@dataclass(frozen=True)
class FetchedAudio:
    """Downloaded audio bytes plus provenance.

    `filename` carries the real extension of the downloaded stream (m4a when
    available, else webm/opus) -- gui.pipeline_cache.load dispatches decode
    by that suffix. `resolved_url` differs from `source_url` for Spotify
    input (it is the matched YouTube video)."""

    data: bytes
    filename: str
    title: str
    source_url: str
    resolved_url: str


def classify_url(url: str) -> "Literal['youtube', 'soundcloud', 'spotify', 'unknown']":
    """Classify by hostname. 'unknown' is still handed to yt-dlp (its generic
    extractor supports hundreds of sites); the label only steers UX text and
    the Spotify title-matching branch."""
    try:
        host = (urllib.parse.urlsplit(url.strip()).hostname or "").lower()
    except ValueError:
        return "unknown"
    for kind, pattern in _HOST_PATTERNS:
        if pattern.search(host):
            return kind  # type: ignore[return-value]
    return "unknown"


def spotify_title(url: str, timeout: float = 10.0) -> str:
    """Track title via Spotify's public oEmbed endpoint (no auth, no DRM
    circumvention -- metadata only)."""
    oembed_url = _SPOTIFY_OEMBED + urllib.parse.quote(url.strip(), safe="")
    try:
        with urllib.request.urlopen(oembed_url, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, ValueError, OSError) as e:
        raise UrlFetchError(
            f"Could not resolve the Spotify track (is the link public?): {e}"
        ) from e
    title = payload.get("title")
    if not title:
        raise UrlFetchError("Spotify oEmbed returned no title for this link.")
    return str(title)


def _import_yt_dlp():
    try:
        import yt_dlp  # heavy + fast-moving; [url] extra only
    except ImportError as e:
        raise ImportError(
            "yt-dlp is required for URL audio input. "
            'Install it with: pip install "melody-extractor[url]"'
        ) from e
    return yt_dlp


def _probe(target: str) -> dict:
    """Metadata-only extract_info (no download). For `ytsearch1:` targets the
    single search result is unwrapped. Separate function on purpose: the
    duration-cap tests monkeypatch this seam."""
    yt_dlp = _import_yt_dlp()
    opts = {"quiet": True, "no_warnings": True, "noplaylist": True}
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(target, download=False)
    except yt_dlp.utils.DownloadError as e:
        raise UrlFetchError(_trim_ytdlp_error(e)) from e
    if info is None:
        raise UrlFetchError("No media found at this URL.")
    if info.get("entries") is not None:  # search / playlist wrapper
        entries = [e for e in info["entries"] if e]
        if not entries:
            raise UrlFetchError("No matching video found on YouTube.")
        info = entries[0]
    return info


def _trim_ytdlp_error(e: Exception) -> str:
    """yt-dlp errors are long and ANSI-colored; keep the first line and add
    an update hint (extractors break when platforms change)."""
    first_line = str(e).splitlines()[0] if str(e) else repr(e)
    first_line = re.sub(r"\x1b\[[0-9;]*m", "", first_line)
    return (
        f"Download failed: {first_line} "
        "(private/geo-blocked/removed video? If YouTube changed something, "
        "try: pip install -U yt-dlp)"
    )


def fetch_audio(url: str, max_duration_s: float = DEFAULT_MAX_DURATION_S) -> FetchedAudio:
    """Download the audio stream behind `url` and return it as bytes.

    Spotify URLs are resolved to a title (oEmbed) and matched on YouTube via
    `ytsearch1:`. Duration is probed *before* downloading and refused above
    `max_duration_s` (and for live streams). Prefers a native m4a stream and
    never re-encodes -- webm/opus fall out as-is and decode via the existing
    ffmpeg branch in input_adapter (AUDIO_EXTENSIONS includes them, D-017)."""
    yt_dlp = _import_yt_dlp()
    url = url.strip()

    if classify_url(url) == "spotify":
        target = f"ytsearch1:{spotify_title(url)}"
    else:
        target = url

    info = _probe(target)
    if info.get("is_live"):
        raise UrlFetchError("Live streams are not supported.")
    duration = info.get("duration")
    if duration is None:
        raise UrlFetchError("Could not determine the track's duration; refusing to download.")
    if duration > max_duration_s:
        raise UrlFetchError(
            f"Track is {duration / 60:.1f} min long; the cap is "
            f"{max_duration_s / 60:.0f} min (keeps downloads and memory bounded)."
        )

    resolved_url = info.get("webpage_url") or url
    title = str(info.get("title") or "downloaded audio")

    with tempfile.TemporaryDirectory() as tmp_dir:
        opts = {
            "format": "bestaudio[ext=m4a]/bestaudio/best",
            "outtmpl": str(Path(tmp_dir) / "%(id)s.%(ext)s"),
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "noprogress": True,  # keep the GUI server console clean
        }
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.extract_info(resolved_url, download=True)
        except yt_dlp.utils.DownloadError as e:
            raise UrlFetchError(_trim_ytdlp_error(e)) from e

        files = [p for p in Path(tmp_dir).iterdir() if p.is_file()]
        if not files:
            raise UrlFetchError("yt-dlp reported success but produced no file.")
        # Single video + noplaylist => exactly one output; largest wins if a
        # stray .part/.json ever appears.
        out = max(files, key=lambda p: p.stat().st_size)
        data = out.read_bytes()

    return FetchedAudio(
        data=data,
        filename=out.name,
        title=title,
        source_url=url,
        resolved_url=resolved_url,
    )
