"""Unit tests for melody_extractor.url_fetch (D-017).

Pure and network-free by default (like test_gui_helpers.py: no streamlit).
All network touchpoints are monkeypatched; the one real-download test is
gated behind MELODY_EXTRACTOR_NETWORK_TESTS=1 + the `network` marker so a
default `pytest` run never leaves the machine.
"""
from __future__ import annotations

import io
import json
import os
import urllib.error
import urllib.request

import pytest

from melody_extractor import url_fetch
from melody_extractor.url_fetch import UrlFetchError, classify_url, spotify_title


# ---------------------------------------------------------------------------
# classify_url
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "youtube"),
        ("https://youtu.be/dQw4w9WgXcQ", "youtube"),
        ("https://music.youtube.com/watch?v=abc123", "youtube"),
        ("https://soundcloud.com/artist/track-name", "soundcloud"),
        ("https://open.spotify.com/track/4uLU6hMCjMI75M1A2tKUQC", "spotify"),
        ("https://open.spotify.com/intl-de/track/4uLU6hMCjMI75M1A2tKUQC", "spotify"),
        ("https://example.com/some/audio", "unknown"),
        ("not a url at all", "unknown"),
        ("", "unknown"),
    ],
)
def test_classify_url(url, expected):
    assert classify_url(url) == expected


def test_classify_url_rejects_lookalike_hosts():
    # Suffix match must be anchored: evil-youtube.com.attacker.example is not youtube.
    assert classify_url("https://notyoutube.com/watch?v=x") == "unknown"
    assert classify_url("https://youtube.com.evil.example/watch?v=x") == "unknown"


# ---------------------------------------------------------------------------
# spotify_title (oEmbed, mocked)
# ---------------------------------------------------------------------------

class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def test_spotify_title_parses_oembed_json(monkeypatch):
    seen = {}

    def fake_urlopen(url, timeout=None):
        seen["url"] = url
        return _FakeResponse(json.dumps({"title": "Partita No. 2 in D Minor"}).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    title = spotify_title("https://open.spotify.com/track/abc?si=xyz")
    assert title == "Partita No. 2 in D Minor"
    # The track URL must be percent-quoted into the oEmbed query string.
    assert seen["url"].startswith("https://open.spotify.com/oembed?url=")
    assert "https%3A%2F%2Fopen.spotify.com%2Ftrack%2Fabc" in seen["url"]


def test_spotify_title_http_error_raises_urlfetcherror(monkeypatch):
    def fake_urlopen(url, timeout=None):
        raise urllib.error.URLError("boom")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(UrlFetchError, match="Spotify"):
        spotify_title("https://open.spotify.com/track/abc")


def test_spotify_title_missing_title_raises(monkeypatch):
    monkeypatch.setattr(
        urllib.request, "urlopen",
        lambda url, timeout=None: _FakeResponse(json.dumps({"no": "title"}).encode()),
    )
    with pytest.raises(UrlFetchError, match="no title"):
        spotify_title("https://open.spotify.com/track/abc")


# ---------------------------------------------------------------------------
# fetch_audio guards (probe seam monkeypatched; no yt-dlp needed)
# ---------------------------------------------------------------------------

def test_fetch_audio_without_ytdlp_names_the_extra(monkeypatch):
    """When yt-dlp is absent the error must name the [url] extra (guarded-
    import pattern shared with transcriber/soundsim)."""
    import builtins

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "yt_dlp":
            raise ImportError("No module named 'yt_dlp'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(ImportError, match=r"melody-extractor\[url\]"):
        url_fetch.fetch_audio("https://www.youtube.com/watch?v=x")


def _patch_ytdlp_present(monkeypatch):
    """fetch_audio imports yt_dlp before probing; give it a do-nothing stub so
    the duration-cap logic (which runs before any download) is reachable."""
    import sys
    import types

    stub = types.ModuleType("yt_dlp")
    stub.utils = types.SimpleNamespace(DownloadError=RuntimeError)
    monkeypatch.setitem(sys.modules, "yt_dlp", stub)


def test_fetch_audio_refuses_over_cap_duration_before_download(monkeypatch):
    _patch_ytdlp_present(monkeypatch)
    monkeypatch.setattr(
        url_fetch, "_probe",
        lambda target: {"duration": 3600, "title": "Long", "webpage_url": "u"},
    )
    with pytest.raises(UrlFetchError, match="cap is 15 min"):
        url_fetch.fetch_audio("https://www.youtube.com/watch?v=x")


def test_fetch_audio_refuses_live_streams(monkeypatch):
    _patch_ytdlp_present(monkeypatch)
    monkeypatch.setattr(
        url_fetch, "_probe",
        lambda target: {"is_live": True, "duration": None, "title": "Live"},
    )
    with pytest.raises(UrlFetchError, match="[Ll]ive"):
        url_fetch.fetch_audio("https://www.youtube.com/watch?v=x")


def test_fetch_audio_refuses_unknown_duration(monkeypatch):
    _patch_ytdlp_present(monkeypatch)
    monkeypatch.setattr(url_fetch, "_probe", lambda target: {"title": "NoDuration"})
    with pytest.raises(UrlFetchError, match="duration"):
        url_fetch.fetch_audio("https://www.youtube.com/watch?v=x")


def test_fetch_audio_spotify_goes_through_title_match(monkeypatch):
    """A Spotify URL must be turned into a ytsearch1: target built from the
    oEmbed title -- verified via the target the probe receives."""
    _patch_ytdlp_present(monkeypatch)
    monkeypatch.setattr(url_fetch, "spotify_title", lambda url, timeout=10.0: "Some Song Title")
    seen = {}

    def fake_probe(target):
        seen["target"] = target
        return {"duration": 3600, "title": "Some Song Title"}  # over cap: stop after probe

    monkeypatch.setattr(url_fetch, "_probe", fake_probe)
    with pytest.raises(UrlFetchError):
        url_fetch.fetch_audio("https://open.spotify.com/track/abc")
    assert seen["target"] == "ytsearch1:Some Song Title"


# ---------------------------------------------------------------------------
# real download (opt-in only)
# ---------------------------------------------------------------------------

@pytest.mark.network
@pytest.mark.skipif(
    not os.environ.get("MELODY_EXTRACTOR_NETWORK_TESTS"),
    reason="set MELODY_EXTRACTOR_NETWORK_TESTS=1 to run real yt-dlp downloads",
)
def test_fetch_audio_real_download_smoke():
    pytest.importorskip("yt_dlp")
    # "Me at the zoo" -- the first YouTube video, 19 s, stable public URL.
    fetched = url_fetch.fetch_audio("https://www.youtube.com/watch?v=jNQXAC9IVRw")
    assert len(fetched.data) > 10_000
    assert fetched.title
    from pathlib import Path

    from melody_extractor.input_adapter import AUDIO_EXTENSIONS

    assert Path(fetched.filename).suffix.lower() in AUDIO_EXTENSIONS
