"""The Streamlit caching chain (plan_GUI_MelodyExtractor.md "Caching chain").

`register_bytes(data, filename) -> digest` stores raw uploaded/fixture bytes
(per-session, in `st.session_state`) keyed by their sha256 hex digest and
returns that digest; every stage function below takes the digest (never the
raw bytes/array) so `st.cache_data`'s hashing only ever sees a short string
plus the frozen config dataclasses (already hashable -- no `hash_funcs`
needed).

Chain: `load(digest)` [`st.cache_resource`, content-addressed] ->
`transcribe(digest, mono_cfg)` -> `timbre(digest, mono_cfg, timbre_cfg)` ->
`reduce(digest, mono_cfg, timbre_cfg, stage_cfg)` ->
`render(digest, mono_cfg, timbre_cfg, render_cfg, stage_cfg=None)`
[`st.cache_data`, all `@st.cache_data`].

Effect this preserves (verified manually per the plan's verification
checklist): tweaking a reducer weight only invalidates `reduce`+`render`
(cheap -- `transcribe`/`timbre` cache hit); tweaking `voicing_threshold`
invalidates `transcribe` onward; a new file (new digest) invalidates
everything. Cold `transcribe`/`timbre`/`reduce`/`render` calls show a
`st.spinner` via `cache_data`'s built-in `show_spinner=<message>` (only
fires on an actual cache miss, i.e. on the very computation this needs to
wrap). Every cache is bounded with `max_entries` (D-018): entries here are
full-song objects (AudioBuffers, NoteSequences with 30k-frame tracks,
rendered WAV bytes), so an unbounded cache grows by tens of MB per uploaded
song / parameter combination until the server process dies -- which the
browser reports as "Connection error".

Also provides the eval cache: `run_eval_cached(...)` keyed on the fixtures
directory (as its string path), a digest of the fixture files' mtimes (so
regenerating fixtures invalidates it), and the three sidebar configs; the
`_progress` callback parameter is intentionally underscore-prefixed so
Streamlit's cache hashing skips it (its identity is a closure over an
`st.status` box built for a single click, not part of the cache identity).

MIDI input note: `load()` dispatches like `input_adapter.load()` -- audio ->
AudioBuffer, MIDI -> NoteSequence directly (module CLAUDE.md: "MIDI input
path must not round-trip through audio"). `transcribe`/`timbre` pass a
NoteSequence straight through when the loaded object already is one (no
transcriber/timbre model applies to symbolic input); `render`'s RMS-matching
against the "original" is skipped for the same reason (there is no source
recording).
"""
from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
import streamlit as st

from .. import eval_harness, reducer, soundsim, transcriber
from .. import timbre as timbre_module
from .. import input_adapter
from ..input_adapter import AudioBuffer
from ..reducer import StageConfig
from ..schema import NoteSequence
from ..soundsim import RenderConfig
from ..timbre import TimbreConfig
from ..transcriber import MonoConfig
from . import audio_bytes

_REGISTRY_KEY = "_pipeline_cache_bytes_registry"


def register_bytes(data: bytes, filename: str) -> str:
    """Store `data` (per-session) keyed by its sha256 hex digest; return the
    digest. Downstream stage functions take this digest, never the bytes."""
    digest = hashlib.sha256(data).hexdigest()
    registry = st.session_state.setdefault(_REGISTRY_KEY, {})
    registry.setdefault(digest, (data, filename))
    return digest


@st.cache_resource(show_spinner="Decoding input (input_adapter)...", max_entries=8)
def load(digest: str) -> "Union[AudioBuffer, NoteSequence]":
    """AudioBuffer (audio) or NoteSequence (MIDI), content-addressed by digest."""
    registry = st.session_state.get(_REGISTRY_KEY, {})
    if digest not in registry:
        raise KeyError(f"pipeline_cache.load: no bytes registered for digest {digest[:12]}...")
    data, filename = registry[digest]

    import tempfile
    suffix = Path(filename).suffix.lower()
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / f"input{suffix}"
        path.write_bytes(data)
        if suffix in input_adapter.MIDI_EXTENSIONS:
            return input_adapter.load_midi(path)
        return input_adapter.load_audio(path)


@st.cache_data(show_spinner="Transcribing (transcriber.transcribe_mono)...", max_entries=16)
def transcribe(digest: str, mono_cfg: MonoConfig) -> NoteSequence:
    obj = load(digest)
    if isinstance(obj, NoteSequence):
        return obj  # MIDI input: parsed directly, no transcription step applies
    return transcriber.transcribe_mono(obj, mono_cfg)


@st.cache_data(show_spinner="Analyzing timbre (timbre.add_harmonics)...", max_entries=16)
def timbre(digest: str, mono_cfg: MonoConfig, timbre_cfg: TimbreConfig) -> NoteSequence:
    seq = transcribe(digest, mono_cfg)
    obj = load(digest)
    if not isinstance(obj, AudioBuffer):
        return seq  # MIDI input: no source audio to analyze harmonics from
    return timbre_module.add_harmonics(obj, seq, timbre_cfg)


@st.cache_data(show_spinner="Reducing voices (reducer.reduce)...", max_entries=16)
def reduce(digest: str, mono_cfg: MonoConfig, timbre_cfg: TimbreConfig, stage_cfg: StageConfig) -> NoteSequence:
    seq = timbre(digest, mono_cfg, timbre_cfg)
    return reducer.reduce(seq, stage_cfg)


@st.cache_data(show_spinner="Rendering audio (soundsim)...", max_entries=8)
def render(digest: str, mono_cfg: MonoConfig, timbre_cfg: TimbreConfig, render_cfg: RenderConfig,
           stage_cfg: "Optional[StageConfig]" = None) -> bytes:
    """WAV bytes (via audio_bytes.wav_bytes, D-013) of either the pre-reduction
    NoteSequence (stage_cfg=None) or the reduced one (stage_cfg given),
    RMS-matched to the original audio when one exists (skipped for MIDI
    input -- there is no source recording to match against)."""
    if stage_cfg is not None:
        seq = reduce(digest, mono_cfg, timbre_cfg, stage_cfg)
    else:
        seq = timbre(digest, mono_cfg, timbre_cfg)

    arr = soundsim.render_to_array(seq, render_cfg).astype(np.float64)

    obj = load(digest)
    if isinstance(obj, AudioBuffer):
        arr = audio_bytes.rms_matched(arr, np.asarray(obj.pcm, dtype=np.float64))

    return audio_bytes.wav_bytes(arr, render_cfg.sample_rate)


# ---------------------------------------------------------------------------
# URL fetch cache (D-017)
# ---------------------------------------------------------------------------

@st.cache_data(max_entries=8, show_spinner=False)
def fetch_url_bytes(url: str, max_duration_s: float) -> "tuple[bytes, str, str, str]":
    """Cached `url_fetch.fetch_audio`: (data, filename, title, resolved_url).

    `show_spinner=False` because app.py wraps the call in its own
    `st.status` box. `st.cache_data` does not cache exceptions, so a failed
    fetch retries on the next click while a success survives reruns and
    repeated clicks without re-downloading. The bytes then enter the normal
    digest-keyed flow via `register_bytes` -- the pipeline's determinism
    boundary is unchanged (re-fetching a URL later may yield different bytes;
    that is acquisition, not pipeline)."""
    from .. import url_fetch

    fetched = url_fetch.fetch_audio(url, max_duration_s)
    return (fetched.data, fetched.filename, fetched.title, fetched.resolved_url)


# ---------------------------------------------------------------------------
# eval cache
# ---------------------------------------------------------------------------

def fixtures_mtime_digest(fixtures_dir: "str | Path") -> str:
    """sha256 over every fixture file's (name, mtime_ns) -- invalidates the
    eval cache whenever the fixture corpus is regenerated/edited."""
    fixtures_dir = Path(fixtures_dir)
    h = hashlib.sha256()
    if fixtures_dir.is_dir():
        for p in sorted(fixtures_dir.iterdir()):
            if p.is_file():
                h.update(p.name.encode("utf-8"))
                h.update(str(p.stat().st_mtime_ns).encode("utf-8"))
    return h.hexdigest()


@st.cache_data(show_spinner="Running eval harness (eval_harness.run_eval)...")
def run_eval_cached(
    fixtures_dir_str: str,
    mtime_digest: str,
    mono_cfg: MonoConfig,
    timbre_cfg: TimbreConfig,
    stage_cfg: StageConfig,
    _progress: "Optional[Callable[[str], None]]" = None,
) -> dict:
    """Cached `eval_harness.run_eval`, keyed on the fixtures dir + a digest of
    its files' mtimes + the three sidebar configs. `mtime_digest` is a
    parameter (not computed inside) so it participates in the cache key --
    same configs but a regenerated fixture corpus must not reuse a stale
    result. `_progress` is excluded from hashing by Streamlit's leading-
    underscore convention (it's a closure over a per-click `st.status` box,
    not part of the cached computation's identity); on a cache hit it simply
    never fires, which is correct -- there is nothing new to report."""
    stage_configs = {n: replace(stage_cfg, max_voices=n) for n in (1, 2, 3)}
    return eval_harness.run_eval(
        fixtures_dir_str,
        mono_config=mono_cfg,
        timbre_config=timbre_cfg,
        stage_configs=stage_configs,
        progress=_progress,
    )
