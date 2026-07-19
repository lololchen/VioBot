"""In-memory WAV encoding + loudness matching for the GUI's audio players.

PURE module: numpy + scipy only, no streamlit import (module CLAUDE.md rule
"nothing outside gui/ imports streamlit/plotly" has a mirror rule inside
gui/: figures.py and audio_bytes.py stay presentation-logic-only and pure so
they're independently testable and reusable from a future non-Streamlit UI).

WAV writing follows D-013 (docs/decisions.md): scipy.io.wavfile.write only.
soundfile/libsndfile stamps float-subtype WAVs with a timestamped PEAK chunk
that breaks byte-determinism; scipy's writer has no such chunk and is
byte-deterministic across calls. Here we additionally write int16 PCM (not
float) so there is no float/PEAK-chunk question at all, and so `st.audio`
gets a widely-compatible WAV subtype.
"""
from __future__ import annotations

import io

import numpy as np
from scipy.io import wavfile

# int16 full-scale magnitude. Scaling by 32767 (not 32768) keeps +1.0 mapping
# to the largest representable positive int16 instead of overflowing.
_INT16_FULL_SCALE = 32767.0
# RMS below this is treated as "silence" by rms_matched's guard.
_SILENCE_RMS = 1e-9


def wav_bytes(pcm: np.ndarray, sr: int) -> bytes:
    """Encode float PCM (nominally in [-1, 1]) as an in-memory 16-bit PCM WAV.

    Uses scipy.io.wavfile.write into an io.BytesIO buffer (D-013: never
    soundfile/wave). Samples are clipped to [-1, 1] before scaling to int16
    so out-of-range input can't wrap around instead of clipping.
    """
    arr = np.asarray(pcm, dtype=np.float64)
    clipped = np.clip(arr, -1.0, 1.0)
    int16 = np.round(clipped * _INT16_FULL_SCALE).astype(np.int16)

    buf = io.BytesIO()
    wavfile.write(buf, int(sr), int16)
    return buf.getvalue()


def rms_matched(render: np.ndarray, original: np.ndarray) -> np.ndarray:
    """Scale `render` so its RMS equals `original`'s RMS (the A/B-listening
    loudness-matching contract shared with soundsim.render_paired).

    Silence guard: if either signal's RMS is ~0 (a silent original gives no
    meaningful target level; a silent render has nothing to scale), the
    render is returned UNSCALED rather than raising or producing inf/NaN --
    scaling a near-zero render up to match a loud original would otherwise
    blow a near-silent signal into audible noise, which is worse than
    leaving the (already near-silent) render alone.
    """
    render_arr = np.asarray(render, dtype=np.float64)
    original_arr = np.asarray(original, dtype=np.float64)

    render_rms = _rms(render_arr)
    original_rms = _rms(original_arr)
    if render_rms <= _SILENCE_RMS or original_rms <= _SILENCE_RMS:
        return render_arr

    return render_arr * (original_rms / render_rms)


def _rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x))))
