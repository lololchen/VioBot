"""Input adapters: audio files -> 16 kHz mono float32 PCM; MIDI -> NoteSequence directly.

Contracts (PRD F1, module CLAUDE.md):
- Audio: WAV/FLAC read via soundfile; MP3/M4A/MP4/OGG/WEBM/OPUS decoded by ffmpeg
  (PATH ffmpeg first, else the bundled imageio-ffmpeg binary, else a clear
  error). Output is always AudioBuffer(pcm float32 mono in [-1, 1], sample_rate).
- Resampling must be deterministic: scipy.signal.resample_poly, never
  stochastic/multithreaded-nondeterministic paths.
- MIDI input must NOT round-trip through audio: parse note events straight
  into a NoteSequence (pretty_midi). Velocity maps to a flat AmpEnvelope via
  the inverse of schema._db_to_velocity; confidence = 1.0; tempo/pitch-bend
  events become f0_contour where present.
- Stereo -> mono: mean of channels before resampling.
"""
from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .schema import AmpEnvelope, F0Contour, Meta, Note, NoteSequence, midi_to_hz

TARGET_SR = 16000

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a", ".mp4", ".ogg", ".aiff", ".aif", ".webm", ".opus"}
MIDI_EXTENSIONS = {".mid", ".midi"}

# Extensions libsndfile can read directly; everything else in AUDIO_EXTENSIONS
# goes through ffmpeg first.
_DIRECT_SOUNDFILE_EXTENSIONS = {".wav", ".flac", ".aiff", ".aif"}

# Bend range assumed for GM-style pitch bends: +-2 semitones over the 14-bit
# signed range [-8192, 8191]. Mirrors schema.py's to_midi() encoding.
_BEND_RANGE_SEMITONES = 2.0
_BEND_FULL_SCALE = 8192.0


@dataclass(frozen=True)
class AudioBuffer:
    """Mono float32 PCM in [-1, 1]."""

    pcm: np.ndarray
    sample_rate: int
    source: str = ""

    @property
    def duration_s(self) -> float:
        return len(self.pcm) / self.sample_rate


def find_ffmpeg() -> "str | None":
    """PATH ffmpeg first, else imageio-ffmpeg's bundled binary, else None."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _decode_via_ffmpeg(path: Path) -> "tuple[np.ndarray, int]":
    """Decode an arbitrary audio file to float64 PCM via ffmpeg, through a
    temp WAV file (avoids relying on a seekable pipe for the WAV header)."""
    ffmpeg_exe = find_ffmpeg()
    if ffmpeg_exe is None:
        raise RuntimeError(
            "ffmpeg not found: no 'ffmpeg' on PATH and imageio-ffmpeg is not "
            "installed. Install ffmpeg (e.g. from https://ffmpeg.org/download.html "
            "and add it to PATH) or `pip install imageio-ffmpeg`."
        )
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_path = Path(tmp_dir) / "decoded.wav"
        cmd = [
            ffmpeg_exe, "-v", "error", "-y",
            "-i", str(path),
            "-f", "wav", "-acodec", "pcm_f32le",
            str(out_path),
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0 or not out_path.exists():
            stderr = proc.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"ffmpeg failed to decode {path!s}: {stderr}")
        data, sr = sf.read(str(out_path), dtype="float64", always_2d=True)
    return data, sr


def _resample_deterministic(x: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    """scipy.signal.resample_poly with gcd-reduced up/down factors: deterministic,
    no RNG, no threading nondeterminism."""
    sr = int(sr)
    target_sr = int(target_sr)
    if sr == target_sr:
        return x
    g = math.gcd(sr, target_sr)
    up = target_sr // g
    down = sr // g
    return resample_poly(x, up, down)


def load_audio(path: "str | Path", target_sr: int = TARGET_SR) -> AudioBuffer:
    """Decode any supported audio file to mono float32 at target_sr."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext in _DIRECT_SOUNDFILE_EXTENSIONS:
        data, sr = sf.read(str(path), dtype="float64", always_2d=True)
    elif ext in AUDIO_EXTENSIONS:
        data, sr = _decode_via_ffmpeg(path)
    else:
        raise ValueError(f"unsupported audio extension: {ext!r} ({path!s})")

    mono = data.mean(axis=1)  # float64 math; works for any channel count, incl. 1
    resampled = _resample_deterministic(mono, sr, target_sr)
    pcm = np.clip(resampled, -1.0, 1.0).astype(np.float32)
    return AudioBuffer(pcm=pcm, sample_rate=int(target_sr), source=path.name)


def _velocity_to_db(velocity: int) -> float:
    """Inverse of schema._db_to_velocity: velocity v -> db = (v-1)/126*60 - 60."""
    v = min(127, max(1, int(velocity)))
    return (v - 1) / 126.0 * 60.0 - 60.0


def load_midi(path: "str | Path") -> NoteSequence:
    """Parse a MIDI file directly into a NoteSequence (no DSP, no audio)."""
    import pretty_midi

    path = Path(path)
    pm = pretty_midi.PrettyMIDI(str(path))

    notes: list[Note] = []
    for inst in pm.instruments:
        bends = sorted(inst.pitch_bends, key=lambda b: b.time)
        for note in inst.notes:
            onset = float(note.start)
            offset = float(note.end)
            duration = offset - onset
            if not duration > 0.0:
                continue  # degenerate zero/negative-length note: not a playable event

            pitch_hz = midi_to_hz(float(note.pitch))
            amp_db = _velocity_to_db(note.velocity)
            envelope = AmpEnvelope.constant(amp_db, duration)

            overlapping = [b for b in bends if onset <= b.time <= offset]
            contour = None
            if overlapping:
                times_s = [b.time - onset for b in overlapping]
                f0_hz = [
                    pitch_hz * (2.0 ** ((b.pitch / _BEND_FULL_SCALE * _BEND_RANGE_SEMITONES) / 12.0))
                    for b in overlapping
                ]
                contour = F0Contour(times_s=times_s, f0_hz=f0_hz)

            notes.append(
                Note(
                    pitch_hz=pitch_hz,
                    onset_s=onset,
                    duration_s=duration,
                    amp_db_envelope=envelope,
                    velocity=min(127, max(0, int(note.velocity))),
                    confidence=1.0,
                    f0_contour=contour,
                )
            )

    meta = Meta(source=path.name, source_kind="midi", backends={"input_adapter": "midi-0.1.0"})
    return NoteSequence(notes=tuple(notes), meta=meta).sorted().validate()


def load(path: "str | Path") -> "AudioBuffer | NoteSequence":
    """Dispatch on extension: audio -> AudioBuffer, MIDI -> NoteSequence."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext in AUDIO_EXTENSIONS:
        return load_audio(path)
    if ext in MIDI_EXTENSIONS:
        return load_midi(path)
    raise ValueError(f"unsupported file extension: {ext!r} ({path!s})")
