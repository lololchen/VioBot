"""SoundSim: render a NoteSequence back to audio for A/B listening (PRD F5, D-006).

Backends:
- "additive" (default, always available, deterministic): per note, sum
  sinusoidal partials k = 1..K. Partial amplitudes from note.harmonics
  (linear amps recovered from harmonic_amps_db, normalized so the sum
  matches the envelope level); fallback when harmonics is None: sawtooth-like
  rolloff a_k = 1/k², K = 8. Frequency follows f0_contour when present
  (linear interp between contour points, phase-continuous integration),
  else constant pitch_hz. Amplitude follows amp_db_envelope (linear interp
  in dB, converted to linear). 5 ms raised-cosine fade-in/out per note to
  avoid clicks. Mix all notes, normalize to peak -1 dBFS only if clipping.
- "fluidsynth" (optional, guarded import pyfluidsynth): render via a violin
  SoundFont; SoundFont path from config, else discovered via the
  MELODY_EXTRACTOR_SF2 env var or %LOCALAPPDATA%/MelodyExtractor/soundfonts/
  (_find_default_soundfont). Raise a clear ImportError with install hints
  when unavailable.

render_paired: the listening-test contract (module CLAUDE.md gotcha): emit
original.wav + extracted_render.wav LOUDNESS-MATCHED (scale the render so its
RMS equals the original's RMS over voiced regions) into one directory.

Determinism: additive backend must be byte-identical across runs (float64
synthesis, fixed phase 0 per partial, no RNG).

WAV writing note: files are written with scipy.io.wavfile.write, NOT
soundfile's subtype="FLOAT" writer. libsndfile stamps float-subtype WAVs with
a PEAK chunk that embeds a Unix timestamp, which makes two renders of the same
input differ byte-for-byte if they happen in different seconds -- silently
breaking the determinism contract above. scipy.io.wavfile has no such chunk
and was verified byte-identical across writes; soundfile can still read the
resulting files fine.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from .input_adapter import AudioBuffer
from .schema import Note, NoteSequence, hz_to_midi

# -1 dBFS, linear amplitude.
_MASTER_PEAK_LIMIT = 10.0 ** (-1.0 / 20.0)
# Silent/near-silent placeholder length for an empty NoteSequence.
_EMPTY_SEQUENCE_S = 0.1


@dataclass(frozen=True)
class RenderConfig:
    backend: str = "additive"     # "additive" | "fluidsynth"
    sample_rate: int = 44100
    n_partials: int = 8
    fade_s: float = 0.005
    soundfont_path: "str | None" = None
    # GM program for the fluidsynth backend (0-based; 40 = violin). Ignored by
    # "additive". Meaningful values depend on the loaded SoundFont's bank 0.
    midi_program: int = 40


def render(seq: NoteSequence, out_wav: "str | Path", config: RenderConfig = RenderConfig()) -> Path:
    """Render seq to a WAV file; returns the written path."""
    arr = render_to_array(seq, config)
    out_path = Path(out_wav)
    if out_path.parent:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(str(out_path), config.sample_rate, arr)
    return out_path


def render_to_array(seq: NoteSequence, config: RenderConfig = RenderConfig()) -> np.ndarray:
    """Render to a float32 numpy array (mono, config.sample_rate). Exposed for tests."""
    if config.backend == "additive":
        master = _render_additive(seq, config)
    elif config.backend == "fluidsynth":
        master = _render_fluidsynth(seq, config)
    else:
        raise ValueError(f"soundsim: unknown backend {config.backend!r} (expected 'additive' or 'fluidsynth')")

    peak = float(np.max(np.abs(master))) if master.size else 0.0
    if peak > _MASTER_PEAK_LIMIT:
        master = master * (_MASTER_PEAK_LIMIT / peak)
    return master.astype(np.float32)


def render_paired(original: AudioBuffer, seq: NoteSequence, out_dir: "str | Path",
                  config: RenderConfig = RenderConfig()) -> "tuple[Path, Path]":
    """Write loudness-matched (original.wav, extracted_render.wav) into out_dir.

    Both files are written at original.sample_rate (fair A/B, module CLAUDE.md
    gotcha) and the render is scaled so its overall RMS matches the original's
    RMS. Silent originals (or an all-silent render) are guarded against
    division by zero -- the render is left unscaled in that case.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    render_config = replace(config, sample_rate=original.sample_rate)
    rendered = render_to_array(seq, render_config).astype(np.float64)

    orig_pcm = np.asarray(original.pcm, dtype=np.float64)
    orig_rms = _rms(orig_pcm)
    render_rms = _rms(rendered)
    if orig_rms > 0.0 and render_rms > 0.0:
        rendered = rendered * (orig_rms / render_rms)

    original_path = out_dir / "original.wav"
    render_path = out_dir / "extracted_render.wav"
    wavfile.write(str(original_path), original.sample_rate, orig_pcm.astype(np.float32))
    wavfile.write(str(render_path), original.sample_rate, rendered.astype(np.float32))
    return original_path, render_path


# ---------------------------------------------------------------------------
# additive backend
# ---------------------------------------------------------------------------

def _rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x))))


def _render_additive(seq: NoteSequence, config: RenderConfig) -> np.ndarray:
    sr = config.sample_rate
    notes = seq.notes
    if not notes:
        n_total = max(int(round(_EMPTY_SEQUENCE_S * sr)), 1)
        return np.zeros(n_total, dtype=np.float64)

    total_s = max(n.offset_s for n in notes)
    n_total = max(int(round(total_s * sr)), 1)
    master = np.zeros(n_total, dtype=np.float64)

    for note in notes:
        wave = _synthesize_note(note, sr, config.n_partials, config.fade_s)
        onset_sample = int(round(note.onset_s * sr))
        end_sample = onset_sample + len(wave)
        if end_sample > len(master):
            master = np.pad(master, (0, end_sample - len(master)))
        master[onset_sample:end_sample] += wave

    return master


def _synthesize_note(note: Note, sr: int, n_partials: int, fade_s: float) -> np.ndarray:
    n = max(int(round(note.duration_s * sr)), 1)
    t = np.arange(n, dtype=np.float64) / sr  # time relative to onset

    if note.f0_contour is not None:
        c = note.f0_contour
        # np.interp clamps to the edge values outside [times_s[0], times_s[-1]].
        f0_arr = np.interp(t, c.times_s, c.f0_hz)
    else:
        f0_arr = np.full(n, note.pitch_hz, dtype=np.float64)

    # Phase-continuous integration: phase[n] = 2*pi * sum_{i<n} f0[i]/sr, an
    # exclusive cumulative sum so phase[0] == 0 (fixed zero initial phase --
    # required for byte-identical determinism). For constant f0 this reduces
    # exactly to 2*pi*f0*t.
    phase_inc = 2.0 * np.pi * f0_arr / sr
    phase1 = np.cumsum(phase_inc) - phase_inc

    f0_max = float(np.max(f0_arr))
    nyquist = sr / 2.0

    if note.harmonics is not None:
        partial_amps = [10.0 ** (db / 20.0) for db in note.harmonics.harmonic_amps_db[:n_partials]]
    else:
        partial_amps = [1.0 / (k ** 2) for k in range(1, n_partials + 1)]

    wave = np.zeros(n, dtype=np.float64)
    for idx, amp in enumerate(partial_amps):
        k = idx + 1
        if k * f0_max >= nyquist:
            continue  # partial would alias -- skip it entirely for this note
        wave += amp * np.sin(k * phase1)

    peak = float(np.max(np.abs(wave))) if wave.size else 0.0
    if peak > 0.0:
        wave = wave / peak

    env = note.amp_db_envelope
    env_db = np.interp(t, env.times_s, env.amp_db)  # clamped ends
    gain = 10.0 ** (env_db / 20.0)
    wave = wave * gain

    fade_n = min(int(round(fade_s * sr)), n // 2)
    if fade_n > 0:
        ramp = 0.5 * (1.0 - np.cos(np.pi * np.arange(fade_n, dtype=np.float64) / fade_n))
        wave[:fade_n] *= ramp
        wave[-fade_n:] *= ramp[::-1]

    return wave


# ---------------------------------------------------------------------------
# fluidsynth backend
# ---------------------------------------------------------------------------

def _find_default_soundfont() -> "str | None":
    """Fallback SoundFont discovery when RenderConfig.soundfont_path is unset:
    the MELODY_EXTRACTOR_SF2 env var first, else the alphabetically first .sf2
    under %LOCALAPPDATA%/MelodyExtractor/soundfonts (~ /.local/share off
    Windows). Deterministic per machine (sorted); which soundfont is installed
    is user environment, exactly like soundfont_path itself."""
    import os

    env = os.environ.get("MELODY_EXTRACTOR_SF2")
    if env and Path(env).exists():
        return env
    base = os.environ.get("LOCALAPPDATA")
    sf_dir = (Path(base) if base else Path.home() / ".local" / "share") / "MelodyExtractor" / "soundfonts"
    if sf_dir.is_dir():
        candidates = sorted(p for p in sf_dir.glob("*.sf2") if p.is_file())
        if candidates:
            return str(candidates[0])
    return None


def _import_fluidsynth():
    """Import pyfluidsynth with the FluidSynth DLL made findable first.

    ctypes only searches the process PATH (plus the executable's own
    directory), and a Streamlit server process typically has neither the
    venv's Scripts dir nor %LOCALAPPDATA%/MelodyExtractor/fluidsynth/.../bin
    on PATH -- so `import fluidsynth` failed inside the GUI even with
    everything installed (D-019). Prepend every known DLL location to PATH
    (and register it via os.add_dll_directory) before importing; a failed
    module import is not cached, so retrying after the PATH fix works.
    """
    import os
    import sys

    candidates = [Path(sys.executable).parent]
    base = os.environ.get("LOCALAPPDATA")
    if base:
        candidates += sorted(Path(base).glob("MelodyExtractor/fluidsynth/*/bin"))
    for d in candidates:
        if d.is_dir() and any(d.glob("*fluidsynth*.dll")):
            path = os.environ.get("PATH", "")
            if str(d) not in path.split(os.pathsep):
                os.environ["PATH"] = str(d) + os.pathsep + path
            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(str(d))
                except OSError:
                    pass

    import fluidsynth  # pyfluidsynth
    return fluidsynth


def _render_fluidsynth(seq: NoteSequence, config: RenderConfig) -> np.ndarray:
    try:
        fluidsynth = _import_fluidsynth()
    except ImportError as exc:
        import importlib.util

        if importlib.util.find_spec("fluidsynth") is None:
            hint = (
                "the 'pyfluidsynth' Python package is not installed in this "
                "environment. Install with `pip install pyfluidsynth` (or the extra: "
                "`pip install melody-extractor[render-fluidsynth]`)."
            )
        else:
            hint = (
                "pyfluidsynth is installed but the FluidSynth shared library "
                "(libfluidsynth-3.dll) could not be loaded. Put its bin folder on "
                "PATH, or drop the DLLs into the venv's Scripts directory, or "
                "extract a FluidSynth release under "
                "%LOCALAPPDATA%/MelodyExtractor/fluidsynth/ (auto-searched)."
            )
        raise ImportError(
            "soundsim backend 'fluidsynth': " + hint + " You ALSO need a violin "
            "SoundFont (.sf2) -- not bundled; see RenderConfig.soundfont_path / "
            "MELODY_EXTRACTOR_SF2 / %LOCALAPPDATA%/MelodyExtractor/soundfonts/."
        ) from exc

    soundfont = config.soundfont_path or _find_default_soundfont()
    if not soundfont:
        raise ValueError(
            "soundsim backend 'fluidsynth' requires a .sf2 violin SoundFont: set "
            "RenderConfig.soundfont_path, or set the MELODY_EXTRACTOR_SF2 env var, "
            "or drop a .sf2 file into %LOCALAPPDATA%/MelodyExtractor/soundfonts/."
        )
    soundfont_path = Path(soundfont)
    if not soundfont_path.exists():
        raise FileNotFoundError(f"soundsim: soundfont not found: {soundfont_path}")

    sr = config.sample_rate
    total_s = max((n.offset_s for n in seq.notes), default=0.0)
    n_total = max(int(round(total_s * sr)), 1)

    synth = fluidsynth.Synth(samplerate=float(sr))
    try:
        sfid = synth.sfload(str(soundfont_path))
        # channel 0, bank 0, config.midi_program (GM: 40 = violin default)
        synth.program_select(0, sfid, 0, int(config.midi_program))

        events = []
        for note in seq.notes:
            midi = min(127, max(0, int(round(hz_to_midi(note.pitch_hz)))))
            vel = note.velocity if note.velocity is not None else _velocity_from_envelope(note.amp_db_envelope)
            events.append((note.onset_s, 1, midi, vel))   # 1 = note on (sorts after note-off at same time)
            events.append((note.offset_s, 0, midi, vel))  # 0 = note off
        events.sort(key=lambda e: (e[0], e[1]))

        chunks = []
        cur_sample = 0
        for t_s, kind, midi, vel in events:
            target_sample = int(round(t_s * sr))
            if target_sample > cur_sample:
                chunks.append(_fluidsynth_block(synth, target_sample - cur_sample))
                cur_sample = target_sample
            if kind == 1:
                synth.noteon(0, midi, vel)
            else:
                synth.noteoff(0, midi)
        if cur_sample < n_total:
            chunks.append(_fluidsynth_block(synth, n_total - cur_sample))
            cur_sample = n_total

        out = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float64)
    finally:
        synth.delete()

    return out[:n_total]


def _fluidsynth_block(synth, n_samples: int) -> np.ndarray:
    """Pull n_samples of mono float64 audio out of a pyfluidsynth Synth."""
    block = np.asarray(synth.get_samples(max(n_samples, 1)), dtype=np.float64)
    block = block.reshape(-1, 2).mean(axis=1) / 32768.0  # interleaved stereo int16 -> mono float
    return block[:n_samples]


def _velocity_from_envelope(env) -> int:
    """dBFS peak -> MIDI velocity: -60 dB -> 1, 0 dB -> 127, linear in dB."""
    peak_db = env.peak_db()
    v = int(round((max(-60.0, min(0.0, peak_db)) + 60.0) / 60.0 * 126.0)) + 1
    return min(127, max(1, v))
