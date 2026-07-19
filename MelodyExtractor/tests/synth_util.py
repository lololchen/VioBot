"""Deterministic audio synthesis helpers shared by tests and fixture generation.

Everything here is pure numpy, float64, fixed phases, no RNG — byte-identical
output across runs is a repo requirement (fixtures double as determinism probes).
"""
from __future__ import annotations

import numpy as np

DEFAULT_SR = 16000
# A bright sawtooth-ish recipe with known ground-truth timbre features:
# linear amplitude of harmonic k is 1/k (k = 1..6).
DEFAULT_HARMONIC_AMPS = tuple(1.0 / k for k in range(1, 7))


def harmonic_tone(freq_hz: float, duration_s: float, sample_rate: int = DEFAULT_SR,
                  harmonic_amps=DEFAULT_HARMONIC_AMPS, peak: float = 0.5,
                  fade_s: float = 0.01) -> np.ndarray:
    """Steady harmonic tone, phase 0 partials, raised-cosine fades, peak-normalized."""
    n = int(round(duration_s * sample_rate))
    t = np.arange(n, dtype=np.float64) / sample_rate
    x = np.zeros(n, dtype=np.float64)
    nyquist = sample_rate / 2.0
    for k, a in enumerate(harmonic_amps, start=1):
        f = freq_hz * k
        if f >= nyquist:
            break
        x += a * np.sin(2.0 * np.pi * f * t)
    m = np.max(np.abs(x))
    if m > 0:
        x *= peak / m
    nf = min(int(round(fade_s * sample_rate)), n // 2)
    if nf > 0:
        ramp = 0.5 * (1.0 - np.cos(np.pi * np.arange(nf, dtype=np.float64) / nf))
        x[:nf] *= ramp
        x[-nf:] *= ramp[::-1]
    return x


def sequence_audio(notes, total_s: float | None = None, sample_rate: int = DEFAULT_SR,
                   harmonic_amps=DEFAULT_HARMONIC_AMPS, peak: float = 0.5) -> np.ndarray:
    """Mix (pitch_hz, onset_s, duration_s) triples into one mono track.

    Concurrent notes sum; the mix is rescaled only if it clips.
    """
    end = max(o + d for _, o, d in notes)
    dur = max(end, total_s or 0.0)
    n = int(round(dur * sample_rate))
    x = np.zeros(n, dtype=np.float64)
    for pitch_hz, onset_s, duration_s in notes:
        tone = harmonic_tone(pitch_hz, duration_s, sample_rate, harmonic_amps, peak)
        i0 = int(round(onset_s * sample_rate))
        x[i0:i0 + len(tone)] += tone
    m = np.max(np.abs(x))
    if m > 0.99:
        x *= 0.99 / m
    return x


def write_wav(path, pcm: np.ndarray, sample_rate: int = DEFAULT_SR) -> None:
    # scipy, not soundfile: libsndfile stamps float WAVs with a timestamped
    # PEAK chunk, which breaks the byte-identical-output requirement.
    from scipy.io import wavfile

    wavfile.write(str(path), sample_rate, pcm.astype(np.float32))


def write_ground_truth_midi(path, notes, velocity: int = 90) -> None:
    """(pitch_hz, onset_s, duration_s) triples -> ground-truth MIDI file."""
    import pretty_midi

    from melody_extractor.schema import hz_to_midi

    pm = pretty_midi.PrettyMIDI(resolution=960)
    inst = pretty_midi.Instrument(program=40, name="ground_truth")
    for pitch_hz, onset_s, duration_s in notes:
        inst.notes.append(pretty_midi.Note(
            velocity=velocity,
            pitch=int(round(hz_to_midi(pitch_hz))),
            start=onset_s,
            end=onset_s + duration_s,
        ))
    pm.instruments.append(inst)
    pm.write(str(path))
