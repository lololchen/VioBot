"""Timbre: fill Note.harmonics from the source audio (PRD F3, D-008).

Backends:
- "numpy" (default, always available): per-note STFT harmonic analysis.
  Essentia has no Windows wheels, so classical numpy DSP is the primary
  implementation; the formulas below ARE the Essentia definitions.
- "essentia" (optional, guarded import): SpectralPeaks -> HarmonicPeaks ->
  OddToEvenHarmonicEnergyRatio / Tristimulus / Inharmonicity. Must produce
  the same fields.

numpy backend spec:
- For each note: take analysis frames fully inside [onset, onset+duration]
  (Hann window, frame 4096 samples zero-padded x2, hop 1024 at the
  AudioBuffer's sample rate; skip notes shorter than one frame — leave
  harmonics=None).
- Per frame: magnitude spectrum; for harmonic k = 1..n_harmonics search the
  strongest local peak within ±tolerance_semitones/2 of k*f0 (f0 = note
  pitch_hz, or the f0_contour value at frame center when present); parabolic
  interpolation for peak frequency/amplitude. Missing peak -> amplitude 0.
- Note-level: harmonic_amps_db[k] = 20*log10(mean linear amp of harmonic k+1
  across frames, floor 1e-6). Ratios from note-mean LINEAR amplitudes a_k:
    odd_even_ratio = (a3²+a5²+a7²+...) / (a2²+a4²+a6²+...)   (fundamental excluded,
                      denominator floored at 1e-12)
    tristimulus    = (a1², a2²+a3²+a4², a5²+...) / Σa_k²
    inharmonicity  = Σ_k |f_meas_k - k*f0| / (k*f0) * a_k² / Σa_k²  (k >= 2)
- Deterministic; no RNG. Records meta.backends["timbre"] = "numpy-0.1.0".
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from .input_adapter import AudioBuffer
from .schema import Harmonics, Note, NoteSequence

_LOG_FLOOR = 1e-300  # guards log(0) during parabolic interpolation
_ENERGY_FLOOR = 1e-12  # matches the spec's odd/even-ratio denominator floor
_AMP_DB_FLOOR = 1e-6  # spec floor for harmonic_amps_db linear amplitude


@dataclass(frozen=True)
class TimbreConfig:
    backend: str = "numpy"        # "numpy" | "essentia"
    n_harmonics: int = 8
    frame_size: int = 4096
    hop_size: int = 1024
    zero_pad_factor: int = 2
    tolerance_semitones: float = 1.0  # search window width around k*f0


def add_harmonics(audio: AudioBuffer, seq: NoteSequence, config: TimbreConfig = TimbreConfig()) -> NoteSequence:
    """Return a new NoteSequence whose notes carry Harmonics blocks.

    Pure with respect to `seq` (input is not mutated — schema types are frozen).
    Notes too short for one analysis frame keep harmonics=None.
    """
    if config.backend == "numpy":
        return _add_harmonics_numpy(audio, seq, config)
    if config.backend == "essentia":
        return _add_harmonics_essentia(audio, seq, config)
    raise ValueError(f"unknown timbre backend: {config.backend!r}")


def _add_harmonics_numpy(audio: AudioBuffer, seq: NoteSequence, config: TimbreConfig) -> NoteSequence:
    new_notes = tuple(_analyze_note(audio, note, config) for note in seq.notes)
    new_backends = dict(seq.meta.backends)
    new_backends["timbre"] = "numpy-0.1.0"
    new_meta = replace(seq.meta, backends=new_backends)
    return replace(seq, notes=new_notes, meta=new_meta)


def _analyze_note(audio: AudioBuffer, note: Note, config: TimbreConfig) -> Note:
    sr = audio.sample_rate
    frame_size = config.frame_size
    hop = config.hop_size
    fft_size = frame_size * config.zero_pad_factor

    onset_sample = int(round(note.onset_s * sr))
    dur_samples = int(round(note.duration_s * sr))
    available = min(dur_samples, len(audio.pcm) - onset_sample)
    if available < frame_size:
        return note  # too short for one analysis frame -> harmonics stays None

    n_frames = 1 + (available - frame_size) // hop
    n_h = config.n_harmonics

    window = np.hanning(frame_size)
    window_sum = float(window.sum())
    amp_correction = window_sum / 2.0  # unit-amplitude sine -> ~1.0 linear amp
    freq_bins = np.fft.rfftfreq(fft_size, d=1.0 / sr)
    tol_ratio = 2.0 ** ((config.tolerance_semitones / 2.0) / 12.0)

    contour = note.f0_contour
    f0_note = note.pitch_hz

    amps = np.zeros((n_frames, n_h), dtype=np.float64)
    freqs = np.full((n_frames, n_h), np.nan, dtype=np.float64)

    for fi in range(n_frames):
        start = onset_sample + fi * hop
        frame = audio.pcm[start:start + frame_size].astype(np.float64)

        if contour is not None:
            t_center = (fi * hop + frame_size / 2.0) / sr  # relative to onset
            f0_frame = float(np.interp(t_center, contour.times_s, contour.f0_hz))
        else:
            f0_frame = f0_note

        padded = np.zeros(fft_size, dtype=np.float64)
        padded[:frame_size] = frame * window
        mag = np.abs(np.fft.rfft(padded))

        for k in range(1, n_h + 1):
            f_target = k * f0_frame
            low = f_target / tol_ratio
            high = f_target * tol_ratio
            idx = np.where((freq_bins >= low) & (freq_bins <= high))[0]

            best_i = -1
            best_mag = -1.0
            for i in idx:
                if 0 < i < len(mag) - 1 and mag[i] >= mag[i - 1] and mag[i] >= mag[i + 1]:
                    if mag[i] > best_mag:
                        best_mag = float(mag[i])
                        best_i = int(i)

            if best_i < 0:
                continue  # missing peak -> amplitude stays 0, freq stays NaN

            alpha = math.log(max(mag[best_i - 1], _LOG_FLOOR))
            beta = math.log(max(mag[best_i], _LOG_FLOOR))
            gamma = math.log(max(mag[best_i + 1], _LOG_FLOOR))
            denom = alpha - 2.0 * beta + gamma
            p = 0.0 if denom == 0.0 else 0.5 * (alpha - gamma) / denom
            peak_bin = best_i + p
            interp_log_mag = beta - 0.25 * (alpha - gamma) * p

            amps[fi, k - 1] = math.exp(interp_log_mag) / amp_correction
            freqs[fi, k - 1] = peak_bin * sr / fft_size

    a = amps.mean(axis=0)  # note-mean linear amplitude per harmonic (missing frames count as 0)
    harmonic_amps_db = tuple(20.0 * math.log10(max(v, _AMP_DB_FLOOR)) for v in a)

    total_energy = max(float(np.sum(a ** 2)), _ENERGY_FLOOR)

    odd_energy = sum(a[k - 1] ** 2 for k in range(3, n_h + 1, 2))
    even_energy = sum(a[k - 1] ** 2 for k in range(2, n_h + 1, 2))
    odd_even_ratio = float(odd_energy / max(even_energy, _ENERGY_FLOOR))

    t1 = float(a[0] ** 2 / total_energy)
    t2 = float(sum(a[k - 1] ** 2 for k in range(2, min(4, n_h) + 1)) / total_energy)
    t3 = float(sum(a[k - 1] ** 2 for k in range(5, n_h + 1)) / total_energy)

    inharm_num = 0.0
    for k in range(2, n_h + 1):
        col = freqs[:, k - 1]
        valid = ~np.isnan(col)
        f_meas_k = float(col[valid].mean()) if valid.any() else k * f0_note
        inharm_num += abs(f_meas_k - k * f0_note) / (k * f0_note) * (a[k - 1] ** 2)
    inharmonicity = float(inharm_num / total_energy)

    harmonics = Harmonics(
        harmonic_amps_db=harmonic_amps_db,
        odd_even_ratio=odd_even_ratio,
        tristimulus=(t1, t2, t3),
        inharmonicity=inharmonicity,
    )
    return replace(note, harmonics=harmonics)


def _add_harmonics_essentia(audio: AudioBuffer, seq: NoteSequence, config: TimbreConfig) -> NoteSequence:
    try:
        import essentia  # noqa: F401
        import essentia.standard as _es  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "timbre backend='essentia' requires the 'essentia' package, which is not "
            "installed in this environment (essentia publishes no official Windows "
            "wheels). Use backend='numpy' (the default) here, or install essentia "
            "from conda-forge / build from source on Linux or macOS: "
            "`pip install essentia` (or `conda install -c conda-forge essentia`)."
        ) from exc
    raise NotImplementedError("essentia timbre backend not yet implemented")
