"""Transcriber: AudioBuffer -> NoteSequence (frame track + segmented notes).

Backends (PRD "Algorithm Decisions", D-007, D-011):
- Mono: CREPE (f0 + confidence, Viterbi smoothing) when installed — guarded
  import, TensorFlow-heavy. Always available fallback: deterministic
  numpy YIN (de Cheveigné & Kawahara 2002) so the pipeline and CI run with
  core deps only. backend="auto" prefers crepe, falls back to yin.
- Poly: basic-pitch (note events + pitch bends + amplitude-scaled velocity) —
  guarded import. No fallback; raise a clear ImportError mentioning
  `pip install melody-extractor[poly]`.

Contracts:
- CREPE gets 16 kHz mono (it expects that); basic-pitch loads its own 22050 Hz
  internally — never assume a shared sample rate (module CLAUDE.md gotcha).
- Frame track: hop MonoConfig.hop_s, f0_hz=0 on unvoiced frames, voicing =
  model confidence (yin: 1 - cmndf minimum, clipped to [0,1]), amp_db = frame
  RMS in dBFS (floor -80).
- Segmentation (frames -> notes): voiced runs where voicing >= voicing_threshold;
  split when local f0 deviates from the running note median by
  > split_semitones; discard notes shorter than min_note_s; merge gaps
  < merge_gap_s between same-pitch (within split_semitones) neighbors.
  Note.pitch_hz = median f0 of its frames; f0_contour = the note's frames;
  amp_db_envelope sampled from frame amp at envelope_hop_s; confidence =
  mean voicing.
- Everything deterministic: fixed hop alignment, no RNG. Meta.backends
  records e.g. {"transcriber": "yin-0.1.0"}.

== YIN implementation notes ==
Per de Cheveigné & Kawahara 2002. Frame size (the full per-frame buffer) is
the smallest power of two >= 2*sample_rate/fmin_hz; the difference function's
"integration window" w is half of that. This choice means tau (0..w) never
needs samples beyond the frame buffer (see yin_track for the proof sketch),
so the whole per-frame difference function is one FFT-based autocorrelation
plus a cumulative-sum energy term — no O(frames * taus * window) loop.
Frames are stacked into 2D arrays and FFT'd with batched calls (axis=1),
processed in blocks of _YIN_BLOCK_FRAMES frames so peak memory stays bounded
(~100 MB) regardless of song length — a 5-minute song stacked as one matrix
needs >1 GB across the intermediate arrays, enough to take down the process
on smaller machines. Every per-frame value is computed row-independently, so
blocking is byte-identical to the single-batch formulation (verified).
Only the small, inherently-sequential peak-picking / parabolic-interpolation
step loops per frame (O(taus), not O(taus * window)).
"""
from __future__ import annotations

import bisect
import importlib.metadata
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .input_adapter import AudioBuffer
from .schema import AmpEnvelope, F0Contour, FrameTrack, Meta, Note, NoteSequence, hz_to_midi, midi_to_hz

YIN_VERSION = "0.1.0"

# Poly notes carry a raw-dB envelope derived from the source audio (module
# CLAUDE.md gotcha: basic-pitch's amplitude-scaled velocity is not a
# substitute for real levels). PolyConfig has no envelope-hop field of its
# own (it is not a per-note transcription knob); reuse MonoConfig's default.
_POLY_ENVELOPE_HOP_S = 0.02


@dataclass(frozen=True)
class MonoConfig:
    backend: str = "auto"          # "auto" | "crepe" | "yin"
    hop_s: float = 0.01
    fmin_hz: float = 60.0
    fmax_hz: float = 2200.0
    voicing_threshold: float = 0.5
    min_note_s: float = 0.06
    merge_gap_s: float = 0.03
    split_semitones: float = 0.8
    envelope_hop_s: float = 0.02


@dataclass(frozen=True)
class PolyConfig:
    onset_threshold: float = 0.5
    frame_threshold: float = 0.3
    min_note_s: float = 0.06


def transcribe_mono(audio: AudioBuffer, config: MonoConfig = MonoConfig()) -> NoteSequence:
    """Monophonic transcription: frame-level f0/voicing/amp + segmented notes."""
    if config.backend not in ("auto", "crepe", "yin"):
        raise ValueError(f"MonoConfig.backend must be 'auto', 'crepe', or 'yin' (got {config.backend!r})")

    used_crepe = False
    if config.backend in ("auto", "crepe"):
        try:
            import crepe  # noqa: F401
            used_crepe = True
        except ImportError:
            if config.backend == "crepe":
                raise ImportError(
                    "MonoConfig(backend='crepe') requires the 'crepe' package "
                    "(pulls TensorFlow). Install it with: pip install melody-extractor[mono-dnn]"
                )
            used_crepe = False

    if used_crepe:
        f0_hz, voicing, amp_db, hop_s, backend_tag = _transcribe_mono_crepe(audio, config)
    else:
        f0_hz, voicing, amp_db = yin_track(audio.pcm, audio.sample_rate, config)
        hop_s = config.hop_s
        backend_tag = f"yin-{YIN_VERSION}"

    track = FrameTrack(
        hop_s=hop_s,
        f0_hz=tuple(f0_hz.tolist()),
        voicing=tuple(voicing.tolist()),
        amp_db=tuple(amp_db.tolist()),
        start_s=0.0,
        name="f0",
    )
    notes = _segment_notes(f0_hz, voicing, amp_db, hop_s, config)
    meta = Meta(
        source=audio.source,
        source_kind="audio",
        sample_rate=audio.sample_rate,
        backends={"transcriber": backend_tag},
    )
    seq = NoteSequence(notes=notes, features=(track,), meta=meta)
    return seq.sorted().validate()


def transcribe_poly(audio: AudioBuffer, config: PolyConfig = PolyConfig()) -> NoteSequence:
    """Polyphonic transcription via basic-pitch (optional dep)."""
    try:
        from basic_pitch.inference import predict
    except ImportError as exc:
        raise ImportError(
            "transcribe_poly requires the 'basic-pitch' package (pulls TensorFlow). "
            "Install it with: pip install melody-extractor[poly]"
        ) from exc

    # basic-pitch resamples internally to 22050 Hz; hand it the AudioBuffer's
    # own pcm+sample_rate as-is (never pre-resample here — module CLAUDE.md
    # gotcha: never assume a shared sample rate across backends).
    # Written via scipy.io.wavfile, not soundfile: float WAVs written by
    # libsndfile carry a timestamped PEAK chunk that breaks byte-determinism
    # (module CLAUDE.md); basic-pitch only reads this file, but we still
    # avoid the nondeterministic writer on principle.
    from scipy.io import wavfile

    with tempfile.TemporaryDirectory() as tmp:
        wav_path = Path(tmp) / "poly_input.wav"
        pcm = np.asarray(audio.pcm, dtype=np.float32)
        wavfile.write(str(wav_path), audio.sample_rate, pcm)
        _model_output, _midi_data, note_events = predict(
            str(wav_path),
            onset_threshold=config.onset_threshold,
            frame_threshold=config.frame_threshold,
        )

    notes = []
    for event in note_events:
        start_s, end_s, pitch_midi, amplitude = event[0], event[1], event[2], event[3]
        pitch_bends = event[4] if len(event) > 4 else None
        duration_s = float(end_s) - float(start_s)
        if duration_s < config.min_note_s:
            continue

        base_hz = midi_to_hz(float(pitch_midi))
        contour = None
        if pitch_bends is not None and len(pitch_bends) > 0:
            bend_semitones = np.asarray(pitch_bends, dtype=np.float64)
            f0s = [midi_to_hz(float(pitch_midi) + float(b)) for b in bend_semitones]
            times = np.linspace(0.0, duration_s, len(f0s))
            contour = F0Contour(times_s=tuple(times.tolist()), f0_hz=tuple(f0s))
            pitch_hz = float(np.median(f0s))
        else:
            pitch_hz = base_hz

        velocity = int(min(127, max(0, round(float(amplitude) * 127.0))))
        envelope = _note_amp_envelope_from_audio(
            audio.pcm, audio.sample_rate, float(start_s), duration_s, _POLY_ENVELOPE_HOP_S
        )
        notes.append(Note(
            pitch_hz=pitch_hz,
            onset_s=float(start_s),
            duration_s=duration_s,
            amp_db_envelope=envelope,
            velocity=velocity,
            f0_contour=contour,
        ))

    backend_tag = f"basic-pitch-{_package_version('basic-pitch')}"
    meta = Meta(
        source=audio.source,
        source_kind="audio",
        sample_rate=audio.sample_rate,
        backends={"transcriber": backend_tag},
    )
    seq = NoteSequence(notes=tuple(notes), features=(), meta=meta)
    return seq.sorted().validate()


def yin_track(pcm, sample_rate: int, config: MonoConfig = MonoConfig()):
    """Deterministic YIN f0 tracker (numpy). Returns (f0_hz, voicing, amp_db)
    arrays at config.hop_s hop. Exposed separately for tests."""
    pcm = np.asarray(pcm, dtype=np.float64)
    if pcm.size == 0:
        empty = np.zeros(0, dtype=np.float64)
        return empty, empty.copy(), empty.copy()

    frame_size = _next_pow2(2.0 * sample_rate / config.fmin_hz)
    w = frame_size // 2
    hop = max(1, int(round(config.hop_s * sample_rate)))

    tau_min = max(1, int(math.floor(sample_rate / config.fmax_hz)))
    tau_max = int(math.ceil(sample_rate / config.fmin_hz))
    tau_max = max(2, min(tau_max, w))
    tau_min = max(1, min(tau_min, tau_max - 1))

    n_frames = 1 + (len(pcm) - 1) // hop
    padded = np.concatenate([pcm, np.zeros(frame_size, dtype=np.float64)])

    f0 = np.zeros(n_frames, dtype=np.float64)
    voicing = np.zeros(n_frames, dtype=np.float64)
    amp_rms = np.zeros(n_frames, dtype=np.float64)
    absolute_threshold = 0.1
    taus = np.arange(tau_max + 1)
    taus_pos = np.arange(1, tau_max + 1, dtype=np.float64)

    # Blocked over frames purely to bound memory; every value below is a
    # row-independent function of its own frame, so the result is identical
    # to stacking all frames at once (see module docstring).
    for block_start in range(0, n_frames, _YIN_BLOCK_FRAMES):
        block_end = min(block_start + _YIN_BLOCK_FRAMES, n_frames)
        starts = np.arange(block_start, block_end) * hop
        idx = starts[:, None] + np.arange(frame_size)[None, :]
        frames = padded[idx]  # (n_block, frame_size)
        n_block = frames.shape[0]

        ref = frames[:, :w]  # fixed reference window, does not shift with tau

        # Cross term c(tau) = sum_{j=0}^{w-1} ref[j] * frames[j+tau] for all tau in
        # [0, tau_max] at once, via one batched FFT-based circular correlation.
        # Proof this is exact (no wraparound) for tau in [0, w]: with N=frame_size
        # =2w and ref zero-padded to N, ref's nonzero support (length w) intersects
        # the correlation sum's index window [tau, tau+w-1] without wrapping
        # whenever tau+w-1 <= N-1, i.e. tau <= w -- true by construction.
        R = np.fft.rfft(ref, n=frame_size, axis=1)
        B = np.fft.rfft(frames, n=frame_size, axis=1)
        cross = np.fft.irfft(B * np.conj(R), n=frame_size, axis=1)[:, : tau_max + 1]

        # Energy terms via cumulative sum (O(1) per tau after one cumsum pass).
        e_ref = np.sum(ref ** 2, axis=1, keepdims=True)  # (n_block, 1)
        sq = frames ** 2
        csum = np.concatenate([np.zeros((n_block, 1)), np.cumsum(sq, axis=1)], axis=1)
        e_shift = csum[:, taus + w] - csum[:, taus]  # (n_block, tau_max+1)

        d = e_ref + e_shift - 2.0 * cross
        d = np.maximum(d, 0.0)  # guard tiny negative FFT rounding error

        # Cumulative mean normalized difference function (CMNDF).
        cmndf = np.ones_like(d)
        running = np.cumsum(d[:, 1:], axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = d[:, 1:] * taus_pos[None, :] / running
        cmndf[:, 1:] = np.where(running > 0.0, ratio, 1.0)
        cmndf[:, 0] = 1.0

        amp_rms[block_start:block_end] = np.sqrt(np.mean(ref ** 2, axis=1))

        for bi in range(n_block):
            i = block_start + bi
            f0[i], voicing[i] = _pick_frame_f0(
                d[bi], cmndf[bi], tau_min, tau_max, absolute_threshold, sample_rate
            )

    voiced = (voicing >= config.voicing_threshold) & (f0 >= config.fmin_hz) & (f0 <= config.fmax_hz)
    f0 = np.where(voiced, f0, 0.0)

    amp_db = 20.0 * np.log10(np.maximum(amp_rms, 1e-12))
    amp_db = np.maximum(amp_db, -80.0)

    return f0, voicing, amp_db


# Frames per YIN block (see yin_track): 2048 frames x 1024-sample frames in
# float64 keeps each of the ~6 concurrent intermediate arrays at ~16 MB.
_YIN_BLOCK_FRAMES = 2048


def _pick_frame_f0(d_row: np.ndarray, cmndf_row: np.ndarray, tau_min: int, tau_max: int,
                    absolute_threshold: float, sample_rate: int) -> "tuple[float, float]":
    """Per-frame tau selection + sub-sample refinement (unchanged from the
    original in-loop body of yin_track; factored out for the blocked layout).
    Returns (f0_hz, voicing) for one frame."""
    window = cmndf_row[tau_min : tau_max + 1]
    below = np.nonzero(window < absolute_threshold)[0]
    if below.size:
        j = int(below[0])
        while j + 1 < window.size and window[j + 1] <= window[j]:
            j += 1
    else:
        j = int(np.argmin(window))
    tau_star = tau_min + j
    cmndf_min = float(cmndf_row[tau_star])
    voicing = min(1.0, max(0.0, 1.0 - cmndf_min))

    shift = 0.0
    if tau_min <= tau_star - 1 and tau_star + 1 <= tau_max:
        # Sub-sample refinement around tau_star. A plain 3-point parabolic
        # fit on d(tau) assumes a locally quadratic dip; that is only the
        # small-angle limit of d(tau)'s true shape, a raised cosine of
        # angular rate w=2*pi/tau0 (d(tau) ~ 2E(1-cos(w*(tau-tau0))) for a
        # near-stationary tone). At short periods (high pitch, few
        # samples/cycle) w is large enough that the quadratic
        # approximation measurably biases the estimate, and real tones'
        # upper harmonics add extra fine-grained ripple that a bare
        # 3-point fit picks up directly. _local_avg smooths that ripple
        # out of the 3 stencil points (contamination from harmonic k has
        # tau-wavelength ~tau_star/k, much finer than the fundamental's
        # own ~tau_star-wide dip, so a small local average suppresses it
        # while barely touching the fundamental's shape) and
        # _trig_interpolate_shift inverts the cosine model exactly
        # instead of approximating it with a parabola (it reduces
        # algebraically to the standard parabolic formula as tau_star ->
        # large, i.e. low pitch). Verified empirically against harmonic
        # tones (synth_util.harmonic_tone, 6 harmonics) from 196-1318 Hz.
        x0 = _local_avg(d_row, tau_star - 1, _INTERP_SMOOTH_HALFWIDTH)
        x1 = _local_avg(d_row, tau_star, _INTERP_SMOOTH_HALFWIDTH)
        x2 = _local_avg(d_row, tau_star + 1, _INTERP_SMOOTH_HALFWIDTH)
        shift = _trig_interpolate_shift(x0, x1, x2, tau_star)
        shift = max(-0.5, min(0.5, shift))
    tau_refined = tau_star + shift
    tau_refined = max(tau_refined, 0.5)
    return sample_rate / tau_refined, voicing


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _next_pow2(x: float) -> int:
    return 1 << max(0, int(math.ceil(math.log2(max(x, 1.0)))))


def _sorted_median(sorted_values: "list[float]") -> float:
    """Median of an already-sorted list, reproducing np.median's arithmetic
    bit-exactly: the middle element for odd n; for even n the mean of the two
    middles, (a + b) * 0.5, which is the same IEEE-754 result as np.mean's
    (a + b) / 2 (scaling by a power of two is exact)."""
    m = len(sorted_values)
    half = m // 2
    if m % 2:
        return sorted_values[half]
    return (sorted_values[half - 1] + sorted_values[half]) * 0.5


# Half-width (in tau samples) of the local average applied to the 3 points
# fed into _trig_interpolate_shift (see the comment at its call site in
# yin_track). Empirically tuned: 0 (no smoothing) leaves a harmonic-induced
# bias of ~0.3% at 1318 Hz; half-width 2 (5-tap average) brings every tested
# tone from 196-1318 Hz to well under 0.1% while leaving voicing/tau_star
# selection (computed from the unsmoothed d/cmndf) untouched.
_INTERP_SMOOTH_HALFWIDTH = 2


def _local_avg(row: np.ndarray, center: int, half_width: int) -> float:
    """Mean of row[center-half_width : center+half_width+1], clipped to bounds."""
    lo = max(0, center - half_width)
    hi = min(row.shape[0], center + half_width + 1)
    return float(np.mean(row[lo:hi]))


def _trig_interpolate_shift(x0: float, x1: float, x2: float, tau_star: int) -> float:
    """Sub-sample shift epsilon such that tau_star+epsilon locates the minimum
    of the model x(tau) = A*(1 - cos(w*(tau-tau_star-epsilon))), w=2*pi/tau_star,
    given the 3 samples x0=x(tau_star-1), x1=x(tau_star), x2=x(tau_star+1).

    Exact (no small-angle assumption) for a single stationary sinusoid; reduces
    algebraically to the classic parabolic-interpolation formula
    0.5*(x0-x2)/(x0-2*x1+x2) in the w->0 (large tau_star / low pitch) limit.
    """
    w = 2.0 * math.pi / tau_star
    sin_w = math.sin(w)
    cos_w = math.cos(w)
    num_sin = (x0 - x2) / sin_w if abs(sin_w) > 1e-9 else 0.0
    num_cos = (x2 + x0 - 2.0 * x1) / (1.0 - cos_w) if abs(1.0 - cos_w) > 1e-9 else 0.0
    if num_sin == 0.0 and num_cos == 0.0:
        return 0.0
    return math.atan2(num_sin, num_cos) / w


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _frame_rms_db(pcm: np.ndarray, sample_rate: int, hop_s: float, n_frames: int) -> np.ndarray:
    """RMS-in-dBFS track at a fixed hop, non-overlapping blocks (floor -80)."""
    pcm = np.asarray(pcm, dtype=np.float64)
    hop = max(1, int(round(hop_s * sample_rate)))
    padded = np.concatenate([pcm, np.zeros(hop, dtype=np.float64)])
    starts = np.arange(n_frames) * hop
    idx = starts[:, None] + np.arange(hop)[None, :]
    blocks = padded[idx]
    rms = np.sqrt(np.mean(blocks ** 2, axis=1))
    db = 20.0 * np.log10(np.maximum(rms, 1e-12))
    return np.maximum(db, -80.0)


def _transcribe_mono_crepe(audio: AudioBuffer, config: MonoConfig):
    try:
        import crepe
    except ImportError as exc:
        raise ImportError(
            "MonoConfig(backend='crepe') requires the 'crepe' package (pulls "
            "TensorFlow). Install it with: pip install melody-extractor[mono-dnn]"
        ) from exc

    if audio.sample_rate != 16000:
        raise ValueError(
            f"CREPE requires 16 kHz mono input (got sample_rate={audio.sample_rate}); "
            "resample via input_adapter.load_audio first."
        )

    step_size_ms = max(1.0, config.hop_s * 1000.0)
    _time, frequency, confidence, _activation = crepe.predict(
        np.asarray(audio.pcm, dtype=np.float32),
        audio.sample_rate,
        model_capacity="full",
        viterbi=True,
        step_size=step_size_ms,
        verbose=0,
    )
    f0_hz = np.asarray(frequency, dtype=np.float64)
    voicing = np.asarray(confidence, dtype=np.float64)
    hop_s = step_size_ms / 1000.0
    amp_db = _frame_rms_db(audio.pcm, audio.sample_rate, hop_s, len(f0_hz))

    out_of_range = (
        (f0_hz < config.fmin_hz) | (f0_hz > config.fmax_hz) | (voicing < config.voicing_threshold)
    )
    f0_hz = np.where(out_of_range, 0.0, f0_hz)

    return f0_hz, voicing, amp_db, hop_s, f"crepe-{_package_version('crepe')}"


def _segment_notes(f0_hz: np.ndarray, voicing: np.ndarray, amp_db: np.ndarray,
                    hop_s: float, config: MonoConfig) -> "tuple[Note, ...]":
    n = len(f0_hz)
    if n == 0:
        return ()

    voiced = (voicing >= config.voicing_threshold) & (f0_hz >= config.fmin_hz) & (f0_hz <= config.fmax_hz) & (f0_hz > 0.0)

    # Step 1: contiguous voiced runs, further split wherever the local f0
    # deviates from the running note median by more than split_semitones.
    # The running median is maintained incrementally (bisect.insort into a
    # per-segment sorted list + _sorted_median) instead of re-running
    # np.median over the growing segment every frame -- that was
    # O(len(note)^2) and alone took ~2 minutes on a 5-minute song with
    # sustained notes. _sorted_median reproduces np.median's arithmetic
    # exactly, so the produced notes are value-identical (verified).
    raw_segments: list[list[int]] = []
    seg_sorted: list[list[float]] = []  # sorted f0 values, parallel to raw_segments
    current: list[int] = []
    current_sorted: list[float] = []
    for i in range(n):
        if not voiced[i]:
            if current:
                raw_segments.append(current)
                seg_sorted.append(current_sorted)
            current = []
            current_sorted = []
            continue
        if current:
            median_hz = _sorted_median(current_sorted)
            if abs(hz_to_midi(float(f0_hz[i])) - hz_to_midi(median_hz)) > config.split_semitones:
                raw_segments.append(current)
                seg_sorted.append(current_sorted)
                current = []
                current_sorted = []
        current.append(i)
        bisect.insort(current_sorted, float(f0_hz[i]))
    if current:
        raw_segments.append(current)
        seg_sorted.append(current_sorted)

    if not raw_segments:
        return ()

    # Step 2: merge neighbouring segments across short, pitch-compatible gaps.
    # (sorted(a + b) over two sorted runs is effectively linear -- Timsort
    # gallops over pre-sorted runs.)
    merged: list[dict] = []
    for seg, s_sorted in zip(raw_segments, seg_sorted):
        first, last = seg[0], seg[-1]
        median_hz = _sorted_median(s_sorted)
        if merged:
            prev = merged[-1]
            gap_s = (first - prev["last"] - 1) * hop_s
            pitch_ok = abs(hz_to_midi(median_hz) - hz_to_midi(prev["median_hz"])) <= config.split_semitones
            if gap_s < config.merge_gap_s and pitch_ok:
                merged_sorted = sorted(prev["sorted"] + s_sorted)
                prev["last"] = last
                prev["frames"] = prev["frames"] + seg
                prev["sorted"] = merged_sorted
                prev["median_hz"] = _sorted_median(merged_sorted)
                continue
        merged.append({"first": first, "last": last, "median_hz": median_hz,
                       "frames": list(seg), "sorted": list(s_sorted)})

    # Step 3: drop notes shorter than min_note_s, build Note objects.
    notes: list[Note] = []
    for seg in merged:
        first, last, frame_idx = seg["first"], seg["last"], seg["frames"]
        duration_s = (last - first + 1) * hop_s
        if duration_s < config.min_note_s:
            continue

        pitch_hz = _sorted_median(seg["sorted"])

        contour_times = tuple((j - first) * hop_s for j in frame_idx)
        contour_f0 = tuple(float(f0_hz[j]) for j in frame_idx)
        f0_contour = F0Contour(times_s=contour_times, f0_hz=contour_f0)

        span = amp_db[first : last + 1]
        n_env = max(2, int(math.ceil(duration_s / config.envelope_hop_s)) + 1)
        env_times = np.linspace(0.0, duration_s, n_env)
        span_times = np.arange(len(span)) * hop_s
        env_db = np.interp(env_times, span_times, span)
        amp_envelope = AmpEnvelope(times_s=tuple(env_times.tolist()), amp_db=tuple(env_db.tolist()))

        confidence = float(np.clip(np.mean(voicing[first : last + 1]), 0.0, 1.0))

        notes.append(Note(
            pitch_hz=pitch_hz,
            onset_s=first * hop_s,
            duration_s=duration_s,
            amp_db_envelope=amp_envelope,
            confidence=confidence,
            f0_contour=f0_contour,
        ))

    return tuple(notes)


def _note_amp_envelope_from_audio(pcm: np.ndarray, sample_rate: int, onset_s: float,
                                   duration_s: float, hop_s: float) -> AmpEnvelope:
    """RMS-in-dB envelope sampled at hop_s across [onset_s, onset_s+duration_s)
    of the raw source audio (poly path: basic-pitch's velocity is amplitude-
    scaled 0-127 only, keep the real levels alongside it)."""
    pcm = np.asarray(pcm, dtype=np.float64)
    i0 = max(0, int(round(onset_s * sample_rate)))
    i1 = min(len(pcm), int(round((onset_s + duration_s) * sample_rate)))
    i1 = max(i1, i0 + 1)
    seg = pcm[i0:i1]

    n_pts = max(2, int(math.ceil(duration_s / hop_s)) + 1)
    times = np.linspace(0.0, duration_s, n_pts)
    win = max(1, int(round(hop_s * sample_rate)))
    half = win // 2

    seg_padded = np.concatenate([np.zeros(half, dtype=np.float64), seg, np.zeros(half + win, dtype=np.float64)])
    centers = np.round(times * sample_rate).astype(int) + half

    values = np.empty(n_pts, dtype=np.float64)
    for k, c in enumerate(centers):
        block = seg_padded[max(0, c - half) : c + half + 1]
        rms = math.sqrt(float(np.mean(block ** 2))) if block.size else 0.0
        values[k] = 20.0 * math.log10(max(rms, 1e-12))
    values = np.maximum(values, -80.0)

    return AmpEnvelope(times_s=tuple(times.tolist()), amp_db=tuple(values.tolist()))
