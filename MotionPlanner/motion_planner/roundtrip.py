"""Round-trip validation — target NoteSequence vs simulated execution (D-028).

The reference for mir_eval is the plan's REALIZED timing (note_plan
realized_onset/duration with the target's pitches): the metrics measure how
faithfully the motion tracks execute the plan. Planning distortion (realized
vs score timing — D-024 rolls, lateness shifts) is reported separately as
onset-deviation statistics, so the two error sources never blur.

Both audio renders use the SAME melody_extractor.soundsim additive renderer,
so every audible difference is attributable to the plan/simulation alone.
This module is the algorithm-validation path for MotionPlanner (module
CLAUDE.md "Definition of done").
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from melody_extractor.schema import NoteSequence

from .schema import MotionScore
from .simulate import simulate

ROUNDTRIP_VERSION = "roundtrip-0.1.0"

_FRAME_HOP_S = 0.01


@dataclass(frozen=True)
class RoundtripResult:
    metrics: dict
    predicted: NoteSequence
    listen_dir: Optional[Path] = None


def _note_arrays(intervals_pitches) -> "tuple[np.ndarray, np.ndarray]":
    if not intervals_pitches:
        return np.zeros((0, 2)), np.zeros(0)
    intervals = np.array([[on, on + du] for on, du, _ in intervals_pitches])
    pitches = np.array([hz for _, _, hz in intervals_pitches])
    return intervals, pitches


def _melody_frames(intervals, pitches, t_max: float) -> np.ndarray:
    """Dominant (highest-pitch) melody line sampled on the shared frame grid."""
    n = int(np.ceil(t_max / _FRAME_HOP_S)) + 1
    out = np.zeros(n)
    for (on, off), hz in zip(intervals, pitches):
        k0, k1 = int(np.ceil(on / _FRAME_HOP_S)), int(np.floor(off / _FRAME_HOP_S))
        for k in range(max(k0, 0), min(k1 + 1, n)):
            out[k] = max(out[k], hz)
    return out


def roundtrip(target: NoteSequence, score: MotionScore, out_dir: "str | Path | None" = None,
              render: bool = True) -> RoundtripResult:
    import mir_eval

    predicted = simulate(score)
    target = target.sorted()
    t_notes = target.notes

    # Reference = realized plan timing with target pitches.
    ref = [(p.realized_onset_s, p.realized_duration_s, t_notes[p.note_index].pitch_hz)
           for p in score.note_plan]
    est = [(n.onset_s, n.duration_s, n.pitch_hz) for n in predicted.sorted().notes]
    ref_i, ref_p = _note_arrays(ref)
    est_i, est_p = _note_arrays(est)

    metrics: dict = {}
    if len(ref) and len(est):
        onset_p, onset_r, onset_f, _ = mir_eval.transcription.precision_recall_f1_overlap(
            ref_i, ref_p, est_i, est_p, onset_tolerance=0.05, pitch_tolerance=50.0,
            offset_ratio=None)
        metrics["onset_pitch_f1"] = onset_f
        op, orr, of_, _ = mir_eval.transcription.precision_recall_f1_overlap(
            ref_i, ref_p, est_i, est_p, onset_tolerance=0.05, pitch_tolerance=1e9,
            offset_ratio=None)
        metrics["onset_f1"] = of_
    else:
        metrics["onset_pitch_f1"] = 0.0
        metrics["onset_f1"] = 0.0 if ref else 1.0

    t_max = max([i[1] for i in ref_i.tolist()] + [i[1] for i in est_i.tolist()] + [1.0])
    ref_line = _melody_frames(ref_i, ref_p, t_max)
    est_line = _melody_frames(est_i, est_p, t_max)
    times = np.arange(len(ref_line)) * _FRAME_HOP_S
    ref_voiced = ref_line > 0
    if ref_voiced.any():
        rpa = mir_eval.melody.raw_pitch_accuracy(
            ref_line > 0, mir_eval.melody.hz2cents(ref_line),
            est_line > 0, mir_eval.melody.hz2cents(est_line))
        metrics["rpa"] = rpa
    else:
        metrics["rpa"] = 1.0

    # Planning distortion: realized vs score onsets (D-024 rolls, lateness).
    devs = [abs(p.realized_onset_s - p.onset_s) for p in score.note_plan]
    metrics["mean_onset_shift_s"] = float(np.mean(devs)) if devs else 0.0
    metrics["max_onset_shift_s"] = float(np.max(devs)) if devs else 0.0
    metrics["n_target_notes"] = float(len(ref))
    metrics["n_predicted_notes"] = float(len(est))
    silenced = sum(1 for p in score.note_plan
                   if not _covered(p.realized_onset_s, p.realized_duration_s, est_i))
    metrics["silenced_notes"] = float(silenced)

    listen_dir = None
    if render and out_dir is not None:
        from melody_extractor.soundsim import RenderConfig, render as render_wav

        listen_dir = Path(out_dir)
        listen_dir.mkdir(parents=True, exist_ok=True)
        cfg = RenderConfig()
        render_wav(target, listen_dir / "target_render.wav", cfg)
        if predicted.notes:
            render_wav(predicted, listen_dir / "predicted_render.wav", cfg)
    return RoundtripResult(metrics=metrics, predicted=predicted, listen_dir=listen_dir)


def _covered(onset: float, duration: float, est_intervals: np.ndarray) -> bool:
    """Does any predicted note overlap the middle of this planned interval?"""
    mid = onset + duration / 2.0
    return bool(len(est_intervals)) and bool(
        ((est_intervals[:, 0] <= mid) & (est_intervals[:, 1] >= mid)).any())
