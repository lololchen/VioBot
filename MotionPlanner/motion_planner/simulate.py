"""Forward simulation — MotionScore.tracks → predicted NoteSequence (F5).

The predicted sequence is derived from the SIGNALS, not from the plan's
intent: a kinematic gate decides per hop which strings sound and at what
pitch, and the analytic BowSoundModel decides how loud. Timing and gating
bugs (late fingers, slow inclination ramps, force outside the Schelleng
wedge) therefore become measurable onset/pitch errors instead of staying
invisible — that is the entire point (D-028).

Kinematic gate per hop and string s:
    sounds(s)  ⟺  |inclination − band_angle| ≤ band_halfwidth for a band
                  containing s  ∧  bow y in contact  ∧  belt speed ≥ v_b_min
    pitch(s)   =  highest-position finger pressed on s (press_n ≥ threshold,
                  x lane == s), else the open string. A finger that moves
                  while pressed slides the pitch — audible planning error.
    loudness   =  BowSoundModel.forward(v_b, F, β, s, position); outside the
                  wedge → silence (v0 approximation of raucous/surface sound).

Segmentation of the per-string frame arrays into Notes mirrors the
transcriber's conventions: voiced runs split at pitch jumps > 0.8 st,
sub-20 ms fragments dropped, median pitch per note, envelope + f0 contour
carried from the frames. meta.backends["simulator"] = "motion-sim-0.1.0".
"""
from __future__ import annotations

import math

from melody_extractor.schema import (
    AmpEnvelope,
    F0Contour,
    FrameTrack,
    Meta,
    NoteSequence,
    Note,
)

from .bow_sound_model import AnalyticSchellengModel, BowControls
from .hardware import mm_to_st
from .profile_io import profile_from_dict
from .schema import MotionScore

SIMULATOR_VERSION = "motion-sim-0.1.0"

_CONTACT_Y_M = 5e-4
_SPLIT_ST = 0.8
_MIN_NOTE_S = 0.02
_SILENT_DB = -120.0


def simulate(score: MotionScore, press_threshold_n: float = 1.0) -> NoteSequence:
    """Deterministic. Uses the profile snapshot pinned in score.meta."""
    profile = profile_from_dict(score.meta.hardware_profile["snapshot"])
    model = AnalyticSchellengModel(profile)
    tracks = score.tracks
    if tracks is None or tracks.n_samples() == 0:
        return NoteSequence(meta=_meta(score))
    n = tracks.n_samples()
    hop = tracks.hop_s
    ch = tracks.channels
    n_strings = len(profile.strings.open_hz)
    angles = profile.strings.band_angles_rad
    halfwidth = profile.strings.band_halfwidth_rad
    spacing_m = profile.strings.spacing_bridge_mm / 1000.0
    scale = profile.strings.scale_length_mm

    f0 = [[0.0] * n for _ in range(n_strings)]
    amp = [[_SILENT_DB] * n for _ in range(n_strings)]

    incl = ch["bow.inclination_rad"]
    y = ch["bow.y_m"]
    speed = ch["bow.speed_mps"]
    force = ch["bow.force_n"]
    beta = ch["bow.beta"]
    fingers = [(ch.get(f"f{i}.x_m"), ch.get(f"f{i}.z_m"), ch.get(f"f{i}.press_n"))
               for i in range(len(profile.fingers))]

    for k in range(n):
        if y[k] > _CONTACT_Y_M or speed[k] < profile.bow.schelleng.v_b_min_mps:
            continue
        sounding = _strings_in_band(incl[k], angles, halfwidth, n_strings)
        for s in sounding:
            position_st = 0.0
            for fx, fz, fp in fingers:
                if fx is None or fp is None or fp[k] < press_threshold_n:
                    continue
                lane = int(round(fx[k] / spacing_m)) if spacing_m > 0 else 0
                if lane != s:
                    continue
                z_mm = max(fz[k], 0.0) * 1000.0
                if z_mm < scale:
                    position_st = max(position_st, mm_to_st(z_mm, scale))
            point = model.forward(BowControls(
                v_b_mps=speed[k], force_n=force[k], beta=beta[k],
                string=s, position_st=position_st))
            if not point.sounding:
                continue
            f0[s][k] = profile.strings.open_hz[s] * 2.0 ** (position_st / 12.0)
            amp[s][k] = point.amp_db

    notes = []
    features = []
    for s in range(n_strings):
        notes.extend(_segment(f0[s], amp[s], hop))
        if any(v > 0.0 for v in f0[s]):
            features.append(FrameTrack(
                hop_s=hop, start_s=tracks.start_s, name=f"sim_string{s}",
                f0_hz=tuple(f0[s]),
                voicing=tuple(1.0 if v > 0.0 else 0.0 for v in f0[s]),
                amp_db=tuple(amp[s])))
    return NoteSequence(notes=tuple(notes), features=tuple(features),
                        meta=_meta(score)).validate()


def _meta(score: MotionScore) -> Meta:
    backends = dict(score.meta.backends)
    backends["simulator"] = SIMULATOR_VERSION
    return Meta(source=score.meta.source_note_sequence.get("path_hint", ""),
                source_kind="synthetic", sample_rate=None, backends=backends,
                stage=None, extra={"motion_score_sha256":
                                   score.meta.source_note_sequence.get("sha256", "")})


def _strings_in_band(angle: float, angles, halfwidth: float,
                     n_strings: int) -> "tuple[int, ...]":
    for band, center in enumerate(angles):
        if abs(angle - center) <= halfwidth:
            if band % 2 == 0:
                return (band // 2,)
            return (band // 2, band // 2 + 1)
    return ()


def _segment(f0_row, amp_row, hop: float):
    """Voiced runs → Notes; split at >0.8 st jumps; drop sub-20 ms fragments."""
    notes = []
    k = 0
    n = len(f0_row)
    while k < n:
        if f0_row[k] <= 0.0:
            k += 1
            continue
        start = k
        prev_hz = f0_row[k]
        k += 1
        while k < n and f0_row[k] > 0.0 and \
                abs(1200.0 * math.log2(f0_row[k] / prev_hz)) <= _SPLIT_ST * 100.0:
            prev_hz = f0_row[k]
            k += 1
        end = k  # exclusive
        dur = (end - start) * hop
        if dur < _MIN_NOTE_S:
            continue
        seg_f0 = f0_row[start:end]
        seg_amp = amp_row[start:end]
        median_hz = sorted(seg_f0)[len(seg_f0) // 2]
        rel_t = tuple(i * hop for i in range(end - start))
        notes.append(Note(
            pitch_hz=median_hz,
            onset_s=start * hop,
            duration_s=dur,
            amp_db_envelope=AmpEnvelope(times_s=rel_t, amp_db=tuple(seg_amp)),
            f0_contour=F0Contour(times_s=rel_t, f0_hz=tuple(seg_f0)),
        ))
    return notes
