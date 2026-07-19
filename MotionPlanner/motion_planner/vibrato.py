"""Vibrato planner — f0_contour → (rate, depth, delay) → finger-Z oscillation (F3).

Extraction (pure numpy, deterministic): the note's f0 contour is converted to
cents deviation from its median, linearly resampled onto a uniform grid,
detrended, and autocorrelated; the strongest lag inside
[1/f_hi, 1/f_lo] gives the rate. Depth is the sinusoid-equivalent amplitude
√2·RMS of the detrended signal. Gates (PlannerConfig): depth ≥
vibrato_min_depth_cents, ≥ vibrato_min_cycles within the note, and an
autocorrelation peak ≥ 0.25 (rejects drift/noise contours).

Mapping to hardware: Δz_m = (depth_cents/100) · mm_per_st(position, L) / 1000.
Peak demands v_pk = 2πf·Δz, a_pk = (2πf)²·Δz. Depth (NEVER rate — pitch wobble
frequency is musically salient, its width is not) is clipped so that
a_pk ≤ z.a_max and, when the rate exceeds the axis bandwidth, scaled by
(bandwidth/rate)² (second-order rolloff). Clips set depth_cents_clipped and a
vibrato_clipped violation (D-026 soft-fail rule).

Open strings get no vibrato (no finger on the string).
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from melody_extractor.schema import Note

from .config_io import PlannerConfig
from .hardware import HardwareProfile, mm_per_st
from .schema import VibratoPlan, Violation

VIBRATO_VERSION = "acf-vibrato-0.1.0"

_RESAMPLE_HOP_S = 0.005
_MIN_ACF_PEAK = 0.25


def extract_vibrato(note: Note, config: PlannerConfig) -> Optional[VibratoPlan]:
    """Detect vibrato in a note's f0 contour. None when absent/too weak."""
    contour = note.f0_contour
    if contour is None or len(contour.times_s) < 8:
        return None
    t = np.asarray(contour.times_s, dtype=np.float64)
    f0 = np.asarray(contour.f0_hz, dtype=np.float64)
    span = float(t[-1] - t[0])
    if span < 2.0 * _RESAMPLE_HOP_S:
        return None
    median = float(np.median(f0))
    if median <= 0.0:
        return None
    cents = 1200.0 * np.log2(f0 / median)

    n = max(int(span / _RESAMPLE_HOP_S) + 1, 8)
    tu = t[0] + np.arange(n) * _RESAMPLE_HOP_S
    cu = np.interp(tu, t, cents)
    # Linear detrend (least squares against [1, t]).
    a = np.vstack([np.ones(n), np.arange(n, dtype=np.float64)]).T
    coef, *_ = np.linalg.lstsq(a, cu, rcond=None)
    cu = cu - a @ coef

    rms = float(np.sqrt(np.mean(cu * cu)))
    depth = math.sqrt(2.0) * rms
    if depth < config.vibrato_min_depth_cents:
        return None

    # Autocorrelation, normalized at lag 0.
    acf = np.correlate(cu, cu, mode="full")[n - 1:]
    if acf[0] <= 0.0:
        return None
    acf = acf / acf[0]
    lag_lo = max(int(round(1.0 / (config.vibrato_f_hi_hz * _RESAMPLE_HOP_S))), 2)
    lag_hi = min(int(round(1.0 / (config.vibrato_f_lo_hz * _RESAMPLE_HOP_S))), n - 2)
    if lag_hi <= lag_lo:
        return None
    window = acf[lag_lo:lag_hi + 1]
    best = int(np.argmax(window)) + lag_lo
    if float(acf[best]) < _MIN_ACF_PEAK:
        return None
    rate = 1.0 / (best * _RESAMPLE_HOP_S)
    if span * rate < config.vibrato_min_cycles:
        return None

    # Delay: first time the |deviation| sustains above half the depth.
    above = np.abs(cu) >= 0.5 * depth
    delay = float(tu[int(np.argmax(above))] - t[0]) if bool(above.any()) else 0.0
    return VibratoPlan(rate_hz=rate, depth_cents=depth, delay_s=delay)


def clip_to_axis(vib: VibratoPlan, position_st: float, finger: Optional[int],
                 profile: HardwareProfile) -> "tuple[Optional[VibratoPlan], tuple[Violation, ...]]":
    """Clip depth to the finger-Z axis physics. Open strings → (None, ())."""
    if finger is None:
        return None, ()
    z = profile.fingers[finger].z
    scale = profile.strings.scale_length_mm
    omega = 2.0 * math.pi * vib.rate_hz
    dz_m = (vib.depth_cents / 100.0) * mm_per_st(position_st, scale) / 1000.0
    depth = vib.depth_cents
    # Acceleration bound: a_pk = ω²·Δz ≤ a_max.
    a_pk = omega * omega * dz_m
    if a_pk > z.a_max_mps2:
        depth *= z.a_max_mps2 / a_pk
    # Bandwidth: second-order amplitude rolloff beyond the axis bandwidth.
    if vib.rate_hz > z.bandwidth_hz > 0.0:
        depth *= (z.bandwidth_hz / vib.rate_hz) ** 2
    if depth < vib.depth_cents - 1e-9:
        clipped = VibratoPlan(rate_hz=vib.rate_hz, depth_cents=depth,
                              delay_s=vib.delay_s, depth_cents_clipped=True)
        violation = Violation(kind="vibrato_clipped", axis=f"f{finger}.z",
                              needed=vib.depth_cents, available=depth)
        return clipped, (violation,)
    return vib, ()


def plan_vibrato(note: Note, position_st: float, finger: Optional[int],
                 profile: HardwareProfile, config: PlannerConfig,
                 ) -> "tuple[Optional[VibratoPlan], tuple[Violation, ...]]":
    """extract + clip in one call (what planner.py uses)."""
    vib = extract_vibrato(note, config)
    if vib is None:
        return None, ()
    return clip_to_axis(vib, position_st, finger, profile)
