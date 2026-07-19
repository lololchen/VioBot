"""BowSoundModel — the acoustics↔actuation mapper MotionPlanner owns (D-027).

v0 is the ANALYTIC prior; the learned mapper trained on SysID sweeps replaces
it behind the same interface (sysid.py). All constants are literature-shaped
guesses calibrated for hardware that does not exist yet, so conclusions drawn
through this model are RELATIVE (same model on both sides of a comparison),
never absolute sound quality (PRD out-of-scope).

Physics (Schelleng 1973, folded constants live in profile.bow.schelleng):
    playable wedge   F_min = k_min · Z² · v_b / β²   ≤ F ≤   F_max = k_max · Z · v_b / β
    loudness prior   amp_db ≈ a0 + 20·log10(v_b / β)
    force placement  F = F_min^(1−u) · F_max^u  — log-space interpolation with
                     brightness u ∈ [u_min, u_max]; u = 0.5 is the geometric
                     mean, guaranteed strictly inside the wedge.
Brightness u also shapes the predicted per-harmonic rolloff (sul-ponticello
bright ↔ sul-tasto dark): drop_db_per_harmonic = 14 − 10·u.

Attack: a constant-speed belt never accelerates from rest, so the classic
Guettler (acceleration × force) diagram does not apply; onsets are shaped by
force rise time at Y-engage (profile.bow.guettler.t_attack_min_s) and the
tuple (v_b, F, dF/dt, β) is logged per onset as SysID priority #1 (PRD).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .hardware import HardwareProfile

BOW_SOUND_MODEL_VERSION = "schelleng-analytic-0.1.0"


@dataclass(frozen=True)
class BowControls:
    v_b_mps: float
    force_n: float
    beta: float
    string: int
    position_st: float = 0.0


@dataclass(frozen=True)
class AcousticPoint:
    """What a (controls) point sounds like under the model. sounding=False
    means outside the Schelleng wedge (raucous/surface noise approximated as
    silence in v0 — the sim flags it)."""

    amp_db: float
    brightness_u: float
    sounding: bool
    f_min_n: float
    f_max_n: float


@dataclass(frozen=True)
class InverseResult:
    v_b_mps: float
    force_n: float
    speed_clipped: bool
    force_clipped: bool
    f_min_n: float
    f_max_n: float


class AnalyticSchellengModel:
    """Deterministic, closed-form. Interface shared with the future learned
    model: forward(controls) -> AcousticPoint; inverse(target) -> InverseResult."""

    version = BOW_SOUND_MODEL_VERSION

    def __init__(self, profile: HardwareProfile):
        self.profile = profile

    # -- wedge --

    def wedge(self, v_b: float, beta: float, string: int) -> "tuple[float, float]":
        sch = self.profile.bow.schelleng
        z = self.profile.strings.impedance_z[string]
        f_max = sch.k_max * z * v_b / beta
        f_min = sch.k_min * z * z * v_b / (beta * beta)
        return f_min, f_max

    # -- forward: controls -> sound --

    def forward(self, c: BowControls) -> AcousticPoint:
        sch = self.profile.bow.schelleng
        f_min, f_max = self.wedge(c.v_b_mps, c.beta, c.string)
        if c.v_b_mps < sch.v_b_min_mps or not (f_min <= c.force_n <= f_max):
            return AcousticPoint(amp_db=-120.0, brightness_u=0.5, sounding=False,
                                 f_min_n=f_min, f_max_n=f_max)
        amp_db = sch.a0_db + 20.0 * math.log10(c.v_b_mps / c.beta)
        # Invert the log-space force placement to recover the brightness the
        # force actually realizes (may differ from the requested u after clips).
        if f_max > f_min > 0.0:
            u = math.log(c.force_n / f_min) / math.log(f_max / f_min)
        else:
            u = 0.5
        return AcousticPoint(amp_db=amp_db, brightness_u=min(max(u, 0.0), 1.0),
                             sounding=True, f_min_n=f_min, f_max_n=f_max)

    def harmonic_rolloff_db(self, u: float) -> float:
        """Per-harmonic amplitude drop (dB) implied by brightness u."""
        return 14.0 - 10.0 * min(max(u, 0.0), 1.0)

    # -- inverse: target sound -> controls --

    def inverse(self, target_amp_db: float, u: float, beta: float,
                string: int) -> InverseResult:
        sch = self.profile.bow.schelleng
        v_b = beta * 10.0 ** ((target_amp_db - sch.a0_db) / 20.0)
        speed_clipped = False
        if v_b < sch.v_b_min_mps:
            v_b, speed_clipped = sch.v_b_min_mps, True
        v_cap = self.profile.bow.belt.v_max_mps
        if v_b > v_cap:
            v_b, speed_clipped = v_cap, True
        f_min, f_max = self.wedge(v_b, beta, string)
        force = f_min ** (1.0 - u) * f_max ** u
        force_clipped = False
        f_cap = self.profile.bow.force.f_max_n
        if force > f_cap:
            force, force_clipped = f_cap, True
            force_clipped = force_clipped or not (f_min <= force <= f_max)
        return InverseResult(v_b_mps=v_b, force_n=force, speed_clipped=speed_clipped,
                             force_clipped=force_clipped, f_min_n=f_min, f_max_n=f_max)
