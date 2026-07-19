"""HardwareProfile — the parametric hardware model MotionPlanner plans against.

No mechanics exist yet (PRD): every hardware-specific number lives HERE (or in a
profile JSON, see profile_io), never in planner code. v0 limits are task-space
velocity/acceleration/bandwidth envelopes; joint-space kinematics (four-bar,
capstan, lead screw, bow differential) is declared v1 (D-025).

Unit discipline (module CLAUDE.md): fields suffixed `_mm` are millimetres
(human-facing fingerboard geometry), `_m` metres (axis ranges — same unit the
MotionScore tracks use), `_rad`, `_s`, `_n` (newtons), `_mps`/`_mps2`,
`_radps`/`_radps2`. Convert only at the suffix boundary.

Geometry: a stopped position of `s` semitones above the open string sits
    x = L·(1 − 2^(−s/12))            millimetres from the nut,
with local scale  dx/ds = (L·ln2/12)·2^(−s/12)  mm per semitone, and leaves a
sounding length  L_s = L·2^(−s/12).  β (Schelleng contact point) is
bow–bridge distance divided by the *sounding* length, so β_eff rises with
position when the bow placement is mechanically fixed (PRD β section).

Inclination bands: 7 bands for 4 strings — index 2·s is string s alone,
index 2·s+1 is the (s, s+1) double-stop band. Triples are executed as rolled
pair-then-pair sweeps between adjacent double-stop bands (D-024).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from melody_extractor.reducer import OPEN_STRINGS_HZ

PROFILE_SCHEMA_VERSION = "1"

_LN2 = math.log(2.0)


# ---------- geometry helpers (pure, used by every planner) ----------

def st_to_mm(position_st: float, scale_length_mm: float) -> float:
    """Distance from the nut (mm) for a stopped position in semitones."""
    return scale_length_mm * (1.0 - 2.0 ** (-position_st / 12.0))


def mm_per_st(position_st: float, scale_length_mm: float) -> float:
    """Local fingerboard scale (mm per semitone) at a stopped position."""
    return (scale_length_mm * _LN2 / 12.0) * 2.0 ** (-position_st / 12.0)


def mm_to_st(x_mm: float, scale_length_mm: float) -> float:
    """Inverse of st_to_mm. x_mm must be < scale_length_mm."""
    return -12.0 * math.log2(1.0 - x_mm / scale_length_mm)


def sounding_length_mm(position_st: float, scale_length_mm: float) -> float:
    """Vibrating string length (mm) when stopped at position_st semitones."""
    return scale_length_mm * 2.0 ** (-position_st / 12.0)


def hz_to_position_st(pitch_hz: float, open_hz: float) -> float:
    """Stopped position in semitones above an open string (negative = below)."""
    return 12.0 * math.log2(pitch_hz / open_hz)


def trapezoid_time(distance: float, v_max: float, a_max: float) -> float:
    """Minimum time for a rest-to-rest point-to-point move under a trapezoidal
    (triangular when short) velocity profile. distance/v_max/a_max share any
    consistent unit system; returns seconds. Zero distance -> 0.0."""
    d = abs(distance)
    if d <= 0.0:
        return 0.0
    if v_max <= 0.0 or a_max <= 0.0:
        return math.inf
    d_tri = v_max * v_max / a_max  # distance below which v_max is never reached
    if d <= d_tri:
        return 2.0 * math.sqrt(d / a_max)
    return d / v_max + v_max / a_max


def band_index(strings: "tuple[int, ...]") -> int:
    """Inclination band for a set of simultaneously bowed strings.

    (s,) -> 2s ; (s, s+1) -> 2s+1. Triples never get a band — they are rolled
    across two adjacent double-stop bands (D-024), the bow planner handles it.
    """
    ss = tuple(sorted(strings))
    if len(ss) == 1:
        return 2 * ss[0]
    if len(ss) == 2 and ss[1] == ss[0] + 1:
        return 2 * ss[0] + 1
    raise ValueError(f"no single inclination band for strings {ss}")


# ---------- profile dataclasses ----------

@dataclass(frozen=True)
class Topology:
    """Finger-unit arrangement under comparison (PRD Concept A vs B)."""

    concept: str = "A"                       # "A" roaming fingers | "B" one per string
    n_fingers: int = 1
    finger_home_string: tuple = (None,)      # per finger: fixed string (B) or None (A roams)
    fingers_can_cross: bool = False          # mech TBD — conservative default (PRD open Q1)
    min_finger_separation_mm: float = 25.0


@dataclass(frozen=True)
class StringsConfig:
    open_hz: tuple = OPEN_STRINGS_HZ         # MUST stay == reducer's constants (D-015/D-025)
    scale_length_mm: float = 325.0
    spacing_bridge_mm: float = 11.5
    spacing_nut_mm: float = 5.5
    impedance_z: tuple = (0.55, 0.40, 0.30, 0.25)   # kg/s, literature-order priors; SysID later
    band_angles_rad: tuple = (-0.30, -0.20, -0.10, 0.0, 0.10, 0.20, 0.30)
    band_halfwidth_rad: float = 0.03


@dataclass(frozen=True)
class Fingerboard:
    max_position_st: float = 19.0            # mirrors reducer.StageConfig default


@dataclass(frozen=True)
class BeltAxis:
    """Continuously spinning bow-hair belt: surface speed, single direction (MVP)."""

    v_max_mps: float = 1.0
    a_max_mps2: float = 4.0


@dataclass(frozen=True)
class LinearAxis:
    travel_m: float = 0.02
    v_max_mps: float = 0.2
    a_max_mps2: float = 4.0


@dataclass(frozen=True)
class RotaryAxis:
    range_rad: float = 0.70                  # full inclination sweep G-side to E-side
    v_max_radps: float = 6.0
    a_max_radps2: float = 80.0


@dataclass(frozen=True)
class ForceAxis:
    """Bow pressure after contact (differential common mode). Physical newtons —
    the N·m / FOC-amp conversion is a firmware_bridge concern (D-023)."""

    f_max_n: float = 2.5
    df_dt_max_nps: float = 50.0


@dataclass(frozen=True)
class SchellengConstants:
    """Calibration of the analytic playable-region wedge (D-027):
    F_max = k_max·Z·v_b/β ; F_min = k_min·Z²·v_b/β² ; amp_db ≈ a0 + 20·log10(v_b/β).
    Literature-shaped priors for hardware that does not exist yet — sim results
    are RELATIVE comparisons until SysID measures the real constants."""

    k_max: float = 2.0
    k_min: float = 0.06
    # a0 calibrated so a -20 dBFS note at beta=0.10 asks ~0.25 m/s of belt:
    # a0 = amp_db - 20·log10(v_b/β) = -20 - 20·log10(2.5) ≈ -28.
    a0_db: float = -28.0
    v_b_min_mps: float = 0.02


@dataclass(frozen=True)
class GuettlerConstants:
    """Attack shaping. The classic Guettler diagram (acceleration × force) does
    not apply to a constant-speed belt engaging via Y — v0 shapes the force
    rise time instead and logs the attack tuple for SysID (PRD caveat)."""

    t_attack_min_s: float = 0.015


@dataclass(frozen=True)
class BowConfig:
    belt: BeltAxis = field(default_factory=BeltAxis)
    y: LinearAxis = field(default_factory=LinearAxis)
    incl: RotaryAxis = field(default_factory=RotaryAxis)
    force: ForceAxis = field(default_factory=ForceAxis)
    r_contact_m: float = 0.03                # contact-point radius from the Z axis (ω coupling)
    beta_actuated: bool = False
    beta_default: float = 0.10
    beta_range: tuple = (0.05, 0.25)
    schelleng: SchellengConstants = field(default_factory=SchellengConstants)
    guettler: GuettlerConstants = field(default_factory=GuettlerConstants)
    tilt: "dict | None" = None               # reserved (PRD open Q4); unused v0


@dataclass(frozen=True)
class PressAxis:
    f_max_n: float = 4.0
    t_press_s: float = 0.03
    t_lift_s: float = 0.025
    bandwidth_hz: float = 15.0


@dataclass(frozen=True)
class FingerAxis:
    range_m: float = 0.05
    v_max_mps: float = 0.5
    a_max_mps2: float = 20.0
    bandwidth_hz: float = 12.0


@dataclass(frozen=True)
class FingerUnit:
    """One 3-DoF finger: capstan/four-bar XY (string select + press) + lead-screw Z
    (fingerboard traverse + vibrato). Motor block is informational in v0."""

    x: FingerAxis = field(default_factory=lambda: FingerAxis(range_m=0.05, v_max_mps=0.5, a_max_mps2=20.0))
    press: PressAxis = field(default_factory=PressAxis)
    z: FingerAxis = field(default_factory=lambda: FingerAxis(range_m=0.28, v_max_mps=0.6, a_max_mps2=25.0))
    motor: dict = field(default_factory=lambda: {
        "type": "1503", "kv": 2400, "driver": "DRV8313", "encoder": "AS5048A-14"})
    device_ids: dict = field(default_factory=lambda: {"x": 0, "press": 0, "z": 0})


@dataclass(frozen=True)
class Timing:
    control_hop_s: float = 0.01
    command_latency_s: float = 0.02


@dataclass(frozen=True)
class HardwareProfile:
    name: str = "unnamed"
    comment: str = ""
    topology: Topology = field(default_factory=Topology)
    strings: StringsConfig = field(default_factory=StringsConfig)
    fingerboard: Fingerboard = field(default_factory=Fingerboard)
    bow: BowConfig = field(default_factory=BowConfig)
    fingers: tuple = (FingerUnit(),)
    timing: Timing = field(default_factory=Timing)

    def validate(self) -> "HardwareProfile":
        if self.topology.concept not in ("A", "B"):
            raise ValueError(f"topology.concept must be 'A' or 'B', got {self.topology.concept!r}")
        if len(self.fingers) != self.topology.n_fingers:
            raise ValueError(
                f"topology.n_fingers={self.topology.n_fingers} but {len(self.fingers)} finger units defined")
        if len(self.topology.finger_home_string) != self.topology.n_fingers:
            raise ValueError("finger_home_string length must equal n_fingers")
        if self.topology.concept == "B" and any(h is None for h in self.topology.finger_home_string):
            raise ValueError("concept B requires a home string per finger")
        if tuple(self.strings.open_hz) != tuple(OPEN_STRINGS_HZ):
            raise ValueError(
                "strings.open_hz must equal reducer.OPEN_STRINGS_HZ (D-015/D-025); "
                "changing tuning is a decisions.md-level schema event")
        if len(self.strings.band_angles_rad) != 2 * len(self.strings.open_hz) - 1:
            raise ValueError("band_angles_rad must have 2*n_strings-1 entries")
        angles = self.strings.band_angles_rad
        if any(b <= a for a, b in zip(angles, angles[1:])):
            raise ValueError("band_angles_rad must be strictly increasing")
        if not (0.0 < self.bow.beta_range[0] <= self.bow.beta_default <= self.bow.beta_range[1] < 1.0):
            raise ValueError("beta_default must lie inside beta_range within (0, 1)")
        if self.timing.control_hop_s <= 0:
            raise ValueError("control_hop_s must be > 0")
        return self

    # -- convenience used across planners --

    def candidate_fingers_for_string(self, string: int) -> "tuple[int, ...]":
        """Finger-unit ids allowed to stop notes on `string` (topology dispatch)."""
        if self.topology.concept == "B":
            return tuple(i for i, h in enumerate(self.topology.finger_home_string) if h == string)
        return tuple(range(self.topology.n_fingers))

    def position_mm(self, position_st: float) -> float:
        return st_to_mm(position_st, self.strings.scale_length_mm)

    def beta_eff(self, position_st: float) -> float:
        """Contact point for the (fixed, v0) bow placement at a stopped position.
        beta_default is defined at the open string; sounding length shrinks with
        position, so beta_eff = beta_default · 2^(p/12), clamped to beta_range
        only when the axis is actuated (a fixed bow cannot clamp physics)."""
        beta = self.bow.beta_default * 2.0 ** (position_st / 12.0)
        if self.bow.beta_actuated:
            return min(max(beta, self.bow.beta_range[0]), self.bow.beta_range[1])
        return beta
