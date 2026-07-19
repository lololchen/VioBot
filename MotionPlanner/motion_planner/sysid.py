"""SysID scaffolding — the sweep dataset and generator for the acoustics↔actuation
mapper MotionPlanner owns (F8; protocol doc: docs/SysID_Protocol.md).

The `measured` block of a sweep point is EXACTLY the D-008 timbre feature set
(f0, amp, harmonic amplitudes, odd/even, tristimulus, inharmonicity): measuring
a bench recording = running MelodyExtractor's transcriber + timbre on it —
pure reuse, no new DSP. AudioFeedback recordings convert into sweep rows for
free because every take pins its MotionScore hash (CONCEPT_AudioFeedback.md).

The generator emits one tiny MotionScore per grid point with HAND-SET control
tracks (not planner-chosen — a sweep must command exact controls), consumable
by firmware_bridge.streamer/bench unchanged. LearnedBowSoundModel lands here
once real sweeps exist; until then AnalyticSchellengModel is the only
implementation of the BowSoundModel interface (D-027).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .hardware import HardwareProfile, band_index, st_to_mm
from .profile_io import profile_hash, profile_to_dict
from .schema import (
    BowNotePlan,
    Event,
    MotionScore,
    NotePlan,
    ScoreMeta,
    Tracks,
    sha256_of_text,
)

SYSID_VERSION = "sysid-0.1.0"
SWEEP_SCHEMA_VERSION = "1"

_SETTLE_S = 0.25     # engage/settle before the measured window
_TONE_S = 2.0        # steady measured tone (protocol doc)
_TAIL_S = 0.25


@dataclass(frozen=True)
class SweepConfig:
    """Grid definition. Defaults follow docs/SysID_Protocol.md."""

    strings: tuple = (0, 1, 2, 3)
    positions_st: tuple = (0.0, 2.0, 5.0, 9.0, 14.0, 19.0)
    v_b_mps: tuple = (0.1, 0.2, 0.4)
    force_n: tuple = (0.3, 0.8, 1.5)
    beta: tuple = (0.10,)


@dataclass(frozen=True)
class SweepPoint:
    controls: dict                       # string, position_st, v_b_mps, force_n, beta
    measured: "dict | None" = None       # the D-008 feature block, None until measured
    point_id: str = ""


@dataclass(frozen=True)
class SweepDataset:
    points: tuple = ()
    meta: dict = field(default_factory=dict)

    def to_json_dict(self) -> dict:
        return {"sweep_schema_version": SWEEP_SCHEMA_VERSION,
                "meta": dict(self.meta),
                "points": [{"point_id": p.point_id, "controls": dict(p.controls),
                            "measured": None if p.measured is None else dict(p.measured)}
                           for p in self.points]}

    def to_json(self, path: "str | Path | None" = None) -> str:
        text = json.dumps(self.to_json_dict(), sort_keys=True, indent=1, allow_nan=False) + "\n"
        if path is not None:
            Path(path).write_text(text, encoding="utf-8", newline="\n")
        return text

    @classmethod
    def from_json(cls, source: "str | Path") -> "SweepDataset":
        text = Path(source).read_text(encoding="utf-8") if str(source).endswith(".json") \
            else str(source)
        d = json.loads(text)
        version = d.get("sweep_schema_version", "")
        if version.split(".")[0] != SWEEP_SCHEMA_VERSION:
            raise ValueError(f"incompatible sweep_schema_version {version!r}")
        return cls(points=tuple(SweepPoint(controls=dict(p["controls"]),
                                           measured=p.get("measured"),
                                           point_id=p.get("point_id", ""))
                                for p in d.get("points", [])),
                   meta=dict(d.get("meta", {})))


def sweep_points(config: SweepConfig) -> "list[SweepPoint]":
    """Deterministic grid enumeration; point_id encodes the coordinates."""
    points = []
    for s in config.strings:
        for pos in config.positions_st:
            for v in config.v_b_mps:
                for f in config.force_n:
                    for b in config.beta:
                        pid = f"s{s}_p{pos:g}_v{v:g}_f{f:g}_b{b:g}"
                        points.append(SweepPoint(point_id=pid, controls={
                            "string": s, "position_st": pos, "v_b_mps": v,
                            "force_n": f, "beta": b}))
    return points


def point_to_score(point: SweepPoint, profile: HardwareProfile) -> MotionScore:
    """One sweep point → a tiny MotionScore with hand-set constant control
    tracks (settle → 2 s tone → release), streamable like any plan output."""
    c = point.controls
    s = int(c["string"])
    pos = float(c["position_st"])
    hop = profile.timing.control_hop_s
    total = _SETTLE_S + _TONE_S + _TAIL_S
    n = int(total / hop) + 1
    t_on, t_off = _SETTLE_S, _SETTLE_S + _TONE_S

    def gate(v_on: float, v_off: float = 0.0):
        return tuple(v_on if t_on <= k * hop <= t_off else v_off for k in range(n))

    angle = profile.strings.band_angles_rad[band_index((s,))]
    channels = {
        "bow.speed_mps": gate(float(c["v_b_mps"])),
        "bow.force_n": gate(float(c["force_n"])),
        "bow.inclination_rad": tuple(angle for _ in range(n)),
        "bow.y_m": gate(0.0, profile.bow.y.travel_m / 4.0),
        "bow.beta": tuple(float(c["beta"]) for _ in range(n)),
    }
    fingers = profile.candidate_fingers_for_string(s)
    finger = fingers[0] if (pos > 0.01 and fingers) else None
    z_m = st_to_mm(pos, profile.strings.scale_length_mm) / 1000.0
    x_m = s * profile.strings.spacing_bridge_mm / 1000.0
    for fi, unit in enumerate(profile.fingers):
        pressed = fi == finger
        channels[f"f{fi}.z_m"] = tuple(z_m if pressed else 0.0 for _ in range(n))
        channels[f"f{fi}.x_m"] = tuple(x_m if pressed else 0.0 for _ in range(n))
        channels[f"f{fi}.press_n"] = gate(0.8 * unit.press.f_max_n) if pressed \
            else tuple(0.0 for _ in range(n))

    pitch_hz = profile.strings.open_hz[s] * 2.0 ** (pos / 12.0)
    note = NotePlan(
        note_index=0, pitch_hz=pitch_hz, onset_s=t_on, string=s, finger=finger,
        position_st=pos, position_mm=st_to_mm(pos, profile.strings.scale_length_mm),
        bow=BowNotePlan(segment_id=0, v_b_mps=float(c["v_b_mps"]),
                        force_n=float(c["force_n"]), beta=float(c["beta"]),
                        u_brightness=0.5),
        realized_onset_s=t_on, realized_duration_s=_TONE_S)
    meta = ScoreMeta(
        source_note_sequence={"path_hint": f"sysid:{point.point_id}",
                              "sha256": sha256_of_text(point.point_id)},
        hardware_profile={"snapshot": profile_to_dict(profile),
                          "sha256": profile_hash(profile)},
        backends={"sysid": SYSID_VERSION},
        extra={"sweep_point_id": point.point_id})
    events = (Event(t_s=t_on, kind="bow_land", params={"segment_id": 0, "y_to_m": 0.0}),
              Event(t_s=t_off, kind="bow_lift",
                    params={"segment_id": 0, "y_to_m": profile.bow.y.travel_m / 4.0}))
    return MotionScore(meta=meta, note_plan=(note,), events=events,
                       tracks=Tracks(hop_s=hop, start_s=0.0, channels=channels)).validate()


def measured_from_note(note) -> dict:
    """melody_extractor Note (from extracting a bench recording) → the measured
    block. Field-for-field the D-008 feature set."""
    d = {"f0_hz": note.pitch_hz,
         "amp_db": note.amp_db_envelope.peak_db()}
    if note.harmonics is not None:
        d.update({
            "harmonic_amps_db": list(note.harmonics.harmonic_amps_db),
            "odd_even_ratio": note.harmonics.odd_even_ratio,
            "tristimulus": list(note.harmonics.tristimulus),
            "inharmonicity": note.harmonics.inharmonicity,
        })
    return d


class LearnedBowSoundModel:
    """Placeholder for the mapper fit on real SweepDatasets (D-027).

    Intended shape: deterministic ridge regression on log-features
    (log v_b, log F, log β, position_st, string one-hot) → (amp_db, brightness,
    per-harmonic dbs), fixed solver, dataset hash pinned into `version`.
    """

    def __init__(self, dataset: SweepDataset):
        raise NotImplementedError(
            "LearnedBowSoundModel needs measured sweep data; run the bench sweep "
            "per docs/SysID_Protocol.md first (AnalyticSchellengModel is v0)")
