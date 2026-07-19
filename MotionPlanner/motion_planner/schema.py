"""MotionScore — the versioned output contract of MotionPlanner (D-023).

Consumed by the forward sim, the GUI, the firmware bridge and (future)
AudioFeedback. Governance mirrors melody_extractor.schema.NoteSequence exactly:
frozen dataclasses, tuple coercion, deterministic JSON (sorted keys, indent=1,
allow_nan=False, trailing LF), major-version gate on load, and any layout
change requires a schema-version bump + docs/decisions.md entry.

Three layers (why: D-023):
- note_plan[] — per-note assignment + realized timing + violations. Notes are
  referenced by `note_index` into the source `NoteSequence.sorted().notes`;
  the source is pinned by sha256 in meta, pitch/onset are echoed read-only.
- events[]   — typed discrete actions (bow_contact / bow_incline / bow_lift /
  bow_land / roll / finger_press / finger_lift / finger_move). What the
  firmware bridge and the GUI consume; the future trajectory-streaming
  firmware command maps onto this layer.
- tracks     — fixed-hop task-space samples (what the sim consumes). Channel
  names: bow.speed_mps, bow.force_n, bow.inclination_rad, bow.y_m, bow.beta,
  and per finger unit i: f{i}.x_m, f{i}.press_n, f{i}.z_m. Physical SI units
  only — joint space never enters this schema.

The FeasibilityReport is a SEPARATE artifact (same governance): the topology
comparison consumes it without parsing full scores.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field, replace  # noqa: F401  (replace re-exported for callers)
from pathlib import Path
from typing import Any, Optional

MOTION_SCHEMA_VERSION = "0.1.0"
REPORT_SCHEMA_VERSION = "0.1.0"

EVENT_KINDS = (
    "bow_contact", "bow_incline", "bow_lift", "bow_land", "roll",
    "finger_press", "finger_lift", "finger_move",
)

VIOLATION_KINDS = (
    "late_transition", "vibrato_clipped", "force_out_of_wedge", "speed_clipped",
    "position_out_of_range", "coupling_wobble", "infeasible_assignment",
)


def sha256_of_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _tuple_f(values) -> tuple:
    return tuple(float(v) for v in values)


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def _finite(x: float, what: str) -> None:
    _require(math.isfinite(x), f"{what} is NaN/Inf")


@dataclass(frozen=True)
class BowNotePlan:
    """Steady-state bow parameters chosen for one note (attack shaping is an
    event/track concern). beta is the EFFECTIVE contact point at this note's
    stopped position (hardware.HardwareProfile.beta_eff)."""

    segment_id: int
    v_b_mps: float
    force_n: float
    beta: float
    u_brightness: float

    def validate(self) -> None:
        _require(self.segment_id >= 0, "segment_id must be >= 0")
        for name in ("v_b_mps", "force_n", "beta", "u_brightness"):
            _finite(getattr(self, name), f"bow.{name}")
        _require(self.v_b_mps >= 0.0, "v_b_mps must be >= 0 (belt has one direction)")
        _require(0.0 < self.beta < 1.0, f"beta out of (0,1): {self.beta}")


@dataclass(frozen=True)
class VibratoPlan:
    rate_hz: float
    depth_cents: float
    delay_s: float
    depth_cents_clipped: bool = False

    def validate(self) -> None:
        for name in ("rate_hz", "depth_cents", "delay_s"):
            _finite(getattr(self, name), f"vibrato.{name}")
        _require(self.rate_hz > 0 and self.depth_cents >= 0 and self.delay_s >= 0, "vibrato params invalid")


@dataclass(frozen=True)
class Violation:
    """One recorded feasibility breach. Planners never hard-fail (D-026):
    breaches become finite costs + one of these."""

    kind: str
    axis: str = ""
    needed: float = 0.0
    available: float = 0.0
    late_by_s: float = 0.0

    def validate(self) -> None:
        _require(self.kind in VIOLATION_KINDS, f"unknown violation kind {self.kind!r}")
        for name in ("needed", "available", "late_by_s"):
            _finite(getattr(self, name), f"violation.{name}")


@dataclass(frozen=True)
class NotePlan:
    note_index: int
    pitch_hz: float                  # read-only echo of the source note (human debugging)
    onset_s: float                   # score onset (echo); realized_onset_s may differ (D-024)
    string: int
    finger: Optional[int]            # None = open string
    position_st: float
    position_mm: float
    bow: BowNotePlan
    realized_onset_s: float
    realized_duration_s: float
    vibrato: Optional[VibratoPlan] = None
    violations: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, "violations", tuple(self.violations))

    def validate(self) -> None:
        _require(self.note_index >= 0, "note_index must be >= 0")
        _require(0 <= self.string <= 3, f"string out of range: {self.string}")
        _require(self.position_st >= 0.0, f"position_st negative: {self.position_st}")
        if self.finger is None:
            _require(self.position_st <= 0.01, "open string must have position_st ~ 0")
        _finite(self.realized_onset_s, "realized_onset_s")
        _require(self.realized_duration_s > 0, "realized_duration_s must be > 0")
        self.bow.validate()
        if self.vibrato is not None:
            self.vibrato.validate()
        for v in self.violations:
            v.validate()


@dataclass(frozen=True)
class Event:
    """Typed discrete action. `params` is a flat JSON-safe dict; the set of keys
    per kind is documented in bowing/fingering/trajectory docstrings. One
    generic class (not eight) keeps serialization canonical and the schema
    stable while event vocabularies evolve behind kind strings."""

    t_s: float
    kind: str
    params: dict = field(default_factory=dict)

    def validate(self) -> None:
        _require(self.kind in EVENT_KINDS, f"unknown event kind {self.kind!r}")
        _finite(self.t_s, "event t_s")
        _require(self.t_s >= 0.0, "event t_s must be >= 0")

    def sort_key(self) -> tuple:
        return (self.t_s, self.kind, json.dumps(self.params, sort_keys=True, allow_nan=False))


@dataclass(frozen=True)
class Tracks:
    """Fixed-hop sampled task-space channels. All channels share hop/start and
    equal length (the trajectory stage guarantees it)."""

    hop_s: float
    start_s: float
    channels: dict = field(default_factory=dict)   # name -> tuple[float, ...]

    def __post_init__(self):
        object.__setattr__(self, "channels", {k: _tuple_f(v) for k, v in sorted(self.channels.items())})

    def validate(self) -> None:
        _require(self.hop_s > 0, "tracks hop_s must be > 0")
        lengths = {len(v) for v in self.channels.values()}
        _require(len(lengths) <= 1, f"track channels differ in length: {sorted(lengths)}")
        for name, values in self.channels.items():
            _require(all(math.isfinite(v) for v in values), f"track {name} contains NaN/Inf")

    def n_samples(self) -> int:
        return 0 if not self.channels else len(next(iter(self.channels.values())))

    def times_s(self) -> tuple:
        return tuple(self.start_s + i * self.hop_s for i in range(self.n_samples()))


@dataclass(frozen=True)
class ScoreMeta:
    source_note_sequence: dict = field(default_factory=dict)   # {"path_hint": str, "sha256": str}
    hardware_profile: dict = field(default_factory=dict)       # {"snapshot": {...}, "sha256": str}
    planner_config: dict = field(default_factory=dict)
    backends: dict = field(default_factory=dict)               # stage -> "name-version"
    feasibility_summary: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)


@dataclass(frozen=True)
class MotionScore:
    meta: ScoreMeta = field(default_factory=ScoreMeta)
    note_plan: tuple = ()
    events: tuple = ()
    tracks: Optional[Tracks] = None

    def __post_init__(self):
        object.__setattr__(self, "note_plan", tuple(self.note_plan))
        object.__setattr__(self, "events", tuple(self.events))

    def validate(self) -> "MotionScore":
        for p in self.note_plan:
            p.validate()
        for e in self.events:
            e.validate()
        if self.tracks is not None:
            self.tracks.validate()
        return self

    def sorted(self) -> "MotionScore":
        """Canonical order: note_plan by note_index, events by (t_s, kind, params)."""
        return replace(
            self,
            note_plan=tuple(sorted(self.note_plan, key=lambda p: p.note_index)),
            events=tuple(sorted(self.events, key=lambda e: e.sort_key())),
        )

    # ---------- JSON ----------

    def to_json_dict(self) -> dict:
        s = self.sorted()
        return {
            "schema_version": MOTION_SCHEMA_VERSION,
            "meta": {
                "source_note_sequence": dict(s.meta.source_note_sequence),
                "hardware_profile": dict(s.meta.hardware_profile),
                "planner_config": dict(s.meta.planner_config),
                "backends": dict(s.meta.backends),
                "feasibility_summary": dict(s.meta.feasibility_summary),
                "extra": dict(s.meta.extra),
            },
            "note_plan": [_note_plan_to_dict(p) for p in s.note_plan],
            "events": [{"t_s": e.t_s, "kind": e.kind, "params": dict(e.params)} for e in s.events],
            "tracks": None if s.tracks is None else {
                "hop_s": s.tracks.hop_s,
                "start_s": s.tracks.start_s,
                "channels": {k: list(v) for k, v in s.tracks.channels.items()},
            },
        }

    def to_json(self, path: "str | Path | None" = None) -> str:
        text = json.dumps(self.to_json_dict(), sort_keys=True, indent=1, allow_nan=False) + "\n"
        if path is not None:
            Path(path).write_text(text, encoding="utf-8", newline="\n")
        return text

    @classmethod
    def from_json_dict(cls, d: dict) -> "MotionScore":
        version = d.get("schema_version")
        _require(isinstance(version, str) and bool(version), "missing schema_version")
        _require(version.split(".")[0] == MOTION_SCHEMA_VERSION.split(".")[0],
                 f"incompatible schema_version {version!r} (this build reads {MOTION_SCHEMA_VERSION})")
        m = d.get("meta", {})
        meta = ScoreMeta(
            source_note_sequence=dict(m.get("source_note_sequence", {})),
            hardware_profile=dict(m.get("hardware_profile", {})),
            planner_config=dict(m.get("planner_config", {})),
            backends=dict(m.get("backends", {})),
            feasibility_summary=dict(m.get("feasibility_summary", {})),
            extra=dict(m.get("extra", {})),
        )
        note_plan = tuple(_note_plan_from_dict(pd) for pd in d.get("note_plan", []))
        events = tuple(Event(t_s=ed["t_s"], kind=ed["kind"], params=dict(ed.get("params", {})))
                       for ed in d.get("events", []))
        td = d.get("tracks")
        tracks = None if td is None else Tracks(hop_s=td["hop_s"], start_s=td["start_s"],
                                                channels=td.get("channels", {}))
        return cls(meta=meta, note_plan=note_plan, events=events, tracks=tracks).validate()

    @classmethod
    def from_json(cls, source: "str | Path") -> "MotionScore":
        if isinstance(source, Path) or (isinstance(source, str) and "\n" not in source
                                        and source.strip().endswith(".json")):
            text = Path(source).read_text(encoding="utf-8")
        else:
            text = source
        return cls.from_json_dict(json.loads(text))


def _note_plan_to_dict(p: NotePlan) -> dict:
    d: dict[str, Any] = {
        "note_index": p.note_index,
        "pitch_hz": p.pitch_hz,
        "onset_s": p.onset_s,
        "string": p.string,
        "finger": p.finger,
        "position_st": p.position_st,
        "position_mm": p.position_mm,
        "bow": {
            "segment_id": p.bow.segment_id,
            "v_b_mps": p.bow.v_b_mps,
            "force_n": p.bow.force_n,
            "beta": p.bow.beta,
            "u_brightness": p.bow.u_brightness,
        },
        "realized_onset_s": p.realized_onset_s,
        "realized_duration_s": p.realized_duration_s,
        "violations": [
            {"kind": v.kind, "axis": v.axis, "needed": v.needed,
             "available": v.available, "late_by_s": v.late_by_s}
            for v in p.violations
        ],
    }
    if p.vibrato is not None:
        d["vibrato"] = {
            "rate_hz": p.vibrato.rate_hz,
            "depth_cents": p.vibrato.depth_cents,
            "delay_s": p.vibrato.delay_s,
            "depth_cents_clipped": p.vibrato.depth_cents_clipped,
        }
    return d


def _note_plan_from_dict(d: dict) -> NotePlan:
    vib = d.get("vibrato")
    return NotePlan(
        note_index=d["note_index"],
        pitch_hz=d["pitch_hz"],
        onset_s=d["onset_s"],
        string=d["string"],
        finger=d["finger"],
        position_st=d["position_st"],
        position_mm=d["position_mm"],
        bow=BowNotePlan(**d["bow"]),
        realized_onset_s=d["realized_onset_s"],
        realized_duration_s=d["realized_duration_s"],
        vibrato=None if vib is None else VibratoPlan(**vib),
        violations=tuple(Violation(**vd) for vd in d.get("violations", [])),
    )


# ---------- FeasibilityReport (separate artifact, D-023/D-028) ----------

@dataclass(frozen=True)
class FeasibilityReport:
    """What the topology comparison consumes. summary keys: n_notes,
    n_violations, feasibility_pct, total_late_s, worst_late_s.
    axis_utilization: axis name -> {peak_v, p95_v, peak_a, p95_a, v_limit,
    a_limit, utilization_v, utilization_a}."""

    summary: dict = field(default_factory=dict)
    violations: tuple = ()        # dicts: {note_index, kind, axis, needed, available, late_by_s}
    axis_utilization: dict = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "violations", tuple(self.violations))

    def to_json_dict(self) -> dict:
        return {
            "report_schema_version": REPORT_SCHEMA_VERSION,
            "summary": dict(self.summary),
            "violations": [dict(v) for v in self.violations],
            "axis_utilization": {k: dict(v) for k, v in sorted(self.axis_utilization.items())},
        }

    def to_json(self, path: "str | Path | None" = None) -> str:
        text = json.dumps(self.to_json_dict(), sort_keys=True, indent=1, allow_nan=False) + "\n"
        if path is not None:
            Path(path).write_text(text, encoding="utf-8", newline="\n")
        return text

    @classmethod
    def from_json_dict(cls, d: dict) -> "FeasibilityReport":
        version = d.get("report_schema_version", "")
        _require(version.split(".")[0] == REPORT_SCHEMA_VERSION.split(".")[0],
                 f"incompatible report_schema_version {version!r}")
        return cls(summary=dict(d.get("summary", {})),
                   violations=tuple(dict(v) for v in d.get("violations", [])),
                   axis_utilization={k: dict(v) for k, v in d.get("axis_utilization", {}).items()})

    @classmethod
    def from_json(cls, source: "str | Path") -> "FeasibilityReport":
        text = Path(source).read_text(encoding="utf-8") if str(source).endswith(".json") else str(source)
        return cls.from_json_dict(json.loads(text))
