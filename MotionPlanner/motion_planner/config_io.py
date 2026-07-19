"""PlannerConfig + preset JSONs (mirrors melody_extractor.config_io).

PlannerConfig holds every tunable the four planners share. Weights are
PROVISIONAL until SysID/hardware data exists — same caveat as the reducer's
cost weights (D-014). Preset JSON shape::

    {
      "planner_preset_schema_version": "1",
      "name": "...", "comment": "...",
      "configs": {"planner": {... PlannerConfig fields ...}}
    }

The hardware side deliberately does NOT live here: profiles are their own
artifact with their own hash (profile_io, D-025). A preset tunes the planning
*style*; a profile states the machine.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from melody_extractor.config_io import config_from_dict

PLANNER_PRESET_SCHEMA_VERSION = "1"


@dataclass(frozen=True)
class PlannerConfig:
    # -- fingering DP weights (F1; Maezawa shapes per D-026) --
    w_time: float = 4.0            # travel-time hinge weight
    w_shift: float = 1.0           # (Δposition_st / 7)^2 continuity cost
    w_string: float = 0.3          # |Δband| bow-travel proxy
    w_open: float = 0.2            # flat open-string penalty (timbre mismatch vs zero travel)
    steal_fraction: float = 0.3    # tail fraction of the previous note usable for transitions
    pitch_tolerance_semitones: float = 0.3   # mirrors reducer (D-015); keep equal

    # -- bowing (F2) --
    lift_gap_s: float = 0.15       # rests longer than this get bow_lift/bow_land
    roll_span_s: float = 0.08      # D-024 pair-then-pair dwell
    roll_anticipate: bool = False  # reserved (D-024); top-note-on-beat variant
    u_default: float = 0.5         # Schelleng wedge placement when no harmonics present
    u_min: float = 0.2
    u_max: float = 0.8
    coupling_warn_fraction: float = 0.25   # warn when ω·r_contact exceeds this × v_b

    # -- vibrato (F3) --
    vibrato_min_depth_cents: float = 10.0
    vibrato_min_cycles: float = 2.0
    vibrato_f_lo_hz: float = 3.0
    vibrato_f_hi_hz: float = 9.0

    # -- simulate gate --
    press_threshold_n: float = 1.0  # finger force above which the string is stopped

    def config_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class PlannerPreset:
    name: str
    comment: str
    planner: PlannerConfig


def preset_to_dict(preset: PlannerPreset) -> dict:
    return {
        "planner_preset_schema_version": PLANNER_PRESET_SCHEMA_VERSION,
        "name": preset.name,
        "comment": preset.comment,
        "configs": {"planner": preset.planner.config_dict()},
    }


def preset_from_dict(d: dict) -> PlannerPreset:
    version = d.get("planner_preset_schema_version")
    if not isinstance(version, str) or not version:
        raise ValueError("preset: missing planner_preset_schema_version")
    if version.split(".")[0] != PLANNER_PRESET_SCHEMA_VERSION:
        raise ValueError(f"incompatible planner_preset_schema_version {version!r}")
    configs = d.get("configs", {}) or {}
    return PlannerPreset(
        name=d.get("name", ""),
        comment=d.get("comment", ""),
        planner=config_from_dict(PlannerConfig, configs.get("planner", {}) or {}),
    )


def save_preset(preset: PlannerPreset, path: "str | Path") -> Path:
    path = Path(path)
    if path.parent != Path(""):
        path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(preset_to_dict(preset), sort_keys=True, indent=2, allow_nan=False) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def load_preset(path: "str | Path") -> PlannerPreset:
    return preset_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
