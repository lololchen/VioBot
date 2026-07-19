"""profile_io — load/save HardwareProfile JSONs + sha256 pinning (D-025).

Follows melody_extractor.config_io conventions: unknown keys dropped silently,
missing keys fall back to dataclass defaults, lists coerced to tuples where the
default is a tuple, byte-deterministic save (sorted keys, indent=2, LF,
trailing newline). Every MotionScore embeds `profile_hash()` of the profile it
was planned against, so a plan can never be silently re-interpreted under
different hardware assumptions.

Profile JSON shape::

    {
      "profile_schema_version": "1",
      "name": "...", "comment": "...",
      "topology": {...}, "strings": {...}, "fingerboard": {...},
      "bow": {"belt": {...}, "y": {...}, "incl": {...}, "force": {...},
              "schelleng": {...}, "guettler": {...}, ...},
      "fingers": [{"x": {...}, "press": {...}, "z": {...}, "motor": {...},
                   "device_ids": {...}}, ...],
      "timing": {...}
    }
"""
from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import Any

from melody_extractor.config_io import config_from_dict

from .hardware import (
    PROFILE_SCHEMA_VERSION,
    BeltAxis,
    BowConfig,
    FingerAxis,
    Fingerboard,
    FingerUnit,
    ForceAxis,
    GuettlerConstants,
    HardwareProfile,
    LinearAxis,
    PressAxis,
    RotaryAxis,
    SchellengConstants,
    StringsConfig,
    Timing,
    Topology,
)
from .schema import sha256_of_text


def _flat_to_dict(config: Any) -> dict:
    out: dict = {}
    for f in fields(config):
        value = getattr(config, f.name)
        out[f.name] = list(value) if isinstance(value, tuple) else value
    return out


def _bow_to_dict(bow: BowConfig) -> dict:
    return {
        "belt": _flat_to_dict(bow.belt),
        "y": _flat_to_dict(bow.y),
        "incl": _flat_to_dict(bow.incl),
        "force": _flat_to_dict(bow.force),
        "r_contact_m": bow.r_contact_m,
        "beta_actuated": bow.beta_actuated,
        "beta_default": bow.beta_default,
        "beta_range": list(bow.beta_range),
        "schelleng": _flat_to_dict(bow.schelleng),
        "guettler": _flat_to_dict(bow.guettler),
        "tilt": bow.tilt,
    }


def _bow_from_dict(d: dict) -> BowConfig:
    return BowConfig(
        belt=config_from_dict(BeltAxis, d.get("belt", {}) or {}),
        y=config_from_dict(LinearAxis, d.get("y", {}) or {}),
        incl=config_from_dict(RotaryAxis, d.get("incl", {}) or {}),
        force=config_from_dict(ForceAxis, d.get("force", {}) or {}),
        r_contact_m=d.get("r_contact_m", BowConfig().r_contact_m),
        beta_actuated=d.get("beta_actuated", BowConfig().beta_actuated),
        beta_default=d.get("beta_default", BowConfig().beta_default),
        beta_range=tuple(d.get("beta_range", BowConfig().beta_range)),
        schelleng=config_from_dict(SchellengConstants, d.get("schelleng", {}) or {}),
        guettler=config_from_dict(GuettlerConstants, d.get("guettler", {}) or {}),
        tilt=d.get("tilt"),
    )


def _finger_to_dict(f: FingerUnit) -> dict:
    return {
        "x": _flat_to_dict(f.x),
        "press": _flat_to_dict(f.press),
        "z": _flat_to_dict(f.z),
        "motor": dict(f.motor),
        "device_ids": dict(f.device_ids),
    }


def _finger_from_dict(d: dict) -> FingerUnit:
    return FingerUnit(
        x=config_from_dict(FingerAxis, d.get("x", {}) or {}),
        press=config_from_dict(PressAxis, d.get("press", {}) or {}),
        z=config_from_dict(FingerAxis, d.get("z", {}) or {}),
        motor=dict(d.get("motor", FingerUnit().motor)),
        device_ids=dict(d.get("device_ids", FingerUnit().device_ids)),
    )


def profile_to_dict(profile: HardwareProfile) -> dict:
    return {
        "profile_schema_version": PROFILE_SCHEMA_VERSION,
        "name": profile.name,
        "comment": profile.comment,
        "topology": _flat_to_dict(profile.topology),
        "strings": _flat_to_dict(profile.strings),
        "fingerboard": _flat_to_dict(profile.fingerboard),
        "bow": _bow_to_dict(profile.bow),
        "fingers": [_finger_to_dict(f) for f in profile.fingers],
        "timing": _flat_to_dict(profile.timing),
    }


def profile_from_dict(d: dict) -> HardwareProfile:
    version = d.get("profile_schema_version")
    if not isinstance(version, str) or not version:
        raise ValueError("profile: missing profile_schema_version")
    if version.split(".")[0] != PROFILE_SCHEMA_VERSION:
        raise ValueError(
            f"incompatible profile_schema_version {version!r} "
            f"(this build reads major version {PROFILE_SCHEMA_VERSION!r})")
    return HardwareProfile(
        name=d.get("name", ""),
        comment=d.get("comment", ""),
        topology=config_from_dict(Topology, d.get("topology", {}) or {}),
        strings=config_from_dict(StringsConfig, d.get("strings", {}) or {}),
        fingerboard=config_from_dict(Fingerboard, d.get("fingerboard", {}) or {}),
        bow=_bow_from_dict(d.get("bow", {}) or {}),
        fingers=tuple(_finger_from_dict(fd) for fd in d.get("fingers", [])) or (FingerUnit(),),
        timing=config_from_dict(Timing, d.get("timing", {}) or {}),
    ).validate()


def profile_canonical_text(profile: HardwareProfile) -> str:
    return json.dumps(profile_to_dict(profile), sort_keys=True, indent=2, allow_nan=False) + "\n"


def profile_hash(profile: HardwareProfile) -> str:
    """sha256 over the canonical profile bytes — pinned into MotionScore.meta."""
    return sha256_of_text(profile_canonical_text(profile))


def save_profile(profile: HardwareProfile, path: "str | Path") -> Path:
    """Byte-deterministic write: two saves of an equal profile are identical."""
    path = Path(path)
    if path.parent != Path(""):
        path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(profile_canonical_text(profile), encoding="utf-8", newline="\n")
    return path


def load_profile(path: "str | Path") -> HardwareProfile:
    return profile_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
