"""config_io — load/save named preset JSONs for the four pipeline configs.

Core-level module: stdlib + config dataclasses only (no streamlit/plotly —
this file is imported by both `cli.py` and the GUI; see
docs/plan_GUI_MelodyExtractor.md). Code defaults never change here: a preset
only ever *overrides* fields the user explicitly tuned, everything else falls
back to the dataclass's own default.

Preset JSON shape::

    {
      "preset_schema_version": "1",
      "name": "...",
      "comment": "...",
      "configs": {
        "mono":   {... MonoConfig fields ...},
        "timbre": {... TimbreConfig fields ...},
        "stage":  {... StageConfig fields ...},
        "render": {... RenderConfig fields ...}
      }
    }

`config_from_dict` keeps only known dataclass fields (dropping anything else,
notably `StageConfig.config_dict()`'s "reducer_version", which is not a
StageConfig field but is a convenient thing to feed straight back in),
coerces list values to tuples wherever the field's own default is a tuple
(e.g. `StageConfig.open_strings_hz`), and falls back to the field default for
any key missing from `d`.
"""
from __future__ import annotations

import json
from dataclasses import MISSING, dataclass, fields
from pathlib import Path
from typing import Any

from .reducer import StageConfig
from .soundsim import RenderConfig
from .timbre import TimbreConfig
from .transcriber import MonoConfig

PRESET_SCHEMA_VERSION = "1"


def config_from_dict(cls: type, d: dict) -> Any:
    """Build a config dataclass instance from a plain dict.

    - Unknown keys in `d` (fields `cls` doesn't have) are dropped silently.
    - Keys missing from `d` fall back to `cls`'s own field default.
    - A list value is coerced to a tuple wherever that field's default is
      itself a tuple (e.g. `StageConfig.open_strings_hz`).
    """
    kwargs = {}
    for f in fields(cls):
        if f.name not in d:
            continue
        value = d[f.name]
        default = f.default if f.default is not MISSING else None
        if isinstance(default, tuple) and isinstance(value, list):
            value = tuple(value)
        kwargs[f.name] = value
    return cls(**kwargs)


@dataclass(frozen=True)
class Preset:
    """A named, commented bundle of the four pipeline configs."""

    name: str
    comment: str
    mono: MonoConfig
    timbre: TimbreConfig
    stage: StageConfig
    render: RenderConfig


def _config_to_plain_dict(config: Any) -> dict:
    """dataclass instance -> JSON-safe dict (tuples -> lists).

    Uses `config.config_dict()` when the dataclass provides one (StageConfig
    does, and adds a "reducer_version" entry that `config_from_dict` swallows
    on the way back in) so the on-disk shape matches what the reducer itself
    reports in `NoteSequence.meta.stage`.
    """
    if hasattr(config, "config_dict"):
        return config.config_dict()
    out: dict = {}
    for f in fields(config):
        value = getattr(config, f.name)
        out[f.name] = list(value) if isinstance(value, tuple) else value
    return out


def preset_to_dict(preset: Preset) -> dict:
    return {
        "preset_schema_version": PRESET_SCHEMA_VERSION,
        "name": preset.name,
        "comment": preset.comment,
        "configs": {
            "mono": _config_to_plain_dict(preset.mono),
            "timbre": _config_to_plain_dict(preset.timbre),
            "stage": _config_to_plain_dict(preset.stage),
            "render": _config_to_plain_dict(preset.render),
        },
    }


def preset_from_dict(d: dict) -> Preset:
    version = d.get("preset_schema_version")
    if not isinstance(version, str) or not version:
        raise ValueError("preset: missing preset_schema_version")
    major = version.split(".")[0]
    if major != PRESET_SCHEMA_VERSION:
        raise ValueError(
            f"incompatible preset_schema_version {version!r} "
            f"(this build reads major version {PRESET_SCHEMA_VERSION!r})"
        )
    configs = d.get("configs", {}) or {}
    return Preset(
        name=d.get("name", ""),
        comment=d.get("comment", ""),
        mono=config_from_dict(MonoConfig, configs.get("mono", {}) or {}),
        timbre=config_from_dict(TimbreConfig, configs.get("timbre", {}) or {}),
        stage=config_from_dict(StageConfig, configs.get("stage", {}) or {}),
        render=config_from_dict(RenderConfig, configs.get("render", {}) or {}),
    )


def save_preset(preset: Preset, path: "str | Path") -> Path:
    """Write a byte-deterministic preset JSON: sorted keys, indent=2, no NaN,
    UTF-8, LF line endings, trailing newline. Two saves of an equal Preset
    produce byte-identical files."""
    path = Path(path)
    if path.parent != Path(""):
        path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(preset_to_dict(preset), sort_keys=True, indent=2, allow_nan=False) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def load_preset(path: "str | Path") -> Preset:
    text = Path(path).read_text(encoding="utf-8")
    return preset_from_dict(json.loads(text))
