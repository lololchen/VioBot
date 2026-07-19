"""Workspace manifest — the file-based handoff between module GUIs (D-030).

workspace/manifest.json maps stage keys to the latest artifact:

    {"note_sequence": {"path": "workspace/xxx.json", "sha256": "...",
                       "producer": "melody_extractor-gui", "updated_at": "..."}}

Stage keys: "note_sequence" (MelodyExtractor export), "motion_score" and
"feasibility_report" (Sound2Motion), "byte_log" (Firmware dry-run, future).
The manifest is EPHEMERAL GUI coordination state, not a pipeline artifact —
determinism rules govern the files it points at, not the manifest itself
(updated_at is allowed here and nowhere else).
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_DIR = REPO_ROOT / "workspace"
MANIFEST_PATH = WORKSPACE_DIR / "manifest.json"

STAGE_KEYS = ("note_sequence", "motion_score", "feasibility_report", "byte_log")


def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        return {}
    try:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _save_manifest(manifest: dict) -> None:
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, sort_keys=True, indent=1) + "\n",
                             encoding="utf-8", newline="\n")


def register_text(stage: str, filename: str, text: str, producer: str) -> Path:
    """Write `text` into workspace/ and point the manifest's `stage` at it."""
    if stage not in STAGE_KEYS:
        raise ValueError(f"unknown workspace stage {stage!r} (have {STAGE_KEYS})")
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    path = WORKSPACE_DIR / filename
    path.write_text(text, encoding="utf-8", newline="\n")
    manifest = load_manifest()
    manifest[stage] = {
        "path": path.relative_to(REPO_ROOT).as_posix(),
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "producer": producer,
        "updated_at": _dt.datetime.now().isoformat(timespec="seconds"),
    }
    _save_manifest(manifest)
    return path


def latest(stage: str) -> "Path | None":
    """Absolute path of the latest artifact for `stage`, if it still exists."""
    entry = load_manifest().get(stage)
    if not entry:
        return None
    path = REPO_ROOT / entry["path"]
    return path if path.exists() else None


def describe(stage: str) -> str:
    entry = load_manifest().get(stage)
    if not entry:
        return "(nothing in workspace)"
    return f"{Path(entry['path']).name} · {entry['producer']} · {entry.get('updated_at', '?')}"
