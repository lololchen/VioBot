"""Topology comparison — corpus × profiles → the report that judges Concept A
vs Concept B before any hardware exists (F6, D-028).

Headline metric: **tempo_headroom** — the maximum uniform tempo multiplier k
(fixed 24-step bisection on k ∈ [0.25, 2.0], deterministic) at which the piece
still plans with 100% feasibility on that profile. "A-2 plays this up to 1.3×"
is the sentence hardware decisions get made on. Axis-utilization maxima feed
the motor KV choice (PRD open Q6).

All numbers flow from the same pure plan()/simulate() path used everywhere
else; this module only loops, scales tempo, and aggregates. Reports are
deterministic JSON (+ a generated markdown table); the first fixture run is
checked in as out/compare_baseline.json — updating it requires a decisions.md
entry (same governance as MelodyExtractor's eval_baseline, D-016/D-028).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

from melody_extractor.schema import AmpEnvelope, F0Contour, NoteSequence

from .config_io import PlannerConfig
from .planner import plan
from .profile_io import load_profile, profile_hash
from .roundtrip import roundtrip

COMPARE_VERSION = "compare-0.1.0"
COMPARE_SCHEMA_VERSION = "0.1.0"

_K_LO, _K_HI = 0.25, 2.0
_BISECT_STEPS = 24


def scale_tempo(seq: NoteSequence, k: float) -> NoteSequence:
    """Uniform tempo multiplier: k=1.3 plays 30% faster (times divided by k).
    Envelope/contour times scale with the note (their rates scale by k)."""
    notes = []
    for n in seq.sorted().notes:
        env = AmpEnvelope(times_s=tuple(t / k for t in n.amp_db_envelope.times_s),
                          amp_db=n.amp_db_envelope.amp_db)
        contour = None
        if n.f0_contour is not None:
            contour = F0Contour(times_s=tuple(t / k for t in n.f0_contour.times_s),
                                f0_hz=n.f0_contour.f0_hz)
        notes.append(replace(n, onset_s=n.onset_s / k, duration_s=n.duration_s / k,
                             amp_db_envelope=env, f0_contour=contour))
    return replace(seq, notes=tuple(notes))


def motor_count(profile) -> int:
    """Bow: belt + differential pair = 3. Fingers: 3 axes each for roaming
    Concept A, 2 (press+z) for per-string Concept B (no string-select DoF)."""
    per_finger = 2 if profile.topology.concept == "B" else 3
    return 3 + per_finger * profile.topology.n_fingers


def _feasible_at(seq: NoteSequence, k: float, profile, config: PlannerConfig) -> bool:
    _, report = plan(scale_tempo(seq, k), profile, config)
    return report.summary["feasibility_pct"] >= 100.0 - 1e-9


def tempo_headroom(seq: NoteSequence, profile, config: PlannerConfig) -> float:
    """Max k in [0.25, 2.0] with 100% feasibility; 0.0 when even 0.25× fails."""
    if not _feasible_at(seq, _K_LO, profile, config):
        return 0.0
    if _feasible_at(seq, _K_HI, profile, config):
        return _K_HI
    lo, hi = _K_LO, _K_HI
    for _ in range(_BISECT_STEPS):
        mid = (lo + hi) / 2.0
        if _feasible_at(seq, mid, profile, config):
            lo = mid
        else:
            hi = mid
    return round(lo, 4)


@dataclass(frozen=True)
class CompareReport:
    rows: tuple = ()                 # dicts, sorted by (profile, piece)
    meta: dict = field(default_factory=dict)

    def to_json_dict(self) -> dict:
        return {"compare_schema_version": COMPARE_SCHEMA_VERSION,
                "meta": dict(self.meta), "rows": [dict(r) for r in self.rows]}

    def to_json(self) -> str:
        return json.dumps(self.to_json_dict(), sort_keys=True, indent=1,
                          allow_nan=False) + "\n"

    @classmethod
    def from_json(cls, source: "str | Path") -> "CompareReport":
        text = Path(source).read_text(encoding="utf-8") if str(source).endswith(".json") \
            else str(source)
        d = json.loads(text)
        return cls(rows=tuple(dict(r) for r in d.get("rows", [])), meta=dict(d.get("meta", {})))


def evaluate_pair(seq: NoteSequence, profile, config: PlannerConfig,
                  piece: str, render: bool = False) -> dict:
    score, report = plan(seq, profile, config)
    rt = roundtrip(seq, score, render=False).metrics

    hist: dict = {}
    for v in report.violations:
        key = f"{v['kind']}:{v['axis']}"
        hist[key] = hist.get(key, 0) + 1

    detected = [p for p in score.note_plan if p.vibrato is not None]
    unclipped = [p for p in detected if not p.vibrato.depth_cents_clipped]
    notes = seq.sorted().notes
    rolled = [p for p in score.note_plan if notes[p.note_index].rolled]
    roll_budget = config.roll_span_s + 0.2
    roll_ok = [p for p in rolled if p.realized_onset_s - p.onset_s <= roll_budget]

    util = {axis: round(stats["utilization_v"], 4)
            for axis, stats in report.axis_utilization.items()}
    return {
        "profile": profile.name,
        "profile_sha256": profile_hash(profile),
        "piece": piece,
        "motor_count": motor_count(profile),
        "n_fingers": profile.topology.n_fingers,
        "feasibility_pct": round(report.summary["feasibility_pct"], 2),
        "tempo_headroom": tempo_headroom(seq, profile, config),
        "total_late_s": round(report.summary["total_late_s"], 4),
        "worst_late_s": round(report.summary["worst_late_s"], 4),
        "violations_by_axis": dict(sorted(hist.items())),
        "axis_utilization_v": util,
        "vibrato_coverage": round(len(unclipped) / len(detected), 4) if detected else 1.0,
        "roll_compliance": round(len(roll_ok) / len(rolled), 4) if rolled else 1.0,
        "onset_f1": round(rt["onset_f1"], 4),
        "onset_pitch_f1": round(rt["onset_pitch_f1"], 4),
        "rpa": round(rt["rpa"], 4),
        "silenced_notes": int(rt["silenced_notes"]),
    }


def run_compare(profile_paths: "list[Path]", input_paths: "list[Path]",
                config: "PlannerConfig | None" = None, render: bool = False) -> CompareReport:
    config = config or PlannerConfig()
    profiles = sorted((load_profile(p) for p in profile_paths), key=lambda p: p.name)
    rows = []
    for profile in profiles:
        for path in sorted(input_paths, key=lambda p: p.name):
            seq = NoteSequence.from_json(path)
            rows.append(evaluate_pair(seq, profile, config, piece=path.stem, render=render))
    rows.sort(key=lambda r: (r["profile"], r["piece"]))
    meta = {"compare_version": COMPARE_VERSION,
            "planner_config": config.config_dict(),
            "n_profiles": len(profiles), "n_pieces": len(input_paths)}
    return CompareReport(rows=tuple(rows), meta=meta)


def write_markdown(report: CompareReport) -> str:
    cols = ("profile", "piece", "motor_count", "feasibility_pct", "tempo_headroom",
            "worst_late_s", "vibrato_coverage", "roll_compliance",
            "onset_f1", "rpa", "silenced_notes")
    lines = ["# Topology comparison", "",
             "| " + " | ".join(cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for r in report.rows:
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    lines += ["", "Higher tempo_headroom = the profile can play the piece faster before "
              "its first feasibility violation (1.0 = as written; 2.0 = capped).", ""]
    return "\n".join(lines)
