"""planner — the pure entry point of MotionPlanner.

    plan(NoteSequence, HardwareProfile, PlannerConfig)
        -> (MotionScore, FeasibilityReport)

Same purity contract as the reducer: no I/O, no RNG, byte-identical output for
identical inputs (module CLAUDE.md). Orchestrates fingering → bowing → vibrato
→ trajectory and assembles the MotionScore with provenance hashes (source
NoteSequence + HardwareProfile) so a plan can never be silently re-interpreted
against different inputs (D-023).
"""
from __future__ import annotations

from .bow_sound_model import BOW_SOUND_MODEL_VERSION
from .bowing import BOWING_VERSION, plan_bowing
from .config_io import PlannerConfig
from .fingering import FINGERING_VERSION, plan_fingering
from .hardware import HardwareProfile
from .profile_io import profile_hash, profile_to_dict
from .schema import (
    FeasibilityReport,
    MotionScore,
    NotePlan,
    ScoreMeta,
    sha256_of_text,
)
from .trajectory import TRAJECTORY_VERSION, build_trajectory
from .vibrato import VIBRATO_VERSION, plan_vibrato

from melody_extractor.schema import NoteSequence

PLANNER_VERSION = "planner-0.1.0"


def backends_dict() -> dict:
    return {
        "planner": PLANNER_VERSION,
        "fingering": FINGERING_VERSION,
        "bowing": BOWING_VERSION,
        "vibrato": VIBRATO_VERSION,
        "trajectory": TRAJECTORY_VERSION,
        "bow_sound_model": BOW_SOUND_MODEL_VERSION,
    }


def plan(seq: NoteSequence, profile: HardwareProfile, config: PlannerConfig,
         source_path_hint: str = "") -> "tuple[MotionScore, FeasibilityReport]":
    profile = profile.validate()
    canonical = seq.sorted()
    notes = canonical.notes

    fingering = plan_fingering(canonical, profile, config)
    bowing = plan_bowing(canonical, fingering, profile, config)

    vibratos: dict = {}
    vib_violations: dict = {}
    for i, note in enumerate(notes):
        a = fingering.assignments[i]
        vib, violations = plan_vibrato(note, a.position_st, a.finger, profile, config)
        vibratos[i] = vib
        if violations:
            vib_violations[i] = violations

    traj = build_trajectory(canonical, fingering, bowing, vibratos, profile, config,
                            extra_violations=vib_violations)

    note_plan = []
    for i, note in enumerate(notes):
        a = fingering.assignments[i]
        on, du = traj.realized[i]
        merged = tuple(a.violations) + tuple(bowing.violations.get(i, ())) \
            + tuple(vib_violations.get(i, ()))
        note_plan.append(NotePlan(
            note_index=i,
            pitch_hz=note.pitch_hz,
            onset_s=note.onset_s,
            string=a.string,
            finger=a.finger,
            position_st=a.position_st,
            position_mm=a.position_mm,
            bow=bowing.note_bow[i],
            realized_onset_s=on,
            realized_duration_s=du,
            vibrato=vibratos[i],
            violations=merged,
        ))

    meta = ScoreMeta(
        source_note_sequence={"path_hint": source_path_hint,
                              "sha256": sha256_of_text(canonical.to_json())},
        hardware_profile={"snapshot": profile_to_dict(profile),
                          "sha256": profile_hash(profile)},
        planner_config=config.config_dict(),
        backends=backends_dict(),
        feasibility_summary=dict(traj.report.summary),
        extra={},
    )
    score = MotionScore(meta=meta, note_plan=tuple(note_plan),
                        events=tuple(bowing.events) + traj.finger_events,
                        tracks=traj.tracks).validate()
    return score, traj.report
