import json

import pytest

from motion_planner.schema import (
    MOTION_SCHEMA_VERSION,
    BowNotePlan,
    Event,
    FeasibilityReport,
    MotionScore,
    NotePlan,
    ScoreMeta,
    Tracks,
    VibratoPlan,
    Violation,
)


def _sample_score() -> MotionScore:
    plan = NotePlan(
        note_index=0, pitch_hz=440.0, onset_s=0.0, string=2, finger=0,
        position_st=0.0, position_mm=0.0,
        bow=BowNotePlan(segment_id=0, v_b_mps=0.25, force_n=0.8, beta=0.1, u_brightness=0.5),
        realized_onset_s=0.0, realized_duration_s=0.5,
        vibrato=VibratoPlan(rate_hz=5.5, depth_cents=18.0, delay_s=0.1),
        violations=(Violation(kind="late_transition", axis="f0.z", needed=0.08, available=0.05,
                              late_by_s=0.03),),
    )
    events = (
        Event(t_s=0.0, kind="bow_land", params={"y_to_m": 0.0}),
        Event(t_s=0.0, kind="finger_press", params={"finger": 0, "string": 2, "position_st": 0.0}),
        Event(t_s=0.5, kind="bow_lift", params={"y_to_m": 0.005}),
    )
    tracks = Tracks(hop_s=0.01, start_s=0.0, channels={
        "bow.speed_mps": (0.0, 0.25, 0.25), "bow.force_n": (0.0, 0.8, 0.8)})
    meta = ScoreMeta(source_note_sequence={"path_hint": "x.json", "sha256": "ab" * 32},
                     backends={"fingering": "viterbi-fingering-0.1.0"})
    return MotionScore(meta=meta, note_plan=(plan,), events=events, tracks=tracks).validate()


def test_json_roundtrip_byte_identical():
    score = _sample_score()
    text1 = score.to_json()
    score2 = MotionScore.from_json(text1)
    assert score2.to_json() == text1
    assert text1.endswith("\n")


def test_serialization_is_deterministic_across_construction_order():
    score = _sample_score()
    # Same events in reverse construction order must serialize identically.
    shuffled = MotionScore(meta=score.meta, note_plan=score.note_plan,
                           events=tuple(reversed(score.events)), tracks=score.tracks)
    assert shuffled.to_json() == score.to_json()


def test_version_gate_rejects_major_mismatch():
    d = _sample_score().to_json_dict()
    d["schema_version"] = "999.0.0"
    with pytest.raises(ValueError, match="incompatible schema_version"):
        MotionScore.from_json_dict(d)
    assert MOTION_SCHEMA_VERSION.split(".")[0] == "0"


def test_validation_rejects_bad_values():
    with pytest.raises(ValueError, match="beta"):
        BowNotePlan(segment_id=0, v_b_mps=0.2, force_n=1.0, beta=1.5, u_brightness=0.5).validate()
    with pytest.raises(ValueError, match="one direction"):
        BowNotePlan(segment_id=0, v_b_mps=-0.2, force_n=1.0, beta=0.1, u_brightness=0.5).validate()
    with pytest.raises(ValueError, match="unknown event kind"):
        Event(t_s=0.0, kind="bow_teleport").validate()
    with pytest.raises(ValueError, match="differ in length"):
        Tracks(hop_s=0.01, start_s=0.0, channels={"a": (1.0,), "b": (1.0, 2.0)}).validate()
    with pytest.raises(ValueError, match="unknown violation kind"):
        Violation(kind="nonsense").validate()


def test_allow_nan_is_enforced():
    tracks = Tracks(hop_s=0.01, start_s=0.0, channels={"a": (float("nan"),)})
    with pytest.raises(ValueError):
        tracks.validate()
    score = MotionScore(tracks=Tracks(hop_s=0.01, start_s=0.0, channels={"a": (float("nan"),)}))
    with pytest.raises(ValueError):
        score.to_json()  # json.dumps(allow_nan=False) refuses even if validate() was skipped


def test_feasibility_report_roundtrip():
    report = FeasibilityReport(
        summary={"n_notes": 5, "n_violations": 1, "feasibility_pct": 80.0,
                 "total_late_s": 0.03, "worst_late_s": 0.03},
        violations=({"note_index": 3, "kind": "late_transition", "axis": "f0.z",
                     "needed": 0.08, "available": 0.05, "late_by_s": 0.03},),
        axis_utilization={"f0.z": {"peak_v": 0.4, "p95_v": 0.3, "peak_a": 10.0, "p95_a": 8.0,
                                   "v_limit": 0.6, "a_limit": 25.0,
                                   "utilization_v": 0.67, "utilization_a": 0.4}},
    )
    text = report.to_json()
    again = FeasibilityReport.from_json_dict(json.loads(text))
    assert again.to_json() == text
