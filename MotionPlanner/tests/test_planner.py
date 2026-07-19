import dataclasses
import subprocess
import sys
from pathlib import Path

import pytest

from motion_planner.config_io import PlannerConfig
from motion_planner.hardware import FingerAxis, FingerUnit, HardwareProfile
from motion_planner.planner import plan
from motion_planner.schema import MotionScore

CFG = PlannerConfig()


def test_plan_mono_fixture_concept_a1_no_violations(fixture_sequences, profiles):
    for name in ("mono_scale", "mono_arpeggio"):
        score, report = plan(fixture_sequences[name], profiles["concept_a_1finger"], CFG)
        assert report.summary["feasibility_pct"] == 100.0, \
            f"{name}: {report.violations}"
        assert report.summary["worst_late_s"] == 0.0
        assert len(score.note_plan) == len(fixture_sequences[name].notes)


def test_plan_is_byte_deterministic(fixture_sequences, profiles):
    for name, seq in fixture_sequences.items():
        s1, r1 = plan(seq, profiles["concept_b_4finger"], CFG)
        s2, r2 = plan(seq, profiles["concept_b_4finger"], CFG)
        assert s1.to_json() == s2.to_json(), name
        assert r1.to_json() == r2.to_json(), name


def test_score_json_roundtrips(fixture_sequences, profiles):
    score, _ = plan(fixture_sequences["triple_rolled"], profiles["concept_b_4finger"], CFG)
    text = score.to_json()
    assert MotionScore.from_json(text).to_json() == text


def test_tracks_respect_axis_limits(fixture_sequences, profiles):
    # The track builder must never exceed limits (deadlines slip instead).
    score, report = plan(fixture_sequences["two_voice_thirds"], profiles["concept_a_2finger"], CFG)
    for axis, stats in report.axis_utilization.items():
        if stats["v_limit"] > 0:
            assert stats["peak_v"] <= stats["v_limit"] * 1.05, axis
    tr = score.tracks
    assert tr is not None and tr.n_samples() > 0
    assert set(tr.channels) >= {"bow.speed_mps", "bow.force_n", "bow.inclination_rad",
                                "bow.y_m", "bow.beta", "f0.z_m", "f0.x_m", "f0.press_n"}


def test_tightened_limits_produce_violations(fixture_sequences, profiles):
    p = profiles["concept_a_1finger"]
    slow_z = FingerAxis(range_m=0.28, v_max_mps=0.01, a_max_mps2=0.05, bandwidth_hz=1.0)
    crippled = dataclasses.replace(
        p, fingers=(dataclasses.replace(p.fingers[0], z=slow_z),))
    _, healthy = plan(fixture_sequences["mono_scale"], p, CFG)
    _, crippled_report = plan(fixture_sequences["mono_scale"], crippled, CFG)
    assert crippled_report.summary["n_violations"] > healthy.summary["n_violations"]
    assert crippled_report.summary["feasibility_pct"] < 100.0


def test_rolled_realized_onsets_survive_into_note_plan(fixture_sequences, profiles):
    score, _ = plan(fixture_sequences["triple_rolled"], profiles["concept_b_4finger"], CFG)
    notes = fixture_sequences["triple_rolled"].sorted().notes
    triple = [p for p in score.note_plan if notes[p.note_index].rolled]
    assert len(triple) == 3
    realized = sorted(p.realized_onset_s for p in triple)
    assert realized[0] == realized[1] < realized[2]  # D-024 pair-then-pair


def test_meta_pins_source_and_profile(fixture_sequences, profiles):
    seq = fixture_sequences["mono_scale"]
    score, _ = plan(seq, profiles["concept_a_1finger"], CFG, source_path_hint="mono_scale.json")
    from motion_planner.profile_io import profile_hash
    from motion_planner.schema import sha256_of_text
    assert score.meta.source_note_sequence["sha256"] == sha256_of_text(seq.sorted().to_json())
    assert score.meta.hardware_profile["sha256"] == profile_hash(profiles["concept_a_1finger"])
    assert score.meta.backends["fingering"].startswith("viterbi-fingering-")


def test_cli_plan_end_to_end(tmp_path, fixtures_dir):
    import generate_fixtures

    src = generate_fixtures.fixture_path("mono_scale")
    out = tmp_path / "mono.motion.json"
    profile = Path(__file__).resolve().parents[1] / "profiles" / "concept_a_1finger.json"
    from motion_planner.cli import main
    rc = main(["plan", str(src), "--profile", str(profile), "-o", str(out)])
    assert rc == 0
    assert out.exists() and out.with_suffix(".feasibility.json").exists()
    score = MotionScore.from_json(out)
    assert len(score.note_plan) == 8
