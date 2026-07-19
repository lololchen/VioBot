from pathlib import Path

import pytest

from motion_planner.compare import (
    CompareReport,
    motor_count,
    run_compare,
    scale_tempo,
    tempo_headroom,
    write_markdown,
)
from motion_planner.config_io import PlannerConfig

CFG = PlannerConfig()
ROOT = Path(__file__).resolve().parents[1]


def test_scale_tempo_scales_all_times(fixture_sequences):
    seq = fixture_sequences["mono_scale"]
    fast = scale_tempo(seq, 2.0)
    assert fast.notes[1].onset_s == pytest.approx(seq.sorted().notes[1].onset_s / 2.0)
    assert fast.notes[0].duration_s == pytest.approx(seq.sorted().notes[0].duration_s / 2.0)
    assert fast.notes[0].amp_db_envelope.times_s[-1] == \
        pytest.approx(seq.sorted().notes[0].amp_db_envelope.times_s[-1] / 2.0)


def test_motor_counts():
    from motion_planner.profile_io import load_profile
    assert motor_count(load_profile(ROOT / "profiles" / "concept_a_1finger.json")) == 6
    assert motor_count(load_profile(ROOT / "profiles" / "concept_a_2finger.json")) == 9
    assert motor_count(load_profile(ROOT / "profiles" / "concept_b_4finger.json")) == 11


def test_topology_ranking_on_double_stops(fixture_sequences, profiles):
    """The A-1 < A-2 ≈ B ordering on two_voice_thirds is the concept-comparison
    sanity check from the build plan."""
    seq = fixture_sequences["two_voice_thirds"]
    h1 = tempo_headroom(seq, profiles["concept_a_1finger"], CFG)
    h2 = tempo_headroom(seq, profiles["concept_a_2finger"], CFG)
    h4 = tempo_headroom(seq, profiles["concept_b_4finger"], CFG)
    assert h1 == 0.0          # one finger cannot stop two strings at once
    assert h2 > 1.0 and h4 > 1.0


def test_report_roundtrip_and_markdown(fixtures_dir, tmp_path):
    import generate_fixtures

    report = run_compare(
        profile_paths=[ROOT / "profiles" / "concept_a_1finger.json"],
        input_paths=[generate_fixtures.fixture_path("mono_scale")])
    text = report.to_json()
    assert CompareReport.from_json(text).to_json() == text
    md = write_markdown(report)
    assert "tempo_headroom" in md and "concept_a_1finger" in md


def test_compare_is_deterministic(fixtures_dir):
    import generate_fixtures

    args = dict(profile_paths=[ROOT / "profiles" / "concept_b_4finger.json"],
                input_paths=[generate_fixtures.fixture_path("triple_rolled")])
    assert run_compare(**args).to_json() == run_compare(**args).to_json()


@pytest.mark.slow
def test_checked_in_baseline_reproduces(fixtures_dir):
    """out/compare_baseline.json is decision-governed (D-028): regeneration
    must be byte-identical, else a decisions.md entry is due."""
    import generate_fixtures

    baseline_path = ROOT / "out" / "compare_baseline.json"
    if not baseline_path.exists():
        pytest.skip("no baseline checked in yet")
    report = run_compare(
        profile_paths=sorted((ROOT / "profiles").glob("concept_*.json")),
        input_paths=[generate_fixtures.fixture_path(n) for n in generate_fixtures.STAGES])
    assert report.to_json() == baseline_path.read_text(encoding="utf-8")
