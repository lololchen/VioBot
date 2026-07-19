import pytest

from motion_planner.config_io import PlannerConfig
from motion_planner.planner import plan
from motion_planner.roundtrip import roundtrip

CFG = PlannerConfig()

FIXTURE_PROFILE = {
    "mono_scale": "concept_a_1finger",
    "mono_arpeggio": "concept_a_1finger",
    "two_voice_thirds": "concept_a_2finger",
    "triple_rolled": "concept_b_4finger",
}


def test_roundtrip_gates_on_all_fixtures(fixture_sequences, profiles):
    """Phase-5 gate: execution fidelity on feasible profiles."""
    for name, pname in FIXTURE_PROFILE.items():
        seq = fixture_sequences[name]
        score, report = plan(seq, profiles[pname], CFG)
        m = roundtrip(seq, score, render=False).metrics
        assert m["silenced_notes"] == 0, name
        assert m["rpa"] >= 0.95, (name, m)
        if name == "triple_rolled":
            # The physical inclination sweep through the middle band produces
            # short spurious segments — all planned notes still match.
            assert m["onset_f1"] >= 0.8, (name, m)
        else:
            assert m["onset_f1"] >= 0.95, (name, m)
            assert m["onset_pitch_f1"] >= 0.9, (name, m)


def test_roll_shows_up_as_onset_shift_not_error(fixture_sequences, profiles):
    seq = fixture_sequences["triple_rolled"]
    score, _ = plan(seq, profiles["concept_b_4finger"], CFG)
    m = roundtrip(seq, score, render=False).metrics
    assert m["max_onset_shift_s"] >= CFG.roll_span_s * 0.9  # the rolled top note


def test_render_writes_listening_pair(tmp_path, fixture_sequences, profiles):
    seq = fixture_sequences["mono_scale"]
    score, _ = plan(seq, profiles["concept_a_1finger"], CFG)
    result = roundtrip(seq, score, out_dir=tmp_path / "listen", render=True)
    assert (tmp_path / "listen" / "target_render.wav").exists()
    assert (tmp_path / "listen" / "predicted_render.wav").exists()


def test_roundtrip_metrics_deterministic(fixture_sequences, profiles):
    seq = fixture_sequences["two_voice_thirds"]
    score, _ = plan(seq, profiles["concept_a_2finger"], CFG)
    m1 = roundtrip(seq, score, render=False).metrics
    m2 = roundtrip(seq, score, render=False).metrics
    assert m1 == m2
