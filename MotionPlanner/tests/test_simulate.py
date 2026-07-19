import dataclasses

import pytest

from motion_planner.config_io import PlannerConfig
from motion_planner.planner import plan
from motion_planner.simulate import simulate

CFG = PlannerConfig()


def test_predicted_notes_match_planned_pitches(fixture_sequences, profiles):
    seq = fixture_sequences["mono_scale"]
    score, _ = plan(seq, profiles["concept_a_1finger"], CFG)
    predicted = simulate(score)
    assert len(predicted.notes) == len(seq.notes)
    target = sorted(n.pitch_hz for n in seq.notes)
    got = sorted(n.pitch_hz for n in predicted.notes)
    for t, g in zip(target, got):
        assert abs(1200.0 * (g / t - 1.0)) < 60.0 or abs(g - t) / t < 0.03


def test_simulator_metadata_and_features(fixture_sequences, profiles):
    score, _ = plan(fixture_sequences["mono_scale"], profiles["concept_a_1finger"], CFG)
    predicted = simulate(score)
    assert predicted.meta.backends["simulator"] == "motion-sim-0.1.0"
    assert predicted.meta.source_kind == "synthetic"
    assert any(t.name.startswith("sim_string") for t in predicted.features)


def test_lifted_bow_is_silent(fixture_sequences, profiles):
    score, _ = plan(fixture_sequences["mono_scale"], profiles["concept_a_1finger"], CFG)
    # Zero the force track: everything falls below F_min -> silence.
    tr = score.tracks
    channels = dict(tr.channels)
    channels["bow.force_n"] = tuple(0.0 for _ in channels["bow.force_n"])
    silent_score = dataclasses.replace(score, tracks=dataclasses.replace(tr, channels=channels))
    assert simulate(silent_score).notes == ()


def test_out_of_wedge_force_silences(fixture_sequences, profiles):
    score, _ = plan(fixture_sequences["mono_scale"], profiles["concept_a_1finger"], CFG)
    tr = score.tracks
    channels = dict(tr.channels)
    channels["bow.force_n"] = tuple(50.0 for _ in channels["bow.force_n"])  # way above F_max
    loud_score = dataclasses.replace(score, tracks=dataclasses.replace(tr, channels=channels))
    assert simulate(loud_score).notes == ()


def test_simulation_is_deterministic(fixture_sequences, profiles):
    score, _ = plan(fixture_sequences["triple_rolled"], profiles["concept_b_4finger"], CFG)
    assert simulate(score).to_json() == simulate(score).to_json()
