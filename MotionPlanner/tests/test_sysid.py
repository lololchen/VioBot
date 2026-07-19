import pytest

from motion_planner.firmware_bridge.streamer import stream_dry_run
from motion_planner.simulate import simulate
from motion_planner.sysid import (
    LearnedBowSoundModel,
    SweepConfig,
    SweepDataset,
    SweepPoint,
    point_to_score,
    sweep_points,
)


def test_grid_enumeration_deterministic():
    config = SweepConfig(strings=(0, 2), positions_st=(0.0, 5.0), v_b_mps=(0.2,),
                         force_n=(0.8,), beta=(0.1,))
    points = sweep_points(config)
    assert len(points) == 4
    assert points[0].point_id == "s0_p0_v0.2_f0.8_b0.1"
    assert [p.point_id for p in points] == [p.point_id for p in sweep_points(config)]


def test_dataset_json_roundtrip():
    ds = SweepDataset(points=(SweepPoint(point_id="x", controls={"string": 0},
                                         measured={"f0_hz": 196.0, "amp_db": -20.0}),),
                      meta={"session": "bench-1"})
    text = ds.to_json()
    assert SweepDataset.from_json(text).to_json() == text
    with pytest.raises(ValueError, match="sweep_schema_version"):
        SweepDataset.from_json(text.replace('"1"', '"9"'))


def test_point_score_is_streamable_and_sounds(profiles):
    profile = profiles["concept_b_4finger"]
    point = sweep_points(SweepConfig(strings=(2,), positions_st=(5.0,), v_b_mps=(0.2,),
                                     force_n=(0.8,), beta=(0.1,)))[0]
    score = point_to_score(point, profile)
    assert score.meta.extra["sweep_point_id"] == point.point_id
    # Streamable through the firmware bridge unchanged:
    log = stream_dry_run(score, "f2.press_n")
    assert len(log) > 500
    # And the forward sim hears one tone at the commanded pitch:
    predicted = simulate(score)
    assert len(predicted.notes) == 1
    expected_hz = profile.strings.open_hz[2] * 2 ** (5.0 / 12.0)
    assert predicted.notes[0].pitch_hz == pytest.approx(expected_hz, rel=0.01)


def test_learned_model_is_a_documented_stub():
    with pytest.raises(NotImplementedError, match="SysID_Protocol"):
        LearnedBowSoundModel(SweepDataset())
