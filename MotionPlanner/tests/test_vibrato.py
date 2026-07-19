import math

import pytest

from melody_extractor.schema import AmpEnvelope, F0Contour, Note

from motion_planner.config_io import PlannerConfig
from motion_planner.hardware import HardwareProfile, mm_per_st
from motion_planner.vibrato import clip_to_axis, extract_vibrato

CFG = PlannerConfig()


def _vibrato_note(rate_hz=5.5, depth_cents=25.0, dur=1.0, f0=440.0, hop=0.01):
    n = int(dur / hop) + 1
    times = tuple(i * hop for i in range(n))
    f0s = tuple(f0 * 2.0 ** ((depth_cents / 1200.0) * math.sin(2 * math.pi * rate_hz * t))
                for t in times)
    return Note(pitch_hz=f0, onset_s=0.0, duration_s=dur,
                amp_db_envelope=AmpEnvelope.constant(-20.0, dur),
                f0_contour=F0Contour(times_s=times, f0_hz=f0s))


def test_synthetic_vibrato_recovered():
    note = _vibrato_note(rate_hz=5.5, depth_cents=25.0)
    vib = extract_vibrato(note, CFG)
    assert vib is not None
    assert vib.rate_hz == pytest.approx(5.5, rel=0.05)
    assert vib.depth_cents == pytest.approx(25.0, rel=0.10)


def test_flat_contour_yields_none():
    note = _vibrato_note(depth_cents=2.0)  # below the 10-cent gate
    assert extract_vibrato(note, CFG) is None
    plain = Note(pitch_hz=440.0, onset_s=0.0, duration_s=0.5,
                 amp_db_envelope=AmpEnvelope.constant(-20.0, 0.5))
    assert extract_vibrato(plain, CFG) is None


def test_too_short_for_min_cycles_yields_none():
    note = _vibrato_note(rate_hz=4.0, depth_cents=25.0, dur=0.3)  # 1.2 cycles < 2
    assert extract_vibrato(note, CFG) is None


def test_depth_clip_matches_hand_computed_a_pk():
    profile = HardwareProfile().validate()
    vib = extract_vibrato(_vibrato_note(rate_hz=8.0, depth_cents=80.0), CFG)
    assert vib is not None
    clipped, violations = clip_to_axis(vib, position_st=2.0, finger=0, profile=profile)
    z = profile.fingers[0].z
    omega = 2 * math.pi * clipped.rate_hz
    dz_m = (clipped.depth_cents / 100.0) * mm_per_st(2.0, 325.0) / 1000.0
    a_pk = omega * omega * dz_m
    # After clipping, the realized peak acceleration must obey the axis limit
    # (bandwidth rolloff may push it further below).
    assert a_pk <= z.a_max_mps2 * 1.0001
    if clipped.depth_cents < vib.depth_cents:
        assert clipped.depth_cents_clipped and violations
        assert violations[0].kind == "vibrato_clipped"
    assert clipped.rate_hz == vib.rate_hz  # rate is never clipped


def test_open_string_gets_no_vibrato():
    profile = HardwareProfile().validate()
    vib = extract_vibrato(_vibrato_note(), CFG)
    clipped, violations = clip_to_axis(vib, position_st=0.0, finger=None, profile=profile)
    assert clipped is None and violations == ()


def test_moderate_vibrato_unclipped_at_first_position():
    # PRD worked example: ±25 cents @ 5.5 Hz, 1st position — QDD-feasible.
    profile = HardwareProfile().validate()
    vib = extract_vibrato(_vibrato_note(rate_hz=5.5, depth_cents=25.0), CFG)
    clipped, violations = clip_to_axis(vib, position_st=2.0, finger=0, profile=profile)
    assert not clipped.depth_cents_clipped and violations == ()
