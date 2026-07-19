import math

import pytest

from melody_extractor.reducer import OPEN_STRINGS_HZ
from motion_planner.hardware import (
    FingerUnit,
    HardwareProfile,
    StringsConfig,
    Topology,
    band_index,
    hz_to_position_st,
    mm_per_st,
    mm_to_st,
    sounding_length_mm,
    st_to_mm,
    trapezoid_time,
)


def test_octave_is_half_scale_length():
    assert st_to_mm(12.0, 325.0) == pytest.approx(162.5)
    assert sounding_length_mm(12.0, 325.0) == pytest.approx(162.5)


def test_mm_to_st_inverts_st_to_mm():
    for s in (0.0, 1.0, 4.7, 12.0, 19.0):
        assert mm_to_st(st_to_mm(s, 325.0), 325.0) == pytest.approx(s, abs=1e-9)


def test_mm_per_st_matches_finite_difference():
    s = 5.0
    eps = 1e-6
    fd = (st_to_mm(s + eps, 325.0) - st_to_mm(s - eps, 325.0)) / (2 * eps)
    assert mm_per_st(s, 325.0) == pytest.approx(fd, rel=1e-6)


def test_hz_to_position_st_open_and_octave():
    assert hz_to_position_st(OPEN_STRINGS_HZ[2], OPEN_STRINGS_HZ[2]) == pytest.approx(0.0)
    assert hz_to_position_st(880.0, 440.0) == pytest.approx(12.0)


def test_trapezoid_time_triangular_and_cruise():
    # Triangular: d <= v^2/a
    assert trapezoid_time(0.01, 0.5, 20.0) == pytest.approx(2 * math.sqrt(0.01 / 20.0))
    # Cruise: d > v^2/a  ->  d/v + v/a
    assert trapezoid_time(1.0, 0.5, 20.0) == pytest.approx(1.0 / 0.5 + 0.5 / 20.0)
    assert trapezoid_time(0.0, 0.5, 20.0) == 0.0
    assert trapezoid_time(0.01, 0.0, 20.0) == math.inf


def test_band_index():
    assert band_index((0,)) == 0
    assert band_index((3,)) == 6
    assert band_index((1, 2)) == 3
    assert band_index((2, 1)) == 3  # order-insensitive
    with pytest.raises(ValueError):
        band_index((0, 2))  # non-adjacent
    with pytest.raises(ValueError):
        band_index((0, 1, 2))  # triples are rolled, never one band


def test_profile_validation_rejects_bad_configs():
    with pytest.raises(ValueError, match="n_fingers"):
        HardwareProfile(topology=Topology(concept="A", n_fingers=2, finger_home_string=(None, None))).validate()
    with pytest.raises(ValueError, match="home string"):
        HardwareProfile(topology=Topology(concept="B", n_fingers=1, finger_home_string=(None,))).validate()
    with pytest.raises(ValueError, match="OPEN_STRINGS_HZ"):
        HardwareProfile(strings=StringsConfig(open_hz=(196.0, 293.66, 440.0, 659.26))).validate()


def test_candidate_fingers_topology_dispatch():
    a2 = HardwareProfile(
        topology=Topology(concept="A", n_fingers=2, finger_home_string=(None, None)),
        fingers=(FingerUnit(), FingerUnit()),
    ).validate()
    assert a2.candidate_fingers_for_string(0) == (0, 1)
    b4 = HardwareProfile(
        topology=Topology(concept="B", n_fingers=4, finger_home_string=(0, 1, 2, 3)),
        fingers=(FingerUnit(),) * 4,
    ).validate()
    assert b4.candidate_fingers_for_string(2) == (2,)


def test_beta_eff_rises_with_position_when_fixed():
    p = HardwareProfile().validate()
    assert not p.bow.beta_actuated
    betas = [p.beta_eff(s) for s in (0.0, 5.0, 12.0, 19.0)]
    assert betas == sorted(betas)
    assert betas[0] == pytest.approx(p.bow.beta_default)
    assert betas[2] == pytest.approx(p.bow.beta_default * 2.0)  # octave halves sounding length
