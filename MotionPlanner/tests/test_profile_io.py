import pytest

from motion_planner.hardware import HardwareProfile
from motion_planner.profile_io import (
    load_profile,
    profile_from_dict,
    profile_hash,
    profile_to_dict,
    save_profile,
)


def test_roundtrip_equality_and_hash_stability(tmp_path):
    p = HardwareProfile(name="probe").validate()
    path = save_profile(p, tmp_path / "probe.json")
    again = load_profile(path)
    assert again == p
    assert profile_hash(again) == profile_hash(p)
    # Byte-determinism: saving twice is identical.
    bytes1 = path.read_bytes()
    save_profile(again, path)
    assert path.read_bytes() == bytes1


def test_unknown_keys_dropped_and_defaults_fill_in():
    d = profile_to_dict(HardwareProfile())
    d["topology"]["future_field"] = 42       # unknown key -> silently dropped
    del d["timing"]                          # missing block -> defaults
    p = profile_from_dict(d)
    assert p.timing.control_hop_s == HardwareProfile().timing.control_hop_s


def test_version_gate():
    d = profile_to_dict(HardwareProfile())
    d["profile_schema_version"] = "999"
    with pytest.raises(ValueError, match="incompatible profile_schema_version"):
        profile_from_dict(d)


def test_checked_in_profiles_load_and_differ(profiles):
    hashes = {name: profile_hash(p) for name, p in profiles.items()}
    assert len(set(hashes.values())) == 3
    assert profiles["concept_b_4finger"].topology.concept == "B"
    assert profiles["concept_a_2finger"].candidate_fingers_for_string(3) == (0, 1)


def test_fixture_regeneration_is_byte_identical(fixtures_dir):
    import generate_fixtures

    for name in generate_fixtures.STAGES:
        path = generate_fixtures.fixture_path(name)
        before = path.read_bytes()
        generate_fixtures.generate(name)
        assert path.read_bytes() == before, f"{name} regeneration changed bytes"
