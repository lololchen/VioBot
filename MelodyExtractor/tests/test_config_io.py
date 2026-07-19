"""config_io contract tests: round-tripping, unknown-key tolerance, missing
sections falling back to code defaults, version gating, and byte-determinism."""
from __future__ import annotations

import json

import pytest

from melody_extractor.config_io import (
    PRESET_SCHEMA_VERSION,
    Preset,
    config_from_dict,
    load_preset,
    preset_from_dict,
    preset_to_dict,
    save_preset,
)
from melody_extractor.reducer import StageConfig
from melody_extractor.soundsim import RenderConfig
from melody_extractor.timbre import TimbreConfig
from melody_extractor.transcriber import MonoConfig


def _sample_preset() -> Preset:
    return Preset(
        name="bright_bow",
        comment="brighter timbre + tighter voicing_threshold for a noisy source",
        mono=MonoConfig(voicing_threshold=0.6, backend="yin"),
        timbre=TimbreConfig(n_harmonics=6),
        stage=StageConfig(max_voices=2, w_jump=0.9),
        render=RenderConfig(n_partials=6),
    )


# ---------------------------------------------------------------------------
# round-trip
# ---------------------------------------------------------------------------

def test_preset_dict_round_trip_equality():
    preset = _sample_preset()
    round_tripped = preset_from_dict(preset_to_dict(preset))
    assert round_tripped == preset


def test_save_load_round_trip_equality(tmp_path):
    preset = _sample_preset()
    path = tmp_path / "bright_bow.json"
    save_preset(preset, path)

    loaded = load_preset(path)
    assert loaded == preset


def test_config_from_dict_round_trip_for_each_config_type():
    stage = StageConfig(max_voices=3, w_frag=1.5)
    assert config_from_dict(StageConfig, {"max_voices": 3, "w_frag": 1.5}) == stage

    mono = MonoConfig(hop_s=0.02, fmax_hz=1500.0)
    assert config_from_dict(MonoConfig, {"hop_s": 0.02, "fmax_hz": 1500.0}) == mono


# ---------------------------------------------------------------------------
# unknown keys
# ---------------------------------------------------------------------------

def test_config_from_dict_swallows_reducer_version_from_config_dict_output():
    stage = StageConfig(max_voices=2, w_jump=0.9)
    dumped = stage.config_dict()  # includes "reducer_version", not a StageConfig field
    assert "reducer_version" in dumped

    rebuilt = config_from_dict(StageConfig, dumped)
    assert rebuilt == stage


def test_config_from_dict_coerces_open_strings_hz_list_to_tuple():
    dumped = StageConfig().config_dict()
    assert isinstance(dumped["open_strings_hz"], list)

    rebuilt = config_from_dict(StageConfig, dumped)
    assert isinstance(rebuilt.open_strings_hz, tuple)
    assert rebuilt.open_strings_hz == StageConfig().open_strings_hz


def test_config_from_dict_drops_arbitrary_unknown_keys():
    rebuilt = config_from_dict(MonoConfig, {"hop_s": 0.02, "totally_unknown_field": 123})
    assert rebuilt == MonoConfig(hop_s=0.02)


# ---------------------------------------------------------------------------
# missing sections -> defaults
# ---------------------------------------------------------------------------

def test_config_from_dict_missing_keys_fall_back_to_field_defaults():
    rebuilt = config_from_dict(StageConfig, {"max_voices": 2})
    expected = StageConfig(max_voices=2)
    assert rebuilt == expected


def test_preset_from_dict_missing_configs_section_uses_all_defaults():
    d = {"preset_schema_version": "1", "name": "bare", "comment": ""}
    preset = preset_from_dict(d)

    assert preset.name == "bare"
    assert preset.mono == MonoConfig()
    assert preset.timbre == TimbreConfig()
    assert preset.stage == StageConfig()
    assert preset.render == RenderConfig()


def test_preset_from_dict_missing_individual_config_section_uses_default():
    d = {
        "preset_schema_version": "1",
        "name": "partial",
        "comment": "",
        "configs": {"stage": {"max_voices": 2}},
    }
    preset = preset_from_dict(d)

    assert preset.stage == StageConfig(max_voices=2)
    assert preset.mono == MonoConfig()
    assert preset.timbre == TimbreConfig()
    assert preset.render == RenderConfig()


# ---------------------------------------------------------------------------
# version gate
# ---------------------------------------------------------------------------

def test_preset_from_dict_rejects_incompatible_major_version():
    d = {"preset_schema_version": "2.0", "name": "x", "comment": "", "configs": {}}
    with pytest.raises(ValueError):
        preset_from_dict(d)


def test_preset_from_dict_rejects_missing_version():
    d = {"name": "x", "comment": "", "configs": {}}
    with pytest.raises(ValueError):
        preset_from_dict(d)


def test_preset_from_dict_accepts_current_major_with_different_minor():
    d = {"preset_schema_version": f"{PRESET_SCHEMA_VERSION}.7", "name": "x", "comment": "", "configs": {}}
    preset = preset_from_dict(d)  # should not raise
    assert preset.name == "x"


# ---------------------------------------------------------------------------
# byte-determinism
# ---------------------------------------------------------------------------

def test_save_preset_is_byte_identical_across_two_saves(tmp_path):
    preset = _sample_preset()
    path_a = tmp_path / "a" / "preset.json"
    path_b = tmp_path / "b" / "preset.json"

    save_preset(preset, path_a)
    save_preset(preset, path_b)

    assert path_a.read_bytes() == path_b.read_bytes()


def test_save_preset_json_shape_and_formatting(tmp_path):
    preset = _sample_preset()
    path = tmp_path / "preset.json"
    save_preset(preset, path)

    text = path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert "\r" not in text

    d = json.loads(text)
    assert d["preset_schema_version"] == "1"
    assert set(d["configs"].keys()) == {"mono", "timbre", "stage", "render"}

    # sorted keys, 2-space indent -- re-serializing must reproduce the file exactly.
    reserialized = json.dumps(d, sort_keys=True, indent=2, allow_nan=False) + "\n"
    assert reserialized == text


def test_save_preset_bytes_contain_no_absolute_paths(tmp_path):
    preset = _sample_preset()
    path = tmp_path / "some_dir" / "preset.json"
    save_preset(preset, path)

    text = path.read_text(encoding="utf-8")
    assert str(tmp_path) not in text
    assert str(path) not in text
