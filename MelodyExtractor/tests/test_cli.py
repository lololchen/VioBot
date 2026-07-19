"""CLI contract tests: extract | reduce | render | eval, invoked in-process via
`main(argv)` against the generated fixture corpus (conftest.fixtures_dir).

Covers: default/explicit output paths, MIDI vs. audio input dispatch, the
--no-timbre / --poly / --midi-out flags, reducer playability + rolled-triple
semantics through the CLI, render + --paired, the eval report's mir_eval
metrics/gates/baseline handling, and CLI-level error paths (unknown
subcommand, missing input file).
"""
from __future__ import annotations

import json

import numpy as np
import pytest
from scipy.io import wavfile

from melody_extractor.cli import main
from melody_extractor.config_io import Preset, save_preset
from melody_extractor.reducer import StageConfig, playability_violations
from melody_extractor.schema import NoteSequence, midi_to_hz
from melody_extractor.soundsim import RenderConfig
from melody_extractor.timbre import TimbreConfig
from melody_extractor.transcriber import MonoConfig


def _max_concurrent(notes) -> int:
    """Max number of notes simultaneously sounding, scanning change points."""
    if not notes:
        return 0
    points = sorted({n.onset_s for n in notes} | {n.offset_s for n in notes})
    best = 0
    for t0, t1 in zip(points, points[1:]):
        count = sum(1 for n in notes if n.onset_s <= t0 + 1e-9 and n.offset_s >= t1 - 1e-9)
        best = max(best, count)
    return best


# ---------------------------------------------------------------------------
# extract
# ---------------------------------------------------------------------------

def test_extract_audio_writes_json_with_harmonics(fixtures_dir, tmp_path):
    wav = fixtures_dir / "mono_scale.wav"
    out = tmp_path / "mono_scale_out.json"

    rc = main(["extract", str(wav), "-o", str(out)])

    assert rc == 0
    assert out.exists()
    seq = NoteSequence.from_json(out)
    assert len(seq.notes) == 8
    assert all(n.harmonics is not None for n in seq.notes)


def test_extract_no_timbre_omits_harmonics(fixtures_dir, tmp_path):
    wav = fixtures_dir / "mono_scale.wav"
    out = tmp_path / "mono_scale_no_timbre.json"

    rc = main(["extract", str(wav), "-o", str(out), "--no-timbre"])

    assert rc == 0
    seq = NoteSequence.from_json(out)
    assert len(seq.notes) == 8
    assert all(n.harmonics is None for n in seq.notes)


def test_extract_explicit_timbre_flag_is_default_on_noop(fixtures_dir, tmp_path):
    wav = fixtures_dir / "mono_scale.wav"
    out = tmp_path / "mono_scale_explicit_timbre.json"

    rc = main(["extract", str(wav), "-o", str(out), "--timbre"])

    assert rc == 0
    seq = NoteSequence.from_json(out)
    assert all(n.harmonics is not None for n in seq.notes)


def test_extract_default_output_path_is_input_stem_next_to_input(fixtures_dir, tmp_path):
    import shutil

    wav = tmp_path / "mono_scale.wav"
    shutil.copyfile(fixtures_dir / "mono_scale.wav", wav)

    rc = main(["extract", str(wav)])

    assert rc == 0
    expected = tmp_path / "mono_scale.json"
    assert expected.exists()
    assert len(NoteSequence.from_json(expected).notes) == 8


def test_extract_midi_input_gives_exact_equal_tempered_pitches(fixtures_dir, tmp_path):
    mid = fixtures_dir / "mono_scale.mid"
    out = tmp_path / "mono_scale_from_midi.json"

    rc = main(["extract", str(mid), "-o", str(out)])

    assert rc == 0
    seq = NoteSequence.from_json(out)
    assert len(seq.notes) == 8

    # C major scale, MIDI 60..72 by the offsets used in generate_fixtures.py.
    expected = sorted(midi_to_hz(60 + o) for o in (0, 2, 4, 5, 7, 9, 11, 12))
    actual = sorted(n.pitch_hz for n in seq.notes)
    for a, e in zip(actual, expected):
        assert a == pytest.approx(e, rel=1e-9)
    # MIDI input path must not run any DSP: confidence is the flat 1.0 from load_midi.
    assert all(n.confidence == 1.0 for n in seq.notes)


def test_extract_midi_out_writes_additional_midi_file(fixtures_dir, tmp_path):
    wav = fixtures_dir / "mono_scale.wav"
    out_json = tmp_path / "out.json"
    out_midi = tmp_path / "out.mid"

    rc = main(["extract", str(wav), "-o", str(out_json), "--midi-out", str(out_midi)])

    assert rc == 0
    assert out_json.exists()
    assert out_midi.exists()
    assert out_midi.stat().st_size > 0


def test_extract_poly_without_basic_pitch_exits_nonzero_with_stderr(fixtures_dir, tmp_path, capsys):
    try:
        import basic_pitch  # noqa: F401
        pytest.skip("basic-pitch is installed in this environment")
    except ImportError:
        pass

    wav = fixtures_dir / "mono_scale.wav"
    out = tmp_path / "poly.json"

    rc = main(["extract", str(wav), "-o", str(out), "--poly"])

    assert rc == 1
    assert not out.exists()
    captured = capsys.readouterr()
    assert captured.err.strip() != ""


# ---------------------------------------------------------------------------
# reduce
# ---------------------------------------------------------------------------

def test_reduce_stage1_two_voice_thirds_is_playable_and_monophonic(fixtures_dir, tmp_path):
    mid = fixtures_dir / "two_voice_thirds.mid"
    gt_json = tmp_path / "two_voice_thirds_gt.json"
    assert main(["extract", str(mid), "-o", str(gt_json)]) == 0

    rc = main(["reduce", str(gt_json), "--stage", "1"])
    assert rc == 0

    default_out = tmp_path / "two_voice_thirds_gt.stage1.json"
    assert default_out.exists()

    out_seq = NoteSequence.from_json(default_out)
    violations = playability_violations(out_seq, StageConfig.stage(1))
    assert violations == []
    assert _max_concurrent(out_seq.notes) <= 1


def test_reduce_stage3_triple_rolled_keeps_triple_and_marks_rolled(fixtures_dir, tmp_path):
    mid = fixtures_dir / "triple_rolled.mid"
    gt_json = tmp_path / "triple_gt.json"
    assert main(["extract", str(mid), "-o", str(gt_json)]) == 0

    out_json = tmp_path / "triple_stage3.json"
    rc = main(["reduce", str(gt_json), "--stage", "3", "-o", str(out_json)])
    assert rc == 0

    out_seq = NoteSequence.from_json(out_json)
    triple = [n for n in out_seq.notes if abs(n.onset_s - 0.6) < 1e-6]
    assert len(triple) == 3
    assert all(n.rolled for n in triple)
    assert playability_violations(out_seq, StageConfig.stage(3)) == []


def test_reduce_missing_input_returns_nonzero_with_stderr(tmp_path, capsys):
    missing = tmp_path / "does_not_exist.json"

    rc = main(["reduce", str(missing), "--stage", "1"])

    assert rc != 0
    captured = capsys.readouterr()
    assert captured.err.strip() != ""


# ---------------------------------------------------------------------------
# render
# ---------------------------------------------------------------------------

def test_render_writes_nonzero_wav(fixtures_dir, tmp_path):
    mid = fixtures_dir / "mono_scale.mid"
    gt_json = tmp_path / "mono_scale_gt.json"
    assert main(["extract", str(mid), "-o", str(gt_json)]) == 0

    out_wav = tmp_path / "mono_scale.render.wav"
    rc = main(["render", str(gt_json), "-o", str(out_wav)])

    assert rc == 0
    assert out_wav.exists()
    sr, data = wavfile.read(str(out_wav))
    assert sr > 0
    assert len(data) > 0
    assert np.any(data.astype(np.float64) != 0.0)


def test_render_default_output_path(fixtures_dir, tmp_path):
    mid = fixtures_dir / "mono_scale.mid"
    gt_json = tmp_path / "mono_scale_gt.json"
    assert main(["extract", str(mid), "-o", str(gt_json)]) == 0

    rc = main(["render", str(gt_json)])

    assert rc == 0
    expected = tmp_path / "mono_scale_gt.render.wav"
    assert expected.exists()


def test_render_paired_writes_both_files(fixtures_dir, tmp_path):
    mid = fixtures_dir / "mono_scale.mid"
    gt_json = tmp_path / "mono_scale_gt.json"
    assert main(["extract", str(mid), "-o", str(gt_json)]) == 0

    out_dir = tmp_path / "paired"
    original_wav = fixtures_dir / "mono_scale.wav"
    rc = main(["render", str(gt_json), "--paired", str(original_wav), "--out-dir", str(out_dir)])

    assert rc == 0
    original_out = out_dir / "original.wav"
    render_out = out_dir / "extracted_render.wav"
    assert original_out.exists()
    assert render_out.exists()

    _, orig_data = wavfile.read(str(original_out))
    _, render_data = wavfile.read(str(render_out))
    assert len(orig_data) > 0
    assert np.any(render_data.astype(np.float64) != 0.0)


def test_render_paired_without_out_dir_errors(fixtures_dir, tmp_path, capsys):
    mid = fixtures_dir / "mono_scale.mid"
    gt_json = tmp_path / "mono_scale_gt.json"
    assert main(["extract", str(mid), "-o", str(gt_json)]) == 0

    rc = main(["render", str(gt_json), "--paired", str(fixtures_dir / "mono_scale.wav")])

    assert rc != 0
    captured = capsys.readouterr()
    assert "out-dir" in captured.err.lower() or "error" in captured.err.lower()


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------

def test_eval_end_to_end_on_fixture_corpus(fixtures_dir, tmp_path, capsys):
    report_path = tmp_path / "out" / "eval.json"

    rc = main(["eval", "--fixtures", str(fixtures_dir), "--report", str(report_path)])

    assert rc == 0
    assert report_path.exists()

    baseline_path = report_path.parent / "eval_baseline.json"
    assert baseline_path.exists()

    first_run_out = capsys.readouterr()
    assert "NOTE" in first_run_out.out  # baseline-update reminder on first write

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert set(report["meta"]["fixture_names"]) == {
        "mono_arpeggio", "mono_scale", "triple_rolled", "two_voice_thirds",
    }

    mono_seen = 0
    for name, entry in report["fixtures"].items():
        if entry["monophonic"]:
            mono_seen += 1
            assert entry["melody"]["raw_pitch_accuracy"] > 0.90, name
            assert "onset_f1" in entry["transcription"]
        assert entry["determinism"]["equal"] is True, name
        for stage_key, stage_entry in entry["reducer"].items():
            assert stage_entry["violation_count"] == 0, (name, stage_key)
            assert stage_entry["violations"] == []

    assert mono_seen == 2  # mono_scale, mono_arpeggio
    assert report["aggregate"]["mono_fixture_count"] == 2
    assert report["aggregate"]["reducer_violation_total"] == 0
    assert report["aggregate"]["determinism_all_equal"] is True

    # Report JSON must not leak absolute paths.
    report_text = report_path.read_text(encoding="utf-8")
    assert str(fixtures_dir) not in report_text

    # Second run against the same baseline: no regression, exit 0.
    rc2 = main(["eval", "--fixtures", str(fixtures_dir), "--report", str(report_path)])
    assert rc2 == 0
    second_run_out = capsys.readouterr()
    assert "regression" in second_run_out.out.lower()


def test_eval_report_json_is_deterministic_byte_for_byte(fixtures_dir, tmp_path):
    report_a = tmp_path / "a" / "eval.json"
    report_b = tmp_path / "b" / "eval.json"

    assert main(["eval", "--fixtures", str(fixtures_dir), "--report", str(report_a)]) == 0
    assert main(["eval", "--fixtures", str(fixtures_dir), "--report", str(report_b)]) == 0

    assert report_a.read_bytes() == report_b.read_bytes()


def test_eval_skips_fixture_missing_half_with_warning(fixtures_dir, tmp_path, capsys):
    import shutil

    partial_dir = tmp_path / "partial_fixtures"
    partial_dir.mkdir()
    # Copy a full pair plus an orphan .mid with no matching .wav.
    shutil.copyfile(fixtures_dir / "mono_scale.wav", partial_dir / "mono_scale.wav")
    shutil.copyfile(fixtures_dir / "mono_scale.mid", partial_dir / "mono_scale.mid")
    shutil.copyfile(fixtures_dir / "mono_arpeggio.mid", partial_dir / "orphan.mid")

    report_path = tmp_path / "partial_eval.json"
    rc = main(["eval", "--fixtures", str(partial_dir), "--report", str(report_path)])

    assert rc == 0
    captured = capsys.readouterr()
    assert "orphan" in captured.err
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert list(report["fixtures"].keys()) == ["mono_scale"]


def test_eval_missing_fixtures_dir_returns_nonzero(tmp_path, capsys):
    rc = main(["eval", "--fixtures", str(tmp_path / "nope"), "--report", str(tmp_path / "r.json")])

    assert rc != 0
    captured = capsys.readouterr()
    assert captured.err.strip() != ""


# ---------------------------------------------------------------------------
# CLI-level error paths
# ---------------------------------------------------------------------------

def test_unknown_subcommand_returns_nonzero_with_stderr(capsys):
    rc = main(["not-a-real-subcommand"])

    assert rc != 0
    captured = capsys.readouterr()
    assert captured.err.strip() != ""


def test_no_subcommand_returns_nonzero_with_stderr(capsys):
    rc = main([])

    assert rc != 0
    captured = capsys.readouterr()
    assert captured.err.strip() != ""


def test_extract_missing_input_file_returns_nonzero_with_stderr(tmp_path, capsys):
    missing = tmp_path / "does_not_exist.wav"

    rc = main(["extract", str(missing)])

    assert rc != 0
    captured = capsys.readouterr()
    assert "error" in captured.err.lower()


# ---------------------------------------------------------------------------
# --config PATH (config_io presets)
# ---------------------------------------------------------------------------

def _preset(**overrides) -> Preset:
    """A preset with distinguishable-from-default values on request, e.g.
    _preset(mono=MonoConfig(backend="yin"), stage=StageConfig(w_jump=0.9))."""
    return Preset(
        name="test-preset",
        comment="",
        mono=overrides.get("mono", MonoConfig()),
        timbre=overrides.get("timbre", TimbreConfig()),
        stage=overrides.get("stage", StageConfig()),
        render=overrides.get("render", RenderConfig()),
    )


def test_extract_config_applies_preset_mono_and_timbre(fixtures_dir, tmp_path):
    preset_path = tmp_path / "preset.json"
    save_preset(_preset(timbre=TimbreConfig(n_harmonics=4)), preset_path)

    wav = fixtures_dir / "mono_scale.wav"
    out = tmp_path / "out.json"
    rc = main(["extract", str(wav), "-o", str(out), "--config", str(preset_path)])

    assert rc == 0
    seq = NoteSequence.from_json(out)
    assert all(n.harmonics is not None for n in seq.notes)
    assert all(len(n.harmonics.harmonic_amps_db) == 4 for n in seq.notes)


def test_extract_backend_flag_precedence_over_config_preset(fixtures_dir, tmp_path, capsys):
    # Preset requests the (in this environment, uninstalled) crepe backend
    # explicitly -- transcribe_mono(backend="crepe") must raise ImportError
    # when nothing overrides it.
    preset_path = tmp_path / "preset.json"
    save_preset(_preset(mono=MonoConfig(backend="crepe")), preset_path)

    wav = fixtures_dir / "mono_scale.wav"
    out_a = tmp_path / "a.json"
    rc_preset_only = main(["extract", str(wav), "-o", str(out_a), "--config", str(preset_path)])
    assert rc_preset_only == 1
    assert not out_a.exists()
    captured = capsys.readouterr()
    assert "crepe" in captured.err.lower()

    # Explicit --backend must override the preset's backend (flag > preset).
    out_b = tmp_path / "b.json"
    rc_flag_override = main([
        "extract", str(wav), "-o", str(out_b), "--config", str(preset_path), "--backend", "yin",
    ])
    assert rc_flag_override == 0
    assert out_b.exists()


def test_extract_no_config_defaults_unchanged(fixtures_dir, tmp_path):
    wav = fixtures_dir / "mono_scale.wav"
    out = tmp_path / "out.json"

    rc = main(["extract", str(wav), "-o", str(out)])

    assert rc == 0
    seq = NoteSequence.from_json(out)
    assert len(seq.notes) == 8
    # default TimbreConfig.n_harmonics == 8
    assert all(len(n.harmonics.harmonic_amps_db) == 8 for n in seq.notes)


def test_reduce_config_shows_preset_weights_and_overridden_max_voices(fixtures_dir, tmp_path):
    mid = fixtures_dir / "mono_scale.mid"
    gt_json = tmp_path / "mono_scale_gt.json"
    assert main(["extract", str(mid), "-o", str(gt_json)]) == 0

    preset_path = tmp_path / "preset.json"
    save_preset(_preset(stage=StageConfig(w_jump=0.9, w_frag=3.0)), preset_path)

    out_json = tmp_path / "out.json"
    rc = main([
        "reduce", str(gt_json), "--stage", "2", "-o", str(out_json), "--config", str(preset_path),
    ])
    assert rc == 0

    out_seq = NoteSequence.from_json(out_json)
    assert out_seq.meta.stage["max_voices"] == 2       # --stage always overrides preset.stage.max_voices
    assert out_seq.meta.stage["w_jump"] == pytest.approx(0.9)   # preset weight carried through
    assert out_seq.meta.stage["w_frag"] == pytest.approx(3.0)


def test_reduce_without_config_uses_stage_default_weights(fixtures_dir, tmp_path):
    mid = fixtures_dir / "mono_scale.mid"
    gt_json = tmp_path / "mono_scale_gt.json"
    assert main(["extract", str(mid), "-o", str(gt_json)]) == 0

    out_json = tmp_path / "out.json"
    rc = main(["reduce", str(gt_json), "--stage", "1", "-o", str(out_json)])
    assert rc == 0

    out_seq = NoteSequence.from_json(out_json)
    assert out_seq.meta.stage["max_voices"] == 1
    assert out_seq.meta.stage["w_jump"] == pytest.approx(StageConfig().w_jump)


def test_render_config_applies_preset_render_config(fixtures_dir, tmp_path):
    mid = fixtures_dir / "mono_scale.mid"
    gt_json = tmp_path / "mono_scale_gt.json"
    assert main(["extract", str(mid), "-o", str(gt_json)]) == 0

    preset_path = tmp_path / "preset.json"
    save_preset(_preset(render=RenderConfig(n_partials=2)), preset_path)

    out_wav = tmp_path / "out.wav"
    rc = main(["render", str(gt_json), "-o", str(out_wav), "--config", str(preset_path)])

    assert rc == 0
    assert out_wav.exists()
    sr, data = wavfile.read(str(out_wav))
    assert sr > 0
    assert len(data) > 0


def test_eval_config_nondefault_preset_writes_report_but_no_baseline(fixtures_dir, tmp_path, capsys):
    preset_path = tmp_path / "preset.json"
    save_preset(_preset(stage=StageConfig(w_jump=0.9)), preset_path)

    report_path = tmp_path / "out" / "eval.json"
    baseline_path = report_path.parent / "eval_baseline.json"

    rc = main([
        "eval", "--fixtures", str(fixtures_dir), "--report", str(report_path), "--config", str(preset_path),
    ])

    assert rc == 0
    assert report_path.exists()
    assert not baseline_path.exists()

    captured = capsys.readouterr()
    assert "NOTE" in captured.out
    assert "baseline" in captured.out.lower()


def test_eval_config_default_preset_still_writes_baseline(fixtures_dir, tmp_path):
    # A preset whose configs equal the code defaults is NOT "non-default":
    # the guardrail must not block the ordinary first-run baseline write.
    preset_path = tmp_path / "preset.json"
    save_preset(_preset(), preset_path)

    report_path = tmp_path / "out" / "eval.json"
    baseline_path = report_path.parent / "eval_baseline.json"

    rc = main([
        "eval", "--fixtures", str(fixtures_dir), "--report", str(report_path), "--config", str(preset_path),
    ])

    assert rc == 0
    assert baseline_path.exists()


def test_eval_config_stage_configs_sweep_max_voices_1_2_3(fixtures_dir, tmp_path):
    preset_path = tmp_path / "preset.json"
    save_preset(_preset(stage=StageConfig(w_jump=0.9)), preset_path)

    report_path = tmp_path / "eval.json"
    rc = main([
        "eval", "--fixtures", str(fixtures_dir), "--report", str(report_path), "--config", str(preset_path),
    ])
    assert rc == 0

    report = json.loads(report_path.read_text(encoding="utf-8"))
    for entry in report["fixtures"].values():
        assert set(entry["reducer"].keys()) == {"1", "2", "3"}
