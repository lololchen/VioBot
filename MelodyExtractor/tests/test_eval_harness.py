"""eval_harness contract tests.

The key guard here is behavior-preservation: `run_eval()` with all-default
arguments must be the exact same computation `cmd_eval` used to do inline
before it was extracted into this module (docs/plan_GUI_MelodyExtractor.md).
We regression-test that by comparing run_eval()'s canonical serialization
directly against the report file the CLI's `eval` subcommand writes.
"""
from __future__ import annotations

import json
from dataclasses import replace

import pytest

from melody_extractor.cli import main
from melody_extractor.eval_harness import compare_to_baseline, run_eval
from melody_extractor.reducer import StageConfig


def _canonical_json(report: dict) -> str:
    return json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"


# ---------------------------------------------------------------------------
# refactor guard: byte-equality vs the CLI path
# ---------------------------------------------------------------------------

def test_run_eval_defaults_byte_identical_to_cli_eval_report(fixtures_dir, tmp_path):
    report_path = tmp_path / "eval.json"
    rc = main(["eval", "--fixtures", str(fixtures_dir), "--report", str(report_path)])
    assert rc == 0

    cli_bytes = report_path.read_bytes()

    direct_report = run_eval(fixtures_dir)
    direct_bytes = _canonical_json(direct_report).encode("utf-8")

    assert direct_bytes == cli_bytes


def test_run_eval_is_deterministic_across_two_direct_calls(fixtures_dir):
    report_a = run_eval(fixtures_dir)
    report_b = run_eval(fixtures_dir)

    assert _canonical_json(report_a) == _canonical_json(report_b)


# ---------------------------------------------------------------------------
# parametrization: a modified StageConfig changes only reducer-related entries
# ---------------------------------------------------------------------------

def test_modified_stage_config_changes_only_reducer_entries(fixtures_dir):
    baseline_report = run_eval(fixtures_dir)

    modified_stage_configs = {
        n: replace(StageConfig.stage(n), w_jump=0.9, w_frag=3.0)
        for n in (1, 2, 3)
    }
    modified_report = run_eval(fixtures_dir, stage_configs=modified_stage_configs)

    assert set(baseline_report["fixtures"].keys()) == set(modified_report["fixtures"].keys())

    for name, base_entry in baseline_report["fixtures"].items():
        mod_entry = modified_report["fixtures"][name]

        assert mod_entry["monophonic"] == base_entry["monophonic"]
        assert mod_entry["determinism"] == base_entry["determinism"]
        if base_entry["monophonic"]:
            assert mod_entry["melody"] == base_entry["melody"]
            assert mod_entry["transcription"] == base_entry["transcription"]

        # The reducer section is allowed (not guaranteed) to differ; the point
        # of this test is that nothing *else* moved.
        assert set(mod_entry.keys()) == set(base_entry.keys())

    # Non-reducer aggregate entries (mono transcription/melody metrics) must
    # be untouched by a stage-config-only change.
    for key in (
        "mono_fixture_count", "mean_raw_pitch_accuracy", "mean_raw_chroma_accuracy",
        "mean_overall_accuracy", "mean_voicing_recall", "mean_voicing_false_alarm",
        "mean_onset_f1", "mean_onset_pitch_f1", "mean_onset_offset_pitch_f1",
        "determinism_all_equal",
    ):
        assert modified_report["aggregate"][key] == baseline_report["aggregate"][key], key


def test_run_eval_accepts_a_subset_of_stage_numbers(fixtures_dir):
    report = run_eval(fixtures_dir, stage_configs={2: StageConfig.stage(2)})

    for entry in report["fixtures"].values():
        assert set(entry["reducer"].keys()) == {"2"}


# ---------------------------------------------------------------------------
# progress callback
# ---------------------------------------------------------------------------

def test_run_eval_progress_callback_receives_one_message_per_fixture(fixtures_dir):
    messages = []
    report = run_eval(fixtures_dir, progress=messages.append)

    assert len(messages) == len(report["fixtures"])
    for name in report["fixtures"]:
        assert any(name in m for m in messages)


# ---------------------------------------------------------------------------
# compare_to_baseline
# ---------------------------------------------------------------------------

def test_compare_to_baseline_rows_and_drop_points():
    aggregate = {"mean_raw_pitch_accuracy": 0.90, "mean_raw_chroma_accuracy": 0.95}
    baseline = {"mean_raw_pitch_accuracy": 0.95, "mean_raw_chroma_accuracy": 0.95}

    rows, drop = compare_to_baseline(aggregate, baseline)

    assert drop is not None
    assert drop == pytest.approx(5.0)

    row_map = {key: (b, c) for key, _label, b, c in rows}
    assert row_map["mean_raw_pitch_accuracy"] == (0.95, 0.90)


def test_compare_to_baseline_returns_none_drop_when_values_missing():
    rows, drop = compare_to_baseline({}, {})
    assert drop is None
    assert all(b is None and c is None for _key, _label, b, c in rows)
