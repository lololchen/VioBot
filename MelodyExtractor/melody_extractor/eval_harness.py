"""eval_harness — the mir_eval evaluation harness over a fixture corpus.

Behavior-preserving extraction OUT of cli.py (docs/plan_GUI_MelodyExtractor.md)
so both the CLI `eval` subcommand and the GUI Eval Dashboard can drive the
same computation without any argparse / report-writing / baseline / exit-code
plumbing, none of which belongs here (that stays CLI-owned in cli.cmd_eval).

`run_eval(...)` with all-default arguments must produce a report dict whose
canonical JSON serialization (sort_keys, indent=2, allow_nan=False, +"\n") is
byte-identical to what `cmd_eval` produced before this refactor —
tests/test_eval_harness.py regression-tests this directly.

Metrics computed (see cli.py's `eval` docstring / module CLAUDE.md for the
full contract):
  - melody: mir_eval.melody RPA/RCA/OA/voicing recall & false alarm (frame
    track vs ground-truth f0 at a fixed hop), monophonic fixtures only.
  - transcription: mir_eval.transcription onset/onset+pitch/onset+offset+pitch
    F1, monophonic fixtures only.
  - reducer (per configured stage): playability violation count, melody
    retention (fraction of ground-truth top-voice notes whose onset+pitch
    survive reduction), applied to the ground-truth NoteSequence directly
    (isolates reducer quality from transcriber noise).
  - determinism: extraction run twice per fixture, sha256 of both JSON
    outputs compared.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Callable, Optional

import mir_eval.melody
import mir_eval.transcription
import numpy as np

from . import input_adapter, reducer, timbre, transcriber
from .reducer import StageConfig
from .schema import NoteSequence
from .timbre import TimbreConfig
from .transcriber import MonoConfig

_ONSET_TOLERANCE_S = 0.05
_CENTS_TOLERANCE = 50.0
_GT_FRAME_HOP_S = 0.01


# ---------------------------------------------------------------------------
# ground-truth / metric helpers (moved verbatim from cli.py)
# ---------------------------------------------------------------------------

def _is_monophonic(notes) -> bool:
    """True iff no two notes in `notes` overlap in time."""
    ordered = sorted(notes, key=lambda n: n.onset_s)
    for a, b in zip(ordered, ordered[1:]):
        if b.onset_s < a.offset_s - 1e-9:
            return False
    return True


def _gt_frame_f0(notes, hop_s: float = _GT_FRAME_HOP_S):
    """Sample ground-truth notes on a fixed hop grid: f0 of the sounding note,
    0.0 when silent. `notes` must be non-overlapping (monophonic)."""
    if not notes:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
    total = max(n.offset_s for n in notes)
    n_frames = int(math.floor(total / hop_s)) + 1
    times = np.arange(n_frames, dtype=np.float64) * hop_s
    freqs = np.zeros(n_frames, dtype=np.float64)
    ordered = sorted(notes, key=lambda n: n.onset_s)
    for i, t in enumerate(times):
        for note in ordered:
            if note.onset_s <= t < note.offset_s:
                freqs[i] = note.pitch_hz
                break
    return times, freqs


def _to_intervals_pitches(notes):
    if not notes:
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0,), dtype=np.float64)
    intervals = np.array([[n.onset_s, n.offset_s] for n in notes], dtype=np.float64)
    pitches = np.array([n.pitch_hz for n in notes], dtype=np.float64)
    return intervals, pitches


def _top_voice_notes(notes):
    """GT top voice: a note that, at its own onset, is the highest-pitched
    note sounding (ties broken by inclusion -- a note is always active at its
    own onset, so it is trivially its own candidate)."""
    top = []
    for note in notes:
        active = [n.pitch_hz for n in notes if n.onset_s <= note.onset_s < n.offset_s]
        if active and note.pitch_hz >= max(active) - 1e-9:
            top.append(note)
    return top


def _cents(hz_a: float, hz_b: float) -> float:
    return 1200.0 * math.log2(hz_a / hz_b)


def _melody_retention(gt_notes, reduced_notes):
    """Fraction of GT top-voice notes whose onset (+-50ms) and pitch (+-50
    cents) survive in `reduced_notes`. Returns (retention, n_top, n_survived)."""
    top = _top_voice_notes(gt_notes)
    if not top:
        return 1.0, 0, 0
    survived = 0
    for tn in top:
        match = any(
            abs(rn.onset_s - tn.onset_s) <= _ONSET_TOLERANCE_S
            and abs(_cents(rn.pitch_hz, tn.pitch_hz)) <= _CENTS_TOLERANCE
            for rn in reduced_notes
        )
        if match:
            survived += 1
    return survived / len(top), len(top), survived


def _mean(values):
    return float(sum(values) / len(values)) if values else None


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def run_eval(
    fixtures_dir: "str | Path",
    mono_config: MonoConfig = MonoConfig(),
    timbre_config: TimbreConfig = TimbreConfig(),
    stage_configs: "Optional[dict[int, StageConfig]]" = None,
    progress: "Optional[Callable[[str], None]]" = None,
) -> dict:
    """Evaluate every {name}.mid/{name}.wav fixture pair in `fixtures_dir`.

    Returns a report dict `{"fixtures", "aggregate", "meta"}` — the same
    shape `cmd_eval` used to build and write directly, minus any I/O
    (writing the report/baseline, exit codes, and stdout summaries are the
    CLI's job; see `compare_to_baseline` + `cli.cmd_eval`).

    `stage_configs` defaults to `{1: StageConfig.stage(1), 2: StageConfig.stage(2),
    3: StageConfig.stage(3)}`; pass a different mapping to evaluate different
    (or differently-weighted) reducer configs per stage number.

    `progress`, if given, is called with a short status string per fixture
    (e.g. for a GUI `st.status` stream); it never affects the report content.
    """
    if stage_configs is None:
        stage_configs = {n: StageConfig.stage(n) for n in (1, 2, 3)}
    if progress is None:
        def progress(_msg: str) -> None:
            return None

    fixtures_dir = Path(fixtures_dir)
    mid_files = {p.stem: p for p in fixtures_dir.glob("*.mid")}
    wav_files = {p.stem: p for p in fixtures_dir.glob("*.wav")}
    names = sorted(set(mid_files) | set(wav_files))

    fixture_reports: dict = {}
    mono_accum: dict = {
        "rpa": [], "rca": [], "oa": [], "vr": [], "vfa": [],
        "onset_f1": [], "onset_pitch_f1": [], "onset_offset_pitch_f1": [],
    }

    for name in names:
        if name not in mid_files or name not in wav_files:
            print(f"warning: skipping fixture {name!r}: missing .mid or .wav half", file=sys.stderr)
            continue

        mid_path, wav_path = mid_files[name], wav_files[name]
        progress(f"eval: {name}")

        gt_seq: NoteSequence = input_adapter.load_midi(mid_path)
        audio = input_adapter.load_audio(wav_path)

        run1 = timbre.add_harmonics(audio, transcriber.transcribe_mono(audio, mono_config), timbre_config)
        run2 = timbre.add_harmonics(audio, transcriber.transcribe_mono(audio, mono_config), timbre_config)
        json1, json2 = run1.to_json(), run2.to_json()
        sha1 = _sha256_hex(json1)
        sha2 = _sha256_hex(json2)
        equal = json1 == json2

        monophonic = _is_monophonic(gt_seq.notes)
        entry: dict = {
            "monophonic": monophonic,
            "determinism": {"sha256_run1": sha1, "sha256_run2": sha2, "equal": equal},
            "reducer": {},
        }

        if monophonic:
            ref_time, ref_freq = _gt_frame_f0(gt_seq.notes)
            track = run1.features[0]
            est_time = np.asarray(track.times_s(), dtype=np.float64)
            est_freq = np.asarray(track.f0_hz, dtype=np.float64)
            melody_scores = mir_eval.melody.evaluate(ref_time, ref_freq, est_time, est_freq)

            ref_intervals, ref_pitches = _to_intervals_pitches(gt_seq.notes)
            est_intervals, est_pitches = _to_intervals_pitches(run1.notes)
            trans_scores = mir_eval.transcription.evaluate(
                ref_intervals, ref_pitches, est_intervals, est_pitches
            )

            melody_metrics = {
                "raw_pitch_accuracy": float(melody_scores["Raw Pitch Accuracy"]),
                "raw_chroma_accuracy": float(melody_scores["Raw Chroma Accuracy"]),
                "overall_accuracy": float(melody_scores["Overall Accuracy"]),
                "voicing_recall": float(melody_scores["Voicing Recall"]),
                "voicing_false_alarm": float(melody_scores["Voicing False Alarm"]),
            }
            transcription_metrics = {
                "onset_f1": float(trans_scores["Onset_F-measure"]),
                "onset_pitch_f1": float(trans_scores["F-measure_no_offset"]),
                "onset_offset_pitch_f1": float(trans_scores["F-measure"]),
            }
            entry["melody"] = melody_metrics
            entry["transcription"] = transcription_metrics

            mono_accum["rpa"].append(melody_metrics["raw_pitch_accuracy"])
            mono_accum["rca"].append(melody_metrics["raw_chroma_accuracy"])
            mono_accum["oa"].append(melody_metrics["overall_accuracy"])
            mono_accum["vr"].append(melody_metrics["voicing_recall"])
            mono_accum["vfa"].append(melody_metrics["voicing_false_alarm"])
            mono_accum["onset_f1"].append(transcription_metrics["onset_f1"])
            mono_accum["onset_pitch_f1"].append(transcription_metrics["onset_pitch_f1"])
            mono_accum["onset_offset_pitch_f1"].append(transcription_metrics["onset_offset_pitch_f1"])

        for stage in sorted(stage_configs.keys()):
            cfg = stage_configs[stage]
            reduced = reducer.reduce(gt_seq, cfg)
            violations = reducer.playability_violations(reduced, cfg)
            retention, n_top, n_survived = _melody_retention(gt_seq.notes, reduced.notes)
            entry["reducer"][str(stage)] = {
                "violation_count": len(violations),
                "violations": list(violations),
                "melody_retention": retention,
                "top_voice_notes": n_top,
                "top_voice_notes_survived": n_survived,
                "notes_before": len(gt_seq.notes),
                "notes_after": len(reduced.notes),
            }

        fixture_reports[name] = entry

    aggregate = {
        "mono_fixture_count": len(mono_accum["rpa"]),
        "mean_raw_pitch_accuracy": _mean(mono_accum["rpa"]),
        "mean_raw_chroma_accuracy": _mean(mono_accum["rca"]),
        "mean_overall_accuracy": _mean(mono_accum["oa"]),
        "mean_voicing_recall": _mean(mono_accum["vr"]),
        "mean_voicing_false_alarm": _mean(mono_accum["vfa"]),
        "mean_onset_f1": _mean(mono_accum["onset_f1"]),
        "mean_onset_pitch_f1": _mean(mono_accum["onset_pitch_f1"]),
        "mean_onset_offset_pitch_f1": _mean(mono_accum["onset_offset_pitch_f1"]),
        "reducer_violation_total": sum(
            stage_entry["violation_count"]
            for fx in fixture_reports.values()
            for stage_entry in fx["reducer"].values()
        ),
        "determinism_all_equal": all(fx["determinism"]["equal"] for fx in fixture_reports.values()),
    }

    return {
        "fixtures": fixture_reports,
        "aggregate": aggregate,
        "meta": {
            "fixtures_dir": fixtures_dir.name,
            "fixture_names": sorted(fixture_reports.keys()),
        },
    }


def compare_to_baseline(aggregate: dict, baseline_aggregate: dict):
    """Compare current vs. baseline aggregate mono metrics; shared by the CLI
    (`cmd_eval`'s baseline table + regression gate) and the GUI Eval
    Dashboard.

    Returns `(rows, rpa_drop_points)`:
      - `rows`: `[(key, label, baseline_value, current_value), ...]` for the
        five mono metrics both surfaces display (values are raw floats or
        `None` when absent -- formatting is the caller's job).
      - `rpa_drop_points`: `(baseline_rpa - current_rpa) * 100`, or `None`
        when either side is missing/non-numeric (no regression gate applies
        then). The CLI's regression gate fires when this exceeds 1.0.
    """
    labels = (
        ("mean_raw_pitch_accuracy", "RPA"),
        ("mean_raw_chroma_accuracy", "RCA"),
        ("mean_overall_accuracy", "OA"),
        ("mean_voicing_recall", "Voicing Recall"),
        ("mean_voicing_false_alarm", "Voicing False Alarm"),
    )
    rows = [(key, label, baseline_aggregate.get(key), aggregate.get(key)) for key, label in labels]

    base_rpa = baseline_aggregate.get("mean_raw_pitch_accuracy")
    cur_rpa = aggregate.get("mean_raw_pitch_accuracy")
    rpa_drop = None
    if isinstance(base_rpa, (int, float)) and isinstance(cur_rpa, (int, float)):
        rpa_drop = (base_rpa - cur_rpa) * 100.0
    return rows, rpa_drop


def _sha256_hex(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
