"""CLI: extract | reduce | render | eval (PRD interfaces; argparse, stdlib only).

Subcommands
-----------
extract IN [-o OUT.json] [--poly] [--timbre] [--backend auto|crepe|yin] [--config PATH]
    Audio file -> transcribe (mono default, poly with --poly) -> optional
    timbre pass -> NoteSequence JSON (default OUT = IN stem + ".json").
    MIDI input -> NoteSequence directly (no DSP). Also `--midi-out X.mid`
    to export MIDI alongside JSON. `--config PATH` loads a config_io preset
    (mono + timbre configs); `--backend` still overrides the resulting
    mono config's backend when given explicitly (precedence: flag > preset >
    code default).

reduce IN.json --stage {1,2,3} [-o OUT.json] [--config PATH]
    Load NoteSequence, apply reducer.StageConfig.stage(N) (or, with
    `--config PATH`, the preset's StageConfig with max_voices overridden to
    N), write result.

render IN.json [-o OUT.wav] [--paired ORIGINAL_AUDIO --out-dir DIR] [--config PATH]
    Render via soundsim. --paired writes the loudness-matched pair for
    listening tests instead of a single file. `--config PATH` supplies the
    RenderConfig (default: soundsim.RenderConfig()).

eval --fixtures DIR --report OUT.json [--baseline PATH] [--config PATH]
    For every fixture pair {name}.mid (ground truth) + {name}.wav in DIR:
    extract from the WAV, score against the MIDI with mir_eval, and for
    stage-reduced variants check playability. Writes a JSON report with
    per-fixture and aggregate metrics (algorithm-validation skill):
      - melody: mir_eval.melody RPA, RCA, overall accuracy, voicing
        recall/false-alarm (frame track vs ground-truth f0 at 10 ms hop)
      - transcription: mir_eval.transcription onset F1 (50 ms), onset+pitch
        F1, onset+offset+pitch F1
      - reducer (per stage 1..3): playability violation count (MUST be 0),
        melody retention = fraction of ground-truth top-voice notes whose
        onset+pitch survive reduction (matched at 50 ms / 50 cents)
      - determinism: extraction run twice, sha256 of both JSON outputs equal
    Baseline handling: if --baseline exists, print a comparison table and
    exit non-zero when aggregate melody RPA drops > 1.0 point (regression
    gate); if it does not exist, write it (first run) -- UNLESS a non-default
    `--config` preset is active, in which case the report is still written
    but no baseline is auto-created (a baseline must reflect code-default
    behavior; the CLI prints why and returns 0).

    Melody/transcription metrics are computed only for monophonic fixtures
    (ground-truth MIDI has no time-overlapping notes) -- auto-detected per
    fixture. Reducer metrics run for every fixture at every stage, applied to
    the ground-truth NoteSequence (isolates reducer quality from transcriber
    noise). The report JSON is fully deterministic: no timestamps, no
    absolute paths (fixture/basenames only), sorted keys, LF newline.

    Exit codes: 0 clean; 1 reducer playability violations and/or a
    determinism failure (checked before any baseline comparison); 3 a
    baseline regression (aggregate mono RPA dropped > 1.0 point).

Conventions: argparse only, `main(argv=None) -> int`, subcommand functions
`cmd_extract(args)` etc. return exit codes; all file writes go through
schema.NoteSequence.to_json for determinism; print concise progress to
stdout, errors to stderr. No RNG anywhere.

`--config PATH` (all four subcommands) loads a config_io.Preset (see
config_io.py / presets/README.md). Precedence is always: explicit CLI flag >
preset value > code default. Computation itself is delegated to
`eval_harness` for `eval`; this module owns only arg parsing, report/baseline
writing, and exit codes for that subcommand. Per docs/plan_GUI_MelodyExtractor.md,
this module may import only `config_io` and `eval_harness` from the new
(GUI-adjacent) modules.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from . import config_io, eval_harness, input_adapter, reducer, soundsim, timbre, transcriber
from .schema import NoteSequence

# ---------------------------------------------------------------------------
# extract
# ---------------------------------------------------------------------------

def cmd_extract(args) -> int:
    input_path = Path(args.input)
    try:
        if not input_path.exists():
            raise FileNotFoundError(f"input file not found: {input_path}")

        preset = config_io.load_preset(args.config) if args.config else None

        ext = input_path.suffix.lower()
        if ext in input_adapter.MIDI_EXTENSIONS:
            seq = input_adapter.load_midi(input_path)
        elif ext in input_adapter.AUDIO_EXTENSIONS:
            audio = input_adapter.load_audio(input_path)
            if args.poly:
                seq = transcriber.transcribe_poly(audio)
            else:
                mono_config = preset.mono if preset else transcriber.MonoConfig()
                if args.backend is not None:
                    mono_config = replace(mono_config, backend=args.backend)
                seq = transcriber.transcribe_mono(audio, mono_config)
            if args.timbre:
                timbre_config = preset.timbre if preset else timbre.TimbreConfig()
                seq = timbre.add_harmonics(audio, seq, timbre_config)
        else:
            raise ValueError(f"unsupported input extension: {ext!r} ({input_path})")

        output_path = Path(args.output) if args.output else input_path.with_suffix(".json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        seq.to_json(output_path)
        print(f"wrote {output_path} ({len(seq.notes)} notes)")

        if args.midi_out:
            seq.to_midi(args.midi_out)
            print(f"wrote {args.midi_out}")

        return 0
    except Exception as exc:
        print(f"error: extract failed: {exc}", file=sys.stderr)
        return 1


# ---------------------------------------------------------------------------
# reduce
# ---------------------------------------------------------------------------

def cmd_reduce(args) -> int:
    input_path = Path(args.input)
    try:
        if not input_path.exists():
            raise FileNotFoundError(f"input file not found: {input_path}")

        seq = NoteSequence.from_json(input_path)

        if args.config:
            preset = config_io.load_preset(args.config)
            config = replace(preset.stage, max_voices=args.stage)
        else:
            config = reducer.StageConfig.stage(args.stage)

        out_seq = reducer.reduce(seq, config)

        output_path = (
            Path(args.output) if args.output
            else input_path.with_name(f"{input_path.stem}.stage{args.stage}.json")
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out_seq.to_json(output_path)
        print(f"wrote {output_path} ({len(out_seq.notes)} notes, stage {args.stage})")
        return 0
    except Exception as exc:
        print(f"error: reduce failed: {exc}", file=sys.stderr)
        return 1


# ---------------------------------------------------------------------------
# render
# ---------------------------------------------------------------------------

def cmd_render(args) -> int:
    input_path = Path(args.input)
    try:
        if not input_path.exists():
            raise FileNotFoundError(f"input file not found: {input_path}")

        seq = NoteSequence.from_json(input_path)
        config = config_io.load_preset(args.config).render if args.config else soundsim.RenderConfig()

        if args.paired:
            if not args.out_dir:
                raise ValueError("--paired requires --out-dir")
            original_path = Path(args.paired)
            if not original_path.exists():
                raise FileNotFoundError(f"original audio file not found: {original_path}")
            original = input_adapter.load_audio(original_path)
            out_dir = Path(args.out_dir)
            orig_out, render_out = soundsim.render_paired(original, seq, out_dir, config)
            print(f"wrote {orig_out}")
            print(f"wrote {render_out}")
            return 0

        output_path = (
            Path(args.output) if args.output
            else input_path.with_name(f"{input_path.stem}.render.wav")
        )
        out_path = soundsim.render(seq, output_path, config)
        print(f"wrote {out_path}")
        return 0
    except Exception as exc:
        print(f"error: render failed: {exc}", file=sys.stderr)
        return 1


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------

def _write_json_report(path: "str | Path", report: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def cmd_eval(args) -> int:
    try:
        fixtures_dir = Path(args.fixtures)
        if not fixtures_dir.is_dir():
            raise FileNotFoundError(f"fixtures directory not found: {fixtures_dir}")

        report_path = Path(args.report)
        baseline_path = Path(args.baseline) if args.baseline else report_path.parent / "eval_baseline.json"

        preset = config_io.load_preset(args.config) if args.config else None

        default_mono = transcriber.MonoConfig()
        default_timbre = timbre.TimbreConfig()
        default_stage_configs = {n: reducer.StageConfig.stage(n) for n in (1, 2, 3)}

        if preset is not None:
            mono_config = preset.mono
            timbre_config = preset.timbre
            stage_configs = {n: replace(preset.stage, max_voices=n) for n in (1, 2, 3)}
        else:
            mono_config = default_mono
            timbre_config = default_timbre
            stage_configs = default_stage_configs

        preset_is_nondefault = preset is not None and (
            mono_config != default_mono
            or timbre_config != default_timbre
            or stage_configs != default_stage_configs
        )

        report = eval_harness.run_eval(
            fixtures_dir,
            mono_config=mono_config,
            timbre_config=timbre_config,
            stage_configs=stage_configs,
            progress=lambda msg: print(msg),
        )
        aggregate = report["aggregate"]

        _write_json_report(report_path, report)
        print(f"wrote report: {report_path}")

        any_violations = aggregate.get("reducer_violation_total", 0) > 0
        all_deterministic = aggregate.get("determinism_all_equal", True)

        if any_violations:
            print("error: reducer playability violations detected (gate requires 0)", file=sys.stderr)
        if not all_deterministic:
            print("error: extraction is not deterministic for at least one fixture", file=sys.stderr)
        if any_violations or not all_deterministic:
            return 1

        if baseline_path.exists():
            baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
            base_agg = baseline.get("aggregate", {})
            rows, drop_points = eval_harness.compare_to_baseline(aggregate, base_agg)

            print("baseline comparison (aggregate mono metrics):")
            print(f"  {'metric':<28}{'baseline':>12}{'current':>12}")
            for _key, label, b, c in rows:
                b_s = f"{b:.4f}" if isinstance(b, (int, float)) else "n/a"
                c_s = f"{c:.4f}" if isinstance(c, (int, float)) else "n/a"
                print(f"  {label:<28}{b_s:>12}{c_s:>12}")

            if drop_points is not None and drop_points > 1.0:
                base_rpa = base_agg.get("mean_raw_pitch_accuracy")
                cur_rpa = aggregate.get("mean_raw_pitch_accuracy")
                print(
                    f"REGRESSION: aggregate mono RPA dropped {drop_points:.2f} points "
                    f"(baseline {base_rpa * 100:.2f} -> current {cur_rpa * 100:.2f})",
                    file=sys.stderr,
                )
                return 3
            print("no RPA regression vs baseline.")
            return 0
        else:
            if preset_is_nondefault:
                print(
                    "NOTE: --config preset differs from code defaults; not auto-creating a "
                    "baseline from this run (a baseline must reflect code-default behavior so "
                    "it stays comparable across configs). Report written; run `eval` without "
                    "--config to establish/update the baseline, or pass --baseline to compare "
                    "against an existing one."
                )
                return 0
            _write_json_report(baseline_path, report)
            print(f"wrote new baseline: {baseline_path}")
            print(
                "NOTE: baseline updates require a docs/decisions.md entry "
                "(algorithm-validation skill)."
            )
            return 0
    except Exception as exc:
        print(f"error: eval failed: {exc}", file=sys.stderr)
        return 1


# ---------------------------------------------------------------------------
# argparse wiring
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="melody-extractor",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", metavar="command")

    p_extract = sub.add_parser("extract", help="audio/MIDI -> NoteSequence JSON")
    p_extract.add_argument("input", help="input audio file or MIDI file")
    p_extract.add_argument("-o", "--output", help='output JSON path (default: input stem + ".json")')
    p_extract.add_argument("--poly", action="store_true", help="use polyphonic transcription (transcribe_poly)")
    p_extract.add_argument(
        "--timbre", dest="timbre", action="store_true", default=True,
        help="run the timbre pass (default: on; no-op, kept for symmetry with --no-timbre)",
    )
    p_extract.add_argument(
        "--no-timbre", dest="timbre", action="store_false", help="skip the timbre pass"
    )
    p_extract.add_argument(
        "--backend", choices=["auto", "crepe", "yin"], default=None,
        help="mono transcription backend (default: preset's, else 'auto'); "
             "explicit flag always overrides a --config preset",
    )
    p_extract.add_argument("--midi-out", help="also export a MIDI file to this path")
    p_extract.add_argument("--config", help="config_io preset JSON (mono + timbre configs)")
    p_extract.set_defaults(func=cmd_extract)

    p_reduce = sub.add_parser("reduce", help="reduce a NoteSequence to a playable hardware stage")
    p_reduce.add_argument("input", help="input NoteSequence JSON")
    p_reduce.add_argument("--stage", type=int, choices=[1, 2, 3], required=True, help="hardware stage N")
    p_reduce.add_argument(
        "-o", "--output", help='output JSON path (default: stem + ".stageN.json")'
    )
    p_reduce.add_argument(
        "--config", help="config_io preset JSON (StageConfig; --stage overrides its max_voices)"
    )
    p_reduce.set_defaults(func=cmd_reduce)

    p_render = sub.add_parser("render", help="render a NoteSequence to audio via soundsim")
    p_render.add_argument("input", help="input NoteSequence JSON")
    p_render.add_argument("-o", "--output", help='output WAV path (default: stem + ".render.wav")')
    p_render.add_argument("--paired", metavar="ORIGINAL_AUDIO", help="original audio for a loudness-matched pair")
    p_render.add_argument("--out-dir", help="output directory for --paired (required together with --paired)")
    p_render.add_argument("--config", help="config_io preset JSON (RenderConfig)")
    p_render.set_defaults(func=cmd_render)

    p_eval = sub.add_parser("eval", help="mir_eval evaluation harness over a fixture corpus")
    p_eval.add_argument("--fixtures", required=True, help="directory of {name}.mid/{name}.wav fixture pairs")
    p_eval.add_argument("--report", required=True, help="output report JSON path")
    p_eval.add_argument(
        "--baseline", help="baseline report JSON path (default: <report dir>/eval_baseline.json)"
    )
    p_eval.add_argument(
        "--config",
        help="config_io preset JSON (mono/timbre/stage configs); with a non-default preset "
             "and no existing baseline, the report is written but no baseline is auto-created",
    )
    p_eval.set_defaults(func=cmd_eval)

    return parser


def main(argv=None) -> int:
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        code = exc.code
        return code if isinstance(code, int) else 1

    if getattr(args, "command", None) is None:
        parser.print_usage(sys.stderr)
        print("error: a subcommand is required (extract | reduce | render | eval)", file=sys.stderr)
        return 2

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
