"""motion-planner CLI: plan | simulate | roundtrip | compare | bench.

Mirrors melody-extractor's CLI conventions: argparse + stdlib only in core,
--config loads a PlannerPreset (explicit flag > preset > code default),
outputs default to the input stem + suffix, deterministic files everywhere.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from melody_extractor.schema import NoteSequence

from .config_io import PlannerConfig, load_preset
from .planner import plan as run_plan
from .profile_io import load_profile
from .schema import MotionScore


def _load_config(args) -> PlannerConfig:
    if getattr(args, "config", None):
        return load_preset(args.config).planner
    return PlannerConfig()


def _cmd_plan(args) -> int:
    seq = NoteSequence.from_json(Path(args.input))
    profile = load_profile(args.profile)
    config = _load_config(args)
    score, report = run_plan(seq, profile, config, source_path_hint=Path(args.input).name)
    out = Path(args.output) if args.output else Path(args.input).with_suffix(".motion.json")
    score.to_json(out)
    report_path = Path(args.report) if args.report else out.with_suffix(".feasibility.json")
    report.to_json(report_path)
    s = report.summary
    print(f"wrote {out}")
    print(f"wrote {report_path}")
    print(f"feasibility {s['feasibility_pct']:.1f}% | violations {s['n_violations']} "
          f"| worst late {s['worst_late_s'] * 1000:.1f} ms")
    return 0


def _cmd_simulate(args) -> int:
    from .simulate import simulate

    score = MotionScore.from_json(Path(args.input))
    predicted = simulate(score)
    out = Path(args.output) if args.output else Path(args.input).with_suffix(".predicted.json")
    predicted.to_json(out)
    print(f"wrote {out} ({len(predicted.notes)} predicted notes)")
    return 0


def _cmd_roundtrip(args) -> int:
    from .roundtrip import roundtrip

    seq = NoteSequence.from_json(Path(args.target))
    score = MotionScore.from_json(Path(args.score))
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.score).parent / "roundtrip"
    result = roundtrip(seq, score, out_dir=out_dir, render=not args.no_render)
    for key, value in sorted(result.metrics.items()):
        print(f"{key}: {value:.4f}")
    if result.listen_dir is not None:
        print(f"listening pair in {result.listen_dir}")
    return 0


def _cmd_compare(args) -> int:
    from .compare import run_compare, write_markdown

    out = Path(args.out)
    report = run_compare(profile_paths=[Path(p) for p in args.profiles],
                         input_paths=[Path(p) for p in args.inputs],
                         render=not args.no_render)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report.to_json(), encoding="utf-8", newline="\n")
    md = out.with_suffix(".md")
    md.write_text(write_markdown(report), encoding="utf-8", newline="\n")
    print(f"wrote {out}")
    print(f"wrote {md}")
    return 0


def _cmd_bench(args) -> int:
    from .firmware_bridge.bench import run_bench

    return run_bench(args)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="motion-planner",
                                description="Sound2Motion: NoteSequence -> MotionScore + sim/compare")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("plan", help="NoteSequence JSON -> MotionScore + FeasibilityReport")
    sp.add_argument("input", help="reduced NoteSequence .json")
    sp.add_argument("--profile", required=True, help="HardwareProfile .json")
    sp.add_argument("-o", "--output", default=None)
    sp.add_argument("--report", default=None)
    sp.add_argument("--config", default=None, help="PlannerPreset .json")
    sp.set_defaults(func=_cmd_plan)

    ss = sub.add_parser("simulate", help="MotionScore -> predicted NoteSequence")
    ss.add_argument("input", help="MotionScore .json")
    ss.add_argument("-o", "--output", default=None)
    ss.set_defaults(func=_cmd_simulate)

    sr = sub.add_parser("roundtrip", help="target vs predicted: mir_eval + A/B renders")
    sr.add_argument("target", help="target NoteSequence .json")
    sr.add_argument("score", help="MotionScore .json")
    sr.add_argument("--out-dir", default=None)
    sr.add_argument("--no-render", action="store_true")
    sr.set_defaults(func=_cmd_roundtrip)

    sc = sub.add_parser("compare", help="corpus x profiles topology comparison")
    sc.add_argument("--profiles", nargs="+", required=True)
    sc.add_argument("--inputs", nargs="+", required=True)
    sc.add_argument("--out", required=True)
    sc.add_argument("--no-render", action="store_true")
    sc.set_defaults(func=_cmd_compare)

    sb = sub.add_parser("bench", help="single-motor bench utility (firmware bridge)")
    sb.add_argument("--port", default=None, help="COM port; omit for mock transport")
    sb.add_argument("--device", type=int, default=1)
    sb.add_argument("--mode", choices=("position", "speed", "torque"), default="position")
    sb.add_argument("--sine", nargs=2, type=float, metavar=("AMPL", "HZ"), default=None)
    sb.add_argument("--step", type=float, default=None)
    sb.add_argument("--duration", type=float, default=2.0)
    sb.add_argument("--dry-run-log", default=None, help="write the raw byte stream here")
    sb.set_defaults(func=_cmd_bench)
    return p


def main(argv: "list[str] | None" = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
