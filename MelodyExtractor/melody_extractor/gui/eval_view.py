"""Tab 2 — Eval Dashboard (plan_GUI_MelodyExtractor.md "Tab 2").

Read-only header (fixtures dir + baseline path); a button runs the mir_eval
harness (`eval_harness.run_eval`, via `pipeline_cache.run_eval_cached`) with
the CURRENT sidebar configs, streaming per-fixture progress into `st.status`;
results are cached on (configs, fixtures-mtime digest). Aggregate metrics are
compared to the baseline with `eval_harness.compare_to_baseline`, flagging a
REGRESSION in the same wording the CLI uses when RPA drops > 1 point.

Hard rule (D-016, module CLAUDE.md): this view NEVER writes
`eval_baseline.json` or a report file to disk -- baseline adoption stays
CLI-governed. The only artifact offered is a `st.download_button` with the
deterministic report bytes (sorted keys, LF, no timestamps/abs paths -- the
same canonical serialization `cli._write_json_report` uses).
"""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from .. import eval_harness
from ..reducer import StageConfig
from ..timbre import TimbreConfig
from ..transcriber import MonoConfig
from . import pipeline_cache

FIXTURES_DIR = Path(__file__).resolve().parents[2] / "tests" / "fixtures"
BASELINE_PATH = Path(__file__).resolve().parents[2] / "out" / "eval_baseline.json"

_MONO_KEYS = (
    "mean_raw_pitch_accuracy", "mean_raw_chroma_accuracy", "mean_overall_accuracy",
    "mean_voicing_recall", "mean_voicing_false_alarm",
)


def _canonical_report_bytes(report: dict) -> bytes:
    return (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def render(mono_cfg: MonoConfig, timbre_cfg: TimbreConfig, stage_cfg: StageConfig) -> None:
    st.subheader("Eval Dashboard")
    st.caption(f"Fixtures directory: `{FIXTURES_DIR}`")
    if BASELINE_PATH.exists():
        st.caption(f"Baseline: `{BASELINE_PATH}`")
    else:
        st.caption(f"Baseline: `{BASELINE_PATH}` (not found)")

    st.caption(
        "Knob -> metric mapping: mono/timbre configs affect the transcription/melody "
        "metrics (computed against the transcriber's frame track and notes); StageConfig "
        "(reducer weights, stage) acts on the *ground-truth* MIDI notes directly, so it "
        "only ever moves the reducer/melody-retention metrics — never the mono "
        "transcription metrics — isolating reducer quality from transcriber noise."
    )

    run_clicked = st.button("Run eval (eval_harness.run_eval)", key="eval_run_btn")

    if run_clicked:
        if not FIXTURES_DIR.is_dir():
            st.error(f"fixtures directory not found: {FIXTURES_DIR}")
        else:
            mtime_digest = pipeline_cache.fixtures_mtime_digest(FIXTURES_DIR)
            status_box = st.status("Running eval harness...", expanded=True)

            def _progress(msg: str) -> None:
                status_box.write(msg)

            report = pipeline_cache.run_eval_cached(
                str(FIXTURES_DIR), mtime_digest, mono_cfg, timbre_cfg, stage_cfg, _progress=_progress,
            )
            status_box.update(label="Eval complete", state="complete")
            st.session_state["_eval_last_report"] = report

    report = st.session_state.get("_eval_last_report")
    if report is None:
        st.info("Run the eval harness (button above) to see results.")
        return

    aggregate = report["aggregate"]

    st.markdown("#### Aggregate mono metrics vs. baseline")
    baseline_aggregate = None
    if BASELINE_PATH.exists():
        try:
            baseline_aggregate = json.loads(BASELINE_PATH.read_text(encoding="utf-8")).get("aggregate", {})
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            st.warning(f"Could not read baseline: {exc}")

    if baseline_aggregate is not None:
        rows, rpa_drop = eval_harness.compare_to_baseline(aggregate, baseline_aggregate)
        table = []
        for _key, label, base_val, cur_val in rows:
            has_both = isinstance(base_val, (int, float)) and isinstance(cur_val, (int, float))
            delta = (cur_val - base_val) if has_both else None
            table.append({
                "metric": label,
                "baseline": f"{base_val:.4f}" if isinstance(base_val, (int, float)) else "n/a",
                "current": f"{cur_val:.4f}" if isinstance(cur_val, (int, float)) else "n/a",
                "delta": f"{delta:+.4f}" if delta is not None else "n/a",
            })
        st.dataframe(table, use_container_width=True)

        if rpa_drop is not None and rpa_drop > 1.0:
            base_rpa = baseline_aggregate.get("mean_raw_pitch_accuracy")
            cur_rpa = aggregate.get("mean_raw_pitch_accuracy")
            st.error(
                f"REGRESSION: aggregate mono RPA dropped {rpa_drop:.2f} points "
                f"(baseline {base_rpa * 100:.2f} -> current {cur_rpa * 100:.2f})"
            )
        else:
            st.success("no RPA regression vs baseline.")
    else:
        st.warning("No baseline found — showing the current run only.")
        st.json({k: aggregate.get(k) for k in _MONO_KEYS})

    st.markdown("#### Per-fixture")
    fixture_rows = []
    for name, entry in sorted(report["fixtures"].items()):
        melody = entry.get("melody")
        reducer_entry = entry.get("reducer", {})
        fixture_rows.append({
            "fixture": name,
            "monophonic": entry["monophonic"],
            "RPA": f"{melody['raw_pitch_accuracy']:.4f}" if melody else "—",
            "RCA": f"{melody['raw_chroma_accuracy']:.4f}" if melody else "—",
            "violations (stage 1/2/3)": "/".join(
                str(reducer_entry[s]["violation_count"]) if s in reducer_entry else "—"
                for s in ("1", "2", "3")
            ),
            "determinism_equal": entry["determinism"]["equal"],
        })
    st.dataframe(fixture_rows, use_container_width=True)
    st.caption('Poly (non-monophonic) fixtures show "—" for RPA/RCA: melody/transcription '
               "metrics are computed for monophonic ground truth only.")

    st.download_button(
        "Download report JSON",
        data=_canonical_report_bytes(report),
        file_name="eval_report.json",
        mime="application/json",
        key="eval_download_btn",
    )
    st.caption(
        "The GUI never writes eval_baseline.json or report files to disk (D-016). To adopt "
        "these settings: save them as a preset (sidebar), run "
        "`melody-extractor eval --config <preset.json>` from the CLI, and record the change "
        "in docs/decisions.md (algorithm-validation skill) before treating a new run as the baseline."
    )
