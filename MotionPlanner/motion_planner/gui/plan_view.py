"""Tab 1 — Plan & Inspect: input → plan → feasibility → charts → A/B audio.

Input sources (precedence = explicit choice, no magic): the gui_hub workspace
("stream mode" — whatever MelodyExtractor last exported), the module's own
fixtures, or an uploaded reduced NoteSequence JSON. Outputs register back into
the workspace so the Firmware tab can build byte logs (D-030)."""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from motion_planner.gui import figures, pipeline_cache
from motion_planner.profile_io import profile_canonical_text
from motion_planner.schema import FeasibilityReport, MotionScore

_PKG_ROOT = Path(__file__).resolve().parents[2]          # MotionPlanner/
FIXTURES_DIR = _PKG_ROOT / "tests" / "fixtures"


def _workspace():
    try:
        import sys

        repo_root = _PKG_ROOT.parent
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from gui_hub import workspace

        return workspace
    except ImportError:
        return None


def _pick_input() -> "tuple[str, str] | None":
    """Returns (seq_json_text, path_hint) or None."""
    ws = _workspace()
    options = []
    if ws is not None and ws.latest("note_sequence") is not None:
        options.append("workspace latest (MelodyExtractor export)")
    fixture_files = sorted(FIXTURES_DIR.glob("*.stage*.json"))
    options += [f"fixture: {p.name}" for p in fixture_files]
    options.append("upload a NoteSequence JSON")
    choice = st.selectbox("Input NoteSequence", options, key="plan_input_choice",
                          help="A *reduced* NoteSequence (melody-extractor reduce --stage N). "
                               "The workspace entry appears once the MelodyExtractor tab "
                               "exported one.")
    if choice.startswith("workspace"):
        path = ws.latest("note_sequence")
        st.caption(f"workspace: {ws.describe('note_sequence')}")
        return path.read_text(encoding="utf-8"), path.name
    if choice.startswith("fixture: "):
        path = FIXTURES_DIR / choice.split(": ", 1)[1]
        return path.read_text(encoding="utf-8"), path.name
    uploaded = st.file_uploader("Reduced NoteSequence .json", type=["json"],
                                key="plan_uploader")
    if uploaded is None:
        return None
    return uploaded.getvalue().decode("utf-8"), uploaded.name


def render(profile, config) -> None:
    picked = _pick_input()
    if picked is None:
        st.info("Choose or upload a reduced NoteSequence to plan.")
        return
    seq_text, hint = picked

    profile_text = profile_canonical_text(profile)
    config_json = json.dumps(config.config_dict(), sort_keys=True)
    with st.status("Planning (fingering → bowing → vibrato → trajectory)...") as status:
        score_text, report_text = pipeline_cache.plan_from_texts(
            seq_text, profile_text, config_json, hint)
        status.update(label="Plan ready", state="complete")
    score = MotionScore.from_json(score_text)
    report = FeasibilityReport.from_json_dict(json.loads(report_text))

    s = report.summary
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Feasibility", f"{s['feasibility_pct']:.1f}%")
    c2.metric("Violations", int(s["n_violations"]))
    c3.metric("Worst lateness", f"{s['worst_late_s'] * 1000:.0f} ms")
    c4.metric("Notes", int(s["n_notes"]))

    if report.violations:
        with st.expander(f"Violations ({len(report.violations)})", expanded=False):
            st.dataframe(list(report.violations), use_container_width=True)

    st.plotly_chart(figures.fingerboard_timeline(score, profile),
                    use_container_width=True, key="plan_fingerboard")
    st.plotly_chart(figures.bow_tracks_figure(score),
                    use_container_width=True, key="plan_bowtracks")

    with st.expander("Mechanism motion (top view, animated)", expanded=False):
        st.plotly_chart(figures.mechanism_animation(score, profile),
                        use_container_width=True, key="plan_mechanism")

    st.subheader("A/B audio — target vs simulated execution")
    predicted_text = pipeline_cache.simulate_text(score_text)
    col_a, col_b = st.columns(2)
    with col_a:
        st.caption("Target (reduced NoteSequence, additive render)")
        st.audio(pipeline_cache.render_wav_bytes(seq_text), format="audio/wav")
    with col_b:
        st.caption("Predicted (forward sim of the MotionScore)")
        predicted_notes = json.loads(predicted_text).get("notes", [])
        if predicted_notes:
            st.audio(pipeline_cache.render_wav_bytes(predicted_text), format="audio/wav")
        else:
            st.warning("Simulation predicts silence (check violations).")

    st.divider()
    d1, d2, d3 = st.columns(3)
    d1.download_button("Download MotionScore", data=score_text,
                       file_name=Path(hint).stem + ".motion.json")
    d2.download_button("Download FeasibilityReport", data=report_text,
                       file_name=Path(hint).stem + ".feasibility.json")
    ws = _workspace()
    if ws is not None and d3.button("Export to workspace → Firmware tab"):
        ws.register_text("motion_score", Path(hint).stem + ".motion.json",
                         score_text, producer="sound2motion-gui")
        ws.register_text("feasibility_report", Path(hint).stem + ".feasibility.json",
                         report_text, producer="sound2motion-gui")
        st.success("Registered motion_score + feasibility_report in the workspace.")
