"""Tab 2 — Topology comparison dashboard (Concept A vs B, D-028/D-030).

Runs compare over selected profiles × pieces (cached; the tempo-headroom
bisection re-plans each piece ~26×, so expect seconds per pair) or loads the
checked-in decision-governed baseline read-only."""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from motion_planner.gui import figures, pipeline_cache

_PKG_ROOT = Path(__file__).resolve().parents[2]
PROFILES_DIR = _PKG_ROOT / "profiles"
FIXTURES_DIR = _PKG_ROOT / "tests" / "fixtures"
BASELINE_PATH = _PKG_ROOT / "out" / "compare_baseline.json"

_TABLE_COLS = ("profile", "piece", "motor_count", "feasibility_pct", "tempo_headroom",
               "worst_late_s", "vibrato_coverage", "roll_compliance", "onset_f1",
               "rpa", "silenced_notes")


def render(config) -> None:
    profile_files = sorted(PROFILES_DIR.glob("*.json"))
    fixture_files = sorted(FIXTURES_DIR.glob("*.stage*.json"))
    chosen_profiles = st.multiselect(
        "Profiles", [p.name for p in profile_files],
        default=[p.name for p in profile_files], key="cmp_profiles",
        help="HardwareProfile JSONs under MotionPlanner/profiles/. Concept A = roaming "
             "fingers; Concept B = one finger per string (GhostPlay).")
    chosen_inputs = st.multiselect(
        "Pieces", [p.name for p in fixture_files],
        default=[p.name for p in fixture_files], key="cmp_inputs",
        help="Reduced NoteSequence JSONs. Add real songs by planning them in tab 1 "
             "and dropping the JSON into tests/fixtures/ (or extend via CLI --inputs).")

    run = st.button("Run comparison", type="primary",
                    disabled=not (chosen_profiles and chosen_inputs),
                    help="Tempo-headroom bisection re-plans each piece ~26 times per "
                         "profile — seconds per pair, cached afterwards.")
    report_text = None
    if run:
        with st.spinner("Comparing topologies (bisection over tempo)..."):
            report_text = pipeline_cache.compare_from_paths(
                tuple(str(PROFILES_DIR / n) for n in chosen_profiles),
                tuple(str(FIXTURES_DIR / n) for n in chosen_inputs),
                json.dumps(config.config_dict(), sort_keys=True))
        st.session_state["cmp_report_text"] = report_text
    report_text = st.session_state.get("cmp_report_text")

    if st.toggle("Show checked-in baseline instead", value=report_text is None,
                 key="cmp_show_baseline",
                 help="out/compare_baseline.json — decision-governed (D-028); the GUI "
                      "never writes it."):
        if BASELINE_PATH.exists():
            report_text = BASELINE_PATH.read_text(encoding="utf-8")
            st.caption(f"read-only baseline: {BASELINE_PATH.name}")
        else:
            st.info("No baseline checked in yet (run the compare CLI once).")

    if not report_text:
        return
    rows = json.loads(report_text).get("rows", [])
    st.plotly_chart(figures.compare_figure(rows), use_container_width=True,
                    key="cmp_chart")
    st.dataframe([{c: r.get(c) for c in _TABLE_COLS} for r in rows],
                 use_container_width=True)
    st.download_button("Download report JSON", data=report_text,
                       file_name="compare_report.json")
