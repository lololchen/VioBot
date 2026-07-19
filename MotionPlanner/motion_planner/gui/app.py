"""Entry point: sidebar (profile + planner params), three tabs.

Runnable three ways (D-030): `streamlit run app.py`, the gui_hub launcher, or
bare `python app.py` (VS Code Run) which self-launches on the registry port —
or just opens a tab when the hub is already serving."""
from __future__ import annotations

import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent                 # .../motion_planner/gui
_PKG_ROOT = _THIS_DIR.parents[1]                            # MotionPlanner/
_REPO_ROOT = _PKG_ROOT.parent                               # repo root (gui_hub/)
for extra in (_PKG_ROOT, _REPO_ROOT):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import streamlit as st  # noqa: E402

from motion_planner.gui import compare_view, params, plan_view, profile_view, style  # noqa: E402
from motion_planner.profile_io import load_profile  # noqa: E402

PROFILES_DIR = _PKG_ROOT / "profiles"


def main() -> None:
    st.set_page_config(page_title="Sound2Motion — Motion Planner", layout="wide")
    style.inject()
    st.title("Sound2Motion — Motion Planner")

    st.sidebar.header("Hardware profile")
    profile_files = sorted(PROFILES_DIR.glob("*.json"))
    profile_name = st.sidebar.selectbox(
        "Profile", [p.name for p in profile_files], key="sidebar_profile",
        help="The parametric hardware model to plan against (PRD): Concept A = "
             "roaming 3-DoF fingers, Concept B = one finger per string. Edit or "
             "add profiles in the Profile Editor tab.")
    profile = load_profile(PROFILES_DIR / profile_name)
    st.sidebar.caption(f"concept {profile.topology.concept} · "
                       f"{profile.topology.n_fingers} finger(s)")

    st.sidebar.header("Planner parameters")
    config = params.build_planner_config()

    tab_plan, tab_compare, tab_profile = st.tabs(
        ["Plan & Inspect", "Topology Compare", "Profile Editor"])
    with tab_plan:
        plan_view.render(profile, config)
    with tab_compare:
        compare_view.render(config)
    with tab_profile:
        profile_view.render()


if st.runtime.exists():
    main()
else:
    # Bare `python app.py` (VS Code Run): no Streamlit runtime — hand off to
    # the registry-aware launcher (opens a tab if the hub already serves us).
    from motion_planner.gui import launch

    launch.main()
