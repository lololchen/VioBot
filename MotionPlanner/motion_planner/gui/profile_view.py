"""Tab 3 — HardwareProfile editor. Axis-limit widgets are generated from
`dataclasses.fields()` (code and GUI can't drift); everything structural is
editable through a raw-JSON expander. Saves go to profiles/ and NEVER silently
overwrite (house rule)."""
from __future__ import annotations

import json
from dataclasses import fields, is_dataclass, replace
from pathlib import Path

import streamlit as st

from motion_planner.profile_io import (
    load_profile,
    profile_from_dict,
    profile_hash,
    profile_to_dict,
    save_profile,
)

_PKG_ROOT = Path(__file__).resolve().parents[2]
PROFILES_DIR = _PKG_ROOT / "profiles"

_AXIS_HELP = {
    "v_max_mps": "v_max — velocity limit (m/s)",
    "a_max_mps2": "a_max — acceleration limit (m/s²); enters T = 2√(d/a) for short moves",
    "v_max_radps": "ω_max — angular velocity limit (rad/s)",
    "a_max_radps2": "α_max — angular acceleration limit (rad/s²)",
    "range_m": "travel range (m)",
    "range_rad": "sweep range (rad)",
    "travel_m": "travel range (m)",
    "f_max_n": "F_max — force limit (N)",
    "df_dt_max_nps": "dF/dt — force slew limit (N/s), shapes attacks",
    "t_press_s": "press ramp time (s)",
    "t_lift_s": "lift ramp time (s)",
    "bandwidth_hz": "small-signal bandwidth (Hz); vibrato depth rolls off beyond it",
}


def _axis_editor(label: str, obj, key_prefix: str):
    """Numeric widgets for every float field of a flat axis dataclass."""
    st.markdown(f"**{label}**")
    values = {}
    cols = st.columns(min(len(fields(obj)), 4))
    for i, f in enumerate(fields(obj)):
        current = getattr(obj, f.name)
        if not isinstance(current, (int, float)) or isinstance(current, bool):
            values[f.name] = current
            continue
        with cols[i % len(cols)]:
            values[f.name] = st.number_input(
                f.name, value=float(current), key=f"{key_prefix}_{f.name}",
                help=_AXIS_HELP.get(f.name, f.name), format="%.4f")
    return replace(obj, **values)


def render() -> None:
    profile_files = sorted(PROFILES_DIR.glob("*.json"))
    base_name = st.selectbox("Base profile", [p.name for p in profile_files],
                             key="prof_base")
    profile = load_profile(PROFILES_DIR / base_name)
    st.caption(f"sha256 {profile_hash(profile)[:16]} · concept {profile.topology.concept} · "
               f"{profile.topology.n_fingers} finger(s)")

    st.latex(r"x = L\,(1 - 2^{-s/12}) \qquad \beta_{eff} = \beta_0 \cdot 2^{p/12}"
             r"\qquad T_{move} = \min\!\big(2\sqrt{d/a},\; d/v + v/a\big)")

    bow = profile.bow
    with st.expander("Bow axes", expanded=True):
        belt = _axis_editor("Belt (spinning hair)", bow.belt, "prof_belt")
        y = _axis_editor("Y — touch/leave", bow.y, "prof_y")
        incl = _axis_editor("Z — inclination (string select)", bow.incl, "prof_incl")
        force = _axis_editor("Pressure (differential common mode)", bow.force, "prof_force")
        beta_default = st.number_input(
            "beta_default", value=float(bow.beta_default), key="prof_beta", format="%.3f",
            help="β — bow–bridge distance / sounding length at the OPEN string. Smaller β "
                 "→ louder + brighter but a narrower playable-force wedge; β_eff rises "
                 "with stopped position when the bow is fixed (sul-tasto drift).")
        bow = replace(bow, belt=belt, y=y, incl=incl, force=force, beta_default=beta_default)

    new_fingers = []
    with st.expander("Finger units", expanded=False):
        for i, unit in enumerate(profile.fingers):
            st.divider()
            x = _axis_editor(f"f{i}.x — string select (four-bar)", unit.x, f"prof_f{i}x")
            z = _axis_editor(f"f{i}.z — traverse + vibrato (lead screw)", unit.z, f"prof_f{i}z")
            press = _axis_editor(f"f{i}.press", unit.press, f"prof_f{i}p")
            new_fingers.append(replace(unit, x=x, z=z, press=press))
    profile = replace(profile, bow=bow, fingers=tuple(new_fingers or profile.fingers))

    with st.expander("Raw JSON (topology, strings, bands, ...)", expanded=False):
        raw = st.text_area("Full profile JSON", value=json.dumps(
            profile_to_dict(profile), indent=2, sort_keys=True), height=300,
            key="prof_raw",
            help="Everything, including topology/finger_home_string/band angles. "
                 "Applied on save; widget edits above are already merged in.")
        try:
            profile = profile_from_dict(json.loads(raw))
            st.caption(f"valid · new sha256 {profile_hash(profile)[:16]}")
        except (ValueError, KeyError, json.JSONDecodeError) as e:
            st.error(f"invalid profile: {e}")
            return

    name = st.text_input("Save as (profiles/<name>.json)",
                         value=profile.name + "_edit", key="prof_savename")
    overwrite = st.checkbox("Allow overwriting an existing file", value=False,
                            key="prof_overwrite")
    if st.button("Save profile", type="primary"):
        target = PROFILES_DIR / f"{name}.json"
        if target.exists() and not overwrite:
            st.error(f"{target.name} exists — tick 'Allow overwriting' to replace it.")
        else:
            profile = replace(profile, name=name)
            save_profile(profile.validate(), target)
            st.success(f"wrote {target.name} · sha256 {profile_hash(profile)[:16]}")
