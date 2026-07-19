"""Sidebar parameter widgets — PlannerConfig from the UI (D-021 tooltip style).

Every widget gets a `help=` tooltip (Unicode Greek + inline `$...$` KaTeX) —
documentation only, NEVER a source of defaults/ranges: numeric defaults come
from `dataclasses.fields(PlannerConfig)` so code and GUI can't drift. Each
group expander opens with one compact `st.latex` "Key formulas" block.
Widget keys use the `mpw_<field>` convention (presets round-trip them).
"""
from __future__ import annotations

from dataclasses import fields

import streamlit as st

from motion_planner.config_io import PlannerConfig

_DEFAULTS = {f.name: f.default for f in fields(PlannerConfig)}


def _num(label: str, field_name: str, help_text: str, step: float = 0.05,
         min_value: float = 0.0, max_value: "float | None" = None) -> float:
    return st.number_input(
        label, value=float(_DEFAULTS[field_name]), step=step, min_value=min_value,
        max_value=max_value, key=f"mpw_{field_name}", help=help_text)


def build_planner_config() -> PlannerConfig:
    with st.sidebar.expander("Fingering (Viterbi DP)", expanded=False):
        st.latex(r"C = w_t\,\max(0, T_{req}-T_{avail})^2 + w_s\left(\tfrac{\Delta p}{7}\right)^2"
                 r" + w_{str}\,|\Delta band| + w_{open}")
        w_time = _num("w_time — lateness weight", "w_time",
                      "wₜ — cost of arriving late: (T_req − T_avail)² per finger move. "
                      "T_req from trapezoidal axis kinematics + lift/press times.", 0.5)
        w_shift = _num("w_shift — position shift", "w_shift",
                       "wₛ — Maezawa 2012 continuity cost $(\\Delta p/7)^2$ per shift "
                       "of Δp semitones (same shape as the reducer's w_jump).", 0.1)
        w_string = _num("w_string — bow-band travel", "w_string",
                        "w_str — |Δband| between inclination bands; proxy for bow-Z travel.", 0.05)
        w_open = _num("w_open — open-string bias", "w_open",
                      "Flat penalty per open string (timbre mismatch vs zero travel).", 0.05)
        steal = _num("steal_fraction", "steal_fraction",
                     "Fraction of the previous note's tail a transition may consume "
                     "(early release for the next move).", 0.05, 0.0, 0.9)

    with st.sidebar.expander("Bowing (Schelleng)", expanded=False):
        st.latex(r"F_{min} = k_{min}\tfrac{Z^2 v_b}{\beta^2} \le F \le F_{max} = k_{max}\tfrac{Z v_b}{\beta}"
                 r",\quad F = F_{min}^{1-u} F_{max}^{u}")
        lift_gap = _num("lift_gap_s", "lift_gap_s",
                        "Rests longer than this get a bow lift + land (else the hair "
                        "stays on the string with force → 0).", 0.05)
        roll_span = _num("roll_span_s (D-024)", "roll_span_s",
                         "Rolled-triple dwell on the (low,mid) double-stop band before "
                         "sweeping to (mid,high). Realized onsets shift accordingly.", 0.01)
        u_default = _num("u_default — brightness", "u_default",
                         "u ∈ [0,1] places the force inside the Schelleng wedge in log "
                         "space; 0.5 = geometric mean. From tristimulus T₃ when present.",
                         0.05, 0.0, 1.0)
        coupling = _num("coupling_warn_fraction", "coupling_warn_fraction",
                        "Warn when inclination-rate coupling ω·r exceeds this fraction "
                        "of the note's bow speed v_b (loudness wobble during fast rolls).",
                        0.05, 0.0, 1.0)

    with st.sidebar.expander("Vibrato", expanded=False):
        st.latex(r"\Delta z = \tfrac{depth_\mathrm{cents}}{100}\cdot\tfrac{L\ln 2}{12}\,2^{-p/12}"
                 r",\quad a_{pk} = (2\pi f)^2 \Delta z \le a_{max}")
        min_depth = _num("min depth (cents)", "vibrato_min_depth_cents",
                         "Contours shallower than this are treated as no vibrato.", 1.0)
        min_cycles = _num("min cycles", "vibrato_min_cycles",
                          "At least this many oscillation periods must fit in the note.", 0.5)
        f_lo = _num("f_lo (Hz)", "vibrato_f_lo_hz",
                    "Lower rate bound for the autocorrelation search (typ. 3–9 Hz).", 0.5)
        f_hi = _num("f_hi (Hz)", "vibrato_f_hi_hz",
                    "Upper rate bound. Depth (never rate) is clipped to axis physics.", 0.5)

    return PlannerConfig(
        w_time=w_time, w_shift=w_shift, w_string=w_string, w_open=w_open,
        steal_fraction=steal, lift_gap_s=lift_gap, roll_span_s=roll_span,
        u_default=u_default, coupling_warn_fraction=coupling,
        vibrato_min_depth_cents=min_depth, vibrato_min_cycles=min_cycles,
        vibrato_f_lo_hz=f_lo, vibrato_f_hi_hz=f_hi,
    )
