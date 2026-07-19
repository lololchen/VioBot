"""Sidebar widget builders: dataclass fields -> Streamlit widgets -> frozen configs.

Hard rule (module CLAUDE.md / plan): widget *defaults* are read programmatically
from `dataclasses.fields(...)` so the GUI can never silently drift from the
code defaults declared in transcriber.py / timbre.py / reducer.py / soundsim.py.
Only the *ranges* sliders sweep over are hardcoded here (a slider needs bounds;
the code defaults themselves don't carry any) -- see `_RANGES` / `_CHOICES`.

Each `build_*_config()` renders one `st.expander` in the sidebar (widgets +
a "reset to defaults" button that clears that config's widget keys from
`st.session_state`, so the widgets re-read the dataclass defaults on the next
run) and returns the corresponding frozen config dataclass.

`StageConfig.open_strings_hz` is never exposed here (D-015, decisions.md):
it is a decision-governed physical constant, not a tuning knob.

Preset save/load lives at the bottom (`build_preset_controls`): saving never
silently overwrites an existing file (explicit checkbox required), loading
writes the preset's field values straight into the relevant widget keys in
`st.session_state` and reruns so the widgets pick them up.
"""
from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import streamlit as st

from ..config_io import Preset, load_preset, save_preset
from ..reducer import StageConfig
from ..soundsim import RenderConfig
from ..timbre import TimbreConfig
from ..transcriber import MonoConfig

PRESETS_DIR = Path(__file__).resolve().parents[2] / "presets"

# ---------------------------------------------------------------------------
# widget metadata: slider (lo, hi, step) and selectbox choices per field name.
# NEVER a source of default *values* -- those always come from the dataclass.
# ---------------------------------------------------------------------------

_MONO_RANGES = {
    "hop_s": (0.001, 0.05, 0.001),
    "fmin_hz": (20.0, 500.0, 1.0),
    "fmax_hz": (500.0, 5000.0, 10.0),
    "voicing_threshold": (0.0, 1.0, 0.01),
    "min_note_s": (0.0, 0.5, 0.01),
    "merge_gap_s": (0.0, 0.2, 0.005),
    "split_semitones": (0.1, 3.0, 0.05),
    "envelope_hop_s": (0.005, 0.1, 0.005),
}
_MONO_CHOICES = {"backend": ["auto", "crepe", "yin"]}

_TIMBRE_RANGES = {
    "n_harmonics": (1, 16, 1),
    "frame_size": (512, 8192, 512),
    "hop_size": (128, 4096, 128),
    "zero_pad_factor": (1, 4, 1),
    "tolerance_semitones": (0.1, 3.0, 0.1),
}
_TIMBRE_CHOICES = {"backend": ["numpy", "essentia"]}

_STAGE_RANGES = {
    "max_voices": (1, 3, 1),
    "max_pitch_hz": (1000.0, 4000.0, 10.0),
    "max_fingerboard_semitones": (5.0, 30.0, 1.0),
    "max_position_span_semitones": (1.0, 12.0, 0.5),
    "pitch_tolerance_semitones": (0.0, 2.0, 0.05),
    "max_active": (2, 20, 1),
    "min_keep_s": (0.0, 0.5, 0.01),
    "w_amp": (0.0, 5.0, 0.1),
    "w_pitch": (0.0, 5.0, 0.1),
    "w_dur": (0.0, 5.0, 0.1),
    "w_frag": (0.0, 10.0, 0.1),
    "w_jump": (0.0, 5.0, 0.1),
}
_STAGE_CHOICES: dict = {}
_STAGE_SKIP = ("open_strings_hz",)  # D-015: decision-governed constant, not a GUI knob

_RENDER_RANGES = {
    "sample_rate": (8000, 48000, 100),
    "n_partials": (1, 16, 1),
    "fade_s": (0.0, 0.05, 0.001),
}
_RENDER_CHOICES: dict = {}
# The GUI always renders BOTH backends in the A/B panel (additive baseline +
# a SoundFont row), so `backend` is not a sidebar knob anymore; `midi_program`
# gets a custom named-instrument selectbox below instead of a generic slider.
_RENDER_SKIP = ("backend", "midi_program")

# GM bank-0 program -> display name for the SoundFont (fluidsynth) render row.
# The selectbox stores the *program number* under the standard widget key
# (cfgw_render_midi_program) so preset save/load round-trips it untouched.
GM_INSTRUMENTS = {
    40: "Violin",
    41: "Viola",
    42: "Cello",
    43: "Contrabass",
    45: "Pizzicato strings",
    46: "Orchestral harp",
    0: "Acoustic grand piano",
    73: "Flute",
    71: "Clarinet",
}

# ---------------------------------------------------------------------------
# Per-field hover help (shown as Streamlit's `?` tooltip on each widget) and
# a compact per-group formula block (rendered at the top of each expander).
# These are DOCUMENTATION ONLY -- they never feed default values or ranges;
# the text just mirrors the algorithm docstrings in transcriber.py / timbre.py
# / reducer.py / soundsim.py so the sidebar explains the maths it is tuning.
# Tooltip strings use Unicode Greek/math symbols plus inline `$...$` LaTeX,
# both of which Streamlit's Markdown tooltip renderer supports.
# ---------------------------------------------------------------------------

_MONO_HELP = {
    "backend": "f0 tracker. **auto** → CREPE (neural, needs TensorFlow) if installed, "
               "else the always-available numpy **YIN** (de Cheveigné & Kawahara 2002). "
               "**crepe** / **yin** force one.",
    "hop_s": "Frame hop $\\Delta t$ between successive f0 estimates (seconds). "
             "Frame rate $=1/\\Delta t$. Smaller ⇒ finer time resolution but more frames.",
    "fmin_hz": "Lowest detectable pitch $f_{\\min}$. Sets the YIN frame size "
               "$N=\\text{next\\_pow2}(2f_s/f_{\\min})$ and the longest lag "
               "$\\tau_{\\max}=\\lceil f_s/f_{\\min}\\rceil$.",
    "fmax_hz": "Highest detectable pitch $f_{\\max}$. Sets the shortest lag "
               "$\\tau_{\\min}=\\lfloor f_s/f_{\\max}\\rfloor$. Frames outside "
               "$[f_{\\min},f_{\\max}]$ are marked unvoiced.",
    "voicing_threshold": "Voicing gate $\\theta_v\\in[0,1]$. A frame is voiced iff "
                         "confidence $\\ge\\theta_v$. For YIN, confidence $=1-d'(\\tau^*)$ "
                         "(1 minus the CMNDF minimum).",
    "min_note_s": "Minimum note length. Segments shorter than this (after merging) are discarded.",
    "merge_gap_s": "Longest silent gap between two same-pitch segments that still merges "
                   "them into one note (pitch match within `split_semitones`).",
    "split_semitones": "Pitch-deviation split threshold $\\Delta p$ (semitones). A running note "
                       "splits when $\\lvert 12\\log_2(f_0/\\tilde f_0)\\rvert>\\Delta p$ from its "
                       "running median $\\tilde f_0$.",
    "envelope_hop_s": "Hop for sampling each note's amplitude envelope, "
                      "$\\text{amp}_{dB}=20\\log_{10}(\\text{RMS})$ (floor $-80$ dB).",
}
_MONO_FORMULAS = [
    r"d'(\tau)=\frac{d(\tau)\,\tau}{\sum_{j=1}^{\tau} d(j)}\quad\text{(CMNDF)},\qquad f_0=\frac{f_s}{\tau^*}",
    r"\Delta\text{semitones}=12\,\log_2\!\frac{f_a}{f_b},\qquad \text{amp}_{dB}=20\,\log_{10}(\text{RMS})",
]

_TIMBRE_HELP = {
    "backend": "Harmonic analyser. **numpy** = per-note STFT DSP (default, always available). "
               "**essentia** = same fields via Essentia (no Windows wheels).",
    "n_harmonics": "Number of harmonics $K$ analysed; harmonic $k$ is searched near $k\\cdot f_0$ "
                   "for $k=1\\ldots K$.",
    "frame_size": "STFT analysis window length $N$ (samples), Hann-windowed.",
    "hop_size": "STFT hop $H$ (samples) between analysis frames inside one note.",
    "zero_pad_factor": "Zero-pad multiplier $Z$: FFT length $=N\\cdot Z$, so the bin spacing is "
                       "$f_s/(N Z)$. Higher ⇒ finer frequency interpolation.",
    "tolerance_semitones": "Half-window (semitones) of the peak search around each $k\\cdot f_0$: "
                           "$[\\,k f_0\\,2^{-\\text{tol}/24},\\; k f_0\\,2^{+\\text{tol}/24}\\,]$.",
}
_TIMBRE_FORMULAS = [
    r"\text{OER}=\frac{\sum_{k\ \mathrm{odd}\ge 3} a_k^2}{\sum_{k\ \mathrm{even}} a_k^2},\qquad "
    r"T=\Big(\tfrac{a_1^2}{\Sigma},\ \tfrac{a_2^2+a_3^2+a_4^2}{\Sigma},\ \tfrac{\sum_{k\ge5}a_k^2}{\Sigma}\Big),\ \ \Sigma=\textstyle\sum_k a_k^2",
    r"\beta=\frac{1}{\Sigma}\sum_{k\ge 2}\frac{\lvert f_k-k f_0\rvert}{k f_0}\,a_k^2\quad\text{(inharmonicity)}",
]

_STAGE_HELP = {
    "max_voices": "Max simultaneous voices $N\\in\\{1,2,3\\}$ (hardware stage gate): every "
                  "kept subset obeys $\\lvert S\\rvert\\le N$.",
    "max_pitch_hz": "Upper playable pitch bound; notes above it are dropped by the reducer.",
    "max_fingerboard_semitones": "Highest stopped semitone above each open string "
                                 "(PROVISIONAL — not from the papers).",
    "max_position_span_semitones": "Max cross-string finger span in one hand position "
                                   "(Maezawa 2012 $P_\\text{nat}=5$ st, $P_\\max=6$).",
    "rolled_triples": "If set, 3-note chords survive only as **rolled** — single-bow triple "
                      "stops are physically infeasible (D-009).",
    "pitch_tolerance_semitones": "Jitter allowance at range / position boundaries so exact-ET "
                                 "notes aren't rejected (D-015).",
    "max_active": "Pruning bound $M$: if more than $M$ notes are active, keep the top $M$ by "
                  "$\\text{importance}$ before enumerating subsets.",
    "min_keep_s": "After Viterbi clipping, drop any kept note shorter than this.",
    "w_amp": "Emission weight on loudness: adds "
             "$w_\\text{amp}\\cdot\\text{clip}_{01}\\!\\big((\\text{peak}_{dB}+60)/60\\big)$ to importance.",
    "w_pitch": "Emission weight on the top-voice bias: adds $w_\\text{pitch}\\cdot\\text{rank}_{01}(n)$ "
               "(pitch percentile among concurrent notes).",
    "w_dur": "Emission weight on duration: adds $w_\\text{dur}\\cdot\\min(1,\\text{dur}_s)$.",
    "w_frag": "Transition penalty charged per fragmentation event (a mid-note drop or a late entry).",
    "w_jump": "Transition penalty on register jumps: $w_\\text{jump}\\cdot(\\Delta p/7)^2$ "
              "(Maezawa 2012: ≤7 st stays in one hand position).",
}
_STAGE_FORMULAS = [
    r"\mathrm{imp}(n)=w_a\,\mathrm{clip}_{01}\!\Big(\tfrac{\mathrm{peak}_{dB}+60}{60}\Big)"
    r"+w_p\,\mathrm{rank}_{01}(n)+w_d\min(1,\mathrm{dur})",
    r"E(S,k)=\!\!\sum_{n\notin S}\!\!\mathrm{imp}(n)\,\lvert I_k\rvert,\qquad "
    r"T=w_f\,n_\mathrm{frag}+w_j\Big(\tfrac{\Delta p}{7}\Big)^2",
    r"\text{Viterbi minimizes}\quad \textstyle\sum_k E(S_k,k)+\sum_k T(S_{k-1}\!\to S_k)",
]

_RENDER_HELP = {
    "midi_program": "Instrument for the SoundFont (fluidsynth) render row — a GM bank-0 "
                    "program of the loaded `.sf2`. The additive render row is unaffected.",
    "sample_rate": "Output WAV sample rate $f_s$ (Hz). Shared by both render rows.",
    "n_partials": "Partial count $K$ for the additive fallback when a note has no measured "
                  "harmonics ($a_k=1/k^2$ rolloff).",
    "fade_s": "Raised-cosine fade in/out per note (seconds) to avoid click transients "
              "(additive backend).",
    "soundfont_path": "Explicit `.sf2` path (SoundFont row only). Blank ⇒ discover via "
                      "`MELODY_EXTRACTOR_SF2` or the per-user soundfonts folder.",
}
_RENDER_FORMULAS = [
    r"x(t)=\sum_{k=1}^{K} a_k\,\sin\!\Big(2\pi k\!\int_0^t f_0(\tau)\,d\tau\Big),"
    r"\qquad a_k=1/k^2\ \text{(fallback rolloff)}",
]


def _render_formulas(formulas: list) -> None:
    """Compact 'Key formulas' block rendered at the top of a group's expander."""
    if not formulas:
        return
    st.caption("Key formulas")
    for tex in formulas:
        st.latex(tex)
    st.markdown("---")


def _widget_key(prefix: str, field_name: str) -> str:
    return f"cfgw_{prefix}_{field_name}"


def _build_config_widgets(cls: type, prefix: str, ranges: dict, choices: dict,
                          skip: tuple = (), help_map: "dict | None" = None) -> Any:
    """Render one widget per field of `cls` (skipping `skip`) using `st.*`
    calls -- callers wrap this in a `with st.expander(...):` block so these
    land inside it. Widget *default* values come only from
    `dataclasses.fields(cls)`; `ranges`/`choices` only bound/enumerate them.
    `help_map` (field name -> tooltip markdown) is documentation only.
    """
    help_map = help_map or {}
    kwargs = {}
    for f in fields(cls):
        if f.name in skip:
            continue
        default = f.default
        key = _widget_key(prefix, f.name)
        help_text = help_map.get(f.name)

        if f.name in choices:
            options = choices[f.name]
            index = options.index(default) if default in options else 0
            kwargs[f.name] = st.selectbox(f.name, options, index=index, key=key, help=help_text)
        elif isinstance(default, bool):
            kwargs[f.name] = st.checkbox(f.name, value=default, key=key, help=help_text)
        elif isinstance(default, int):
            lo, hi, step = ranges.get(f.name, (0, max(1, default * 4), 1))
            kwargs[f.name] = st.slider(
                f.name, min_value=int(lo), max_value=int(hi), value=int(default), step=int(step),
                key=key, help=help_text,
            )
        elif isinstance(default, float):
            lo, hi, step = ranges.get(f.name, (0.0, max(1.0, default * 4.0), 0.01))
            kwargs[f.name] = st.slider(
                f.name, min_value=float(lo), max_value=float(hi), value=float(default), step=float(step),
                key=key, help=help_text,
            )
        elif default is None or isinstance(default, str):
            text = st.text_input(f.name, value=(default or ""), key=key, help=help_text)
            kwargs[f.name] = (text if text else None) if default is None else text
        else:
            continue  # tuple-valued fields (e.g. open_strings_hz) are always in `skip`

    return cls(**kwargs)


def _reset_button(cls: type, prefix: str, skip: tuple = ()) -> None:
    if st.button("Reset to defaults", key=f"{_widget_key(prefix, 'reset')}"):
        for f in fields(cls):
            if f.name in skip:
                continue
            st.session_state.pop(_widget_key(prefix, f.name), None)
        st.rerun()


def build_mono_config() -> MonoConfig:
    with st.sidebar.expander("Mono transcription — transcriber.MonoConfig", expanded=False):
        _render_formulas(_MONO_FORMULAS)
        cfg = _build_config_widgets(MonoConfig, "mono", _MONO_RANGES, _MONO_CHOICES, help_map=_MONO_HELP)
        _reset_button(MonoConfig, "mono")
    return cfg


def build_timbre_config() -> TimbreConfig:
    with st.sidebar.expander("Timbre analysis — timbre.TimbreConfig", expanded=False):
        _render_formulas(_TIMBRE_FORMULAS)
        cfg = _build_config_widgets(TimbreConfig, "timbre", _TIMBRE_RANGES, _TIMBRE_CHOICES, help_map=_TIMBRE_HELP)
        _reset_button(TimbreConfig, "timbre")
    return cfg


def build_stage_config() -> StageConfig:
    """All StageConfig fields + the 5 cost weights, EXCEPT `open_strings_hz`
    (D-015: decision-governed physical constant). `max_voices` here is the
    config value (round-trips through presets); the Reducer panel's stage
    radio (inspector_view.py) FOLLOWS it — re-seeded whenever this slider
    changes — and can override it per-preview via `dataclasses.replace`,
    mirroring the CLI's `--stage` flag overriding a preset's `max_voices`.
    """
    with st.sidebar.expander("Reducer — reducer.StageConfig (advanced)", expanded=False):
        _render_formulas(_STAGE_FORMULAS)
        st.caption("open_strings_hz is a decision-governed constant (D-015) and is not exposed here.")
        cfg = _build_config_widgets(StageConfig, "stage", _STAGE_RANGES, _STAGE_CHOICES,
                                    skip=_STAGE_SKIP, help_map=_STAGE_HELP)
        _reset_button(StageConfig, "stage", skip=_STAGE_SKIP)
    return cfg


def build_render_config() -> RenderConfig:
    """One GLOBAL RenderConfig shared by every A/B render row. The A/B panel
    always renders the additive baseline AND a SoundFont (fluidsynth) row, so
    `backend` is no longer a sidebar choice (inspector_view derives both via
    `dataclasses.replace`); the dropdown instead picks the SoundFont row's
    instrument (GM program), stored in `midi_program`.
    """
    with st.sidebar.expander("Render — soundsim.RenderConfig", expanded=False):
        _render_formulas(_RENDER_FORMULAS)
        st.caption("The A/B panel always renders additive (baseline) + a SoundFont row; "
                   "these parameters are shared by both.")

        options = list(GM_INSTRUMENTS)
        program_key = _widget_key("render", "midi_program")
        # A preset saved by a future build may carry a program outside our
        # curated list; drop the stale session value instead of crashing.
        if st.session_state.get(program_key) not in options and program_key in st.session_state:
            st.session_state.pop(program_key)
        default_program = RenderConfig().midi_program
        midi_program = st.selectbox(
            "instrument (SoundFont row)", options,
            index=options.index(default_program) if default_program in options else 0,
            format_func=lambda p: f"{GM_INSTRUMENTS[p]} (GM {p})",
            key=program_key, help=_RENDER_HELP["midi_program"],
        )

        cfg = _build_config_widgets(RenderConfig, "render", _RENDER_RANGES, _RENDER_CHOICES,
                                    skip=_RENDER_SKIP, help_map=_RENDER_HELP)
        cfg = replace(cfg, midi_program=int(midi_program))
        _reset_button(RenderConfig, "render", skip=("backend",))  # also clears midi_program
    return cfg


# ---------------------------------------------------------------------------
# preset save/load
# ---------------------------------------------------------------------------

_CONFIG_SPECS = (
    (MonoConfig, "mono", ()),
    (TimbreConfig, "timbre", ()),
    (StageConfig, "stage", _STAGE_SKIP),
    # `backend` has no widget (both backends always render); `midi_program`
    # DOES have one (the instrument selectbox) and must round-trip via presets.
    (RenderConfig, "render", ("backend",)),
)


def _apply_preset_to_widget_state(preset: Preset) -> None:
    """Write a loaded preset's field values into the widgets' session_state
    keys (skipping fields with no widget, e.g. StageConfig.open_strings_hz --
    D-015) so the next rerun's widgets reflect the preset."""
    configs = {"mono": preset.mono, "timbre": preset.timbre, "stage": preset.stage, "render": preset.render}
    for cls, prefix, skip in _CONFIG_SPECS:
        cfg = configs[prefix]
        for f in fields(cls):
            if f.name in skip:
                continue
            value = getattr(cfg, f.name)
            if value is None:
                value = ""
            st.session_state[_widget_key(prefix, f.name)] = value


def build_preset_controls(mono_cfg: MonoConfig, timbre_cfg: TimbreConfig,
                           stage_cfg: StageConfig, render_cfg: RenderConfig) -> None:
    """Sidebar-bottom preset save/load UI (plan Tab 1 "Sidebar" bullet).

    Save: name + comment inputs; writes `presets/<name>.json` via
    `config_io.save_preset` (byte-deterministic). NO silent overwrite: if the
    file already exists, an explicit "overwrite" checkbox must be ticked.

    Load: a selectbox over `presets/*.json`; "Apply" pushes the preset's
    values into the widgets' session_state keys and reruns.
    """
    st.sidebar.markdown("---")
    st.sidebar.subheader("Presets (config_io)")

    PRESETS_DIR.mkdir(parents=True, exist_ok=True)
    existing = sorted(p.name for p in PRESETS_DIR.glob("*.json"))

    with st.sidebar.expander("Save current parameters as a preset", expanded=False):
        name = st.text_input("Name", key="preset_save_name")
        comment = st.text_area("Comment", key="preset_save_comment")
        target_name = f"{name}.json" if name else ""
        target_exists = bool(target_name) and (PRESETS_DIR / target_name).exists()
        overwrite = False
        if target_exists:
            overwrite = st.checkbox(f"'{target_name}' already exists — overwrite it", key="preset_save_overwrite")
        if st.button("Save preset", key="preset_save_btn"):
            if not name.strip():
                st.error("Preset name is required.")
            elif target_exists and not overwrite:
                st.warning(f"'{target_name}' already exists. Tick the overwrite checkbox above to replace it.")
            else:
                preset = Preset(
                    name=name, comment=comment,
                    mono=mono_cfg, timbre=timbre_cfg, stage=stage_cfg, render=render_cfg,
                )
                path = save_preset(preset, PRESETS_DIR / target_name)
                st.success(f"Saved {path.name}")

    with st.sidebar.expander("Load a preset", expanded=False):
        if not existing:
            st.caption("No presets saved yet.")
        else:
            choice = st.selectbox("Preset file", existing, key="preset_load_choice")
            if st.button("Apply preset to widgets", key="preset_load_btn"):
                preset = load_preset(PRESETS_DIR / choice)
                _apply_preset_to_widget_state(preset)
                st.success(f"Applied {choice}")
                st.rerun()
