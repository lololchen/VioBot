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

from dataclasses import fields
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
_RENDER_CHOICES = {"backend": ["additive", "fluidsynth"]}


def _widget_key(prefix: str, field_name: str) -> str:
    return f"cfgw_{prefix}_{field_name}"


def _build_config_widgets(cls: type, prefix: str, ranges: dict, choices: dict, skip: tuple = ()) -> Any:
    """Render one widget per field of `cls` (skipping `skip`) using `st.*`
    calls -- callers wrap this in a `with st.expander(...):` block so these
    land inside it. Widget *default* values come only from
    `dataclasses.fields(cls)`; `ranges`/`choices` only bound/enumerate them.
    """
    kwargs = {}
    for f in fields(cls):
        if f.name in skip:
            continue
        default = f.default
        key = _widget_key(prefix, f.name)

        if f.name in choices:
            options = choices[f.name]
            index = options.index(default) if default in options else 0
            kwargs[f.name] = st.selectbox(f.name, options, index=index, key=key)
        elif isinstance(default, bool):
            kwargs[f.name] = st.checkbox(f.name, value=default, key=key)
        elif isinstance(default, int):
            lo, hi, step = ranges.get(f.name, (0, max(1, default * 4), 1))
            kwargs[f.name] = st.slider(
                f.name, min_value=int(lo), max_value=int(hi), value=int(default), step=int(step), key=key
            )
        elif isinstance(default, float):
            lo, hi, step = ranges.get(f.name, (0.0, max(1.0, default * 4.0), 0.01))
            kwargs[f.name] = st.slider(
                f.name, min_value=float(lo), max_value=float(hi), value=float(default), step=float(step), key=key
            )
        elif default is None or isinstance(default, str):
            text = st.text_input(f.name, value=(default or ""), key=key)
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
        cfg = _build_config_widgets(MonoConfig, "mono", _MONO_RANGES, _MONO_CHOICES)
        _reset_button(MonoConfig, "mono")
    return cfg


def build_timbre_config() -> TimbreConfig:
    with st.sidebar.expander("Timbre analysis — timbre.TimbreConfig", expanded=False):
        cfg = _build_config_widgets(TimbreConfig, "timbre", _TIMBRE_RANGES, _TIMBRE_CHOICES)
        _reset_button(TimbreConfig, "timbre")
    return cfg


def build_stage_config() -> StageConfig:
    """All StageConfig fields + the 5 cost weights, EXCEPT `open_strings_hz`
    (D-015: decision-governed physical constant). `max_voices` here is the
    config's own default field; the Reducer panel's stage radio
    (inspector_view.py) overrides it per-preview via `dataclasses.replace`,
    mirroring the CLI's `--stage` flag overriding a preset's `max_voices`.
    """
    with st.sidebar.expander("Reducer — reducer.StageConfig (advanced)", expanded=False):
        st.caption("open_strings_hz is a decision-governed constant (D-015) and is not exposed here.")
        cfg = _build_config_widgets(StageConfig, "stage", _STAGE_RANGES, _STAGE_CHOICES, skip=_STAGE_SKIP)
        _reset_button(StageConfig, "stage", skip=_STAGE_SKIP)
    return cfg


def build_render_config() -> RenderConfig:
    with st.sidebar.expander("Render — soundsim.RenderConfig", expanded=False):
        cfg = _build_config_widgets(RenderConfig, "render", _RENDER_RANGES, _RENDER_CHOICES)
        _reset_button(RenderConfig, "render")
    return cfg


# ---------------------------------------------------------------------------
# preset save/load
# ---------------------------------------------------------------------------

_CONFIG_SPECS = (
    (MonoConfig, "mono", ()),
    (TimbreConfig, "timbre", ()),
    (StageConfig, "stage", _STAGE_SKIP),
    (RenderConfig, "render", ()),
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
