"""st.cache_data wrappers — bounded, keyed on content, JSON-text in/out.

Everything is passed as canonical JSON TEXT (hashable, deterministic) and
returned as JSON text / bytes, mirroring melody_extractor/gui/pipeline_cache's
philosophy: a parameter change recomputes only downstream entries."""
from __future__ import annotations

import io
import json

import streamlit as st

from melody_extractor.schema import NoteSequence


@st.cache_data(max_entries=8, show_spinner=False)
def plan_from_texts(seq_text: str, profile_text: str, config_json: str,
                    path_hint: str) -> "tuple[str, str]":
    from motion_planner.config_io import PlannerConfig
    from motion_planner.planner import plan
    from motion_planner.profile_io import profile_from_dict

    from melody_extractor.config_io import config_from_dict

    seq = NoteSequence.from_json(seq_text)
    profile = profile_from_dict(json.loads(profile_text))
    config = config_from_dict(PlannerConfig, json.loads(config_json))
    score, report = plan(seq, profile, config, source_path_hint=path_hint)
    return score.to_json(), report.to_json()


@st.cache_data(max_entries=8, show_spinner=False)
def simulate_text(score_text: str) -> str:
    from motion_planner.schema import MotionScore
    from motion_planner.simulate import simulate

    return simulate(MotionScore.from_json(score_text)).to_json()


@st.cache_data(max_entries=12, show_spinner=False)
def render_wav_bytes(seq_text: str) -> bytes:
    """Additive render (deterministic, D-011) → WAV bytes for st.audio."""
    from scipy.io import wavfile

    from melody_extractor.soundsim import RenderConfig, render_to_array

    seq = NoteSequence.from_json(seq_text)
    audio = render_to_array(seq, RenderConfig())
    buf = io.BytesIO()
    wavfile.write(buf, RenderConfig().sample_rate, audio)
    return buf.getvalue()


@st.cache_data(max_entries=4, show_spinner=False)
def compare_from_paths(profile_paths: "tuple[str, ...]", input_paths: "tuple[str, ...]",
                       config_json: str) -> str:
    from pathlib import Path

    from melody_extractor.config_io import config_from_dict
    from motion_planner.compare import run_compare
    from motion_planner.config_io import PlannerConfig

    config = config_from_dict(PlannerConfig, json.loads(config_json))
    report = run_compare([Path(p) for p in profile_paths],
                         [Path(p) for p in input_paths], config=config)
    return report.to_json()
