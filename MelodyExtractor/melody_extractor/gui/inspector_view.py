"""Tab 1 — Pipeline Inspector (plan_GUI_MelodyExtractor.md "Tab 1").

Each pipeline stage renders as a NUMBERED `st.status` box (D-019): the box
header carries a spinner while its stage computes ("running"), then a
checkmark. Each stage's `pipeline_cache` call happens INSIDE its own box so
the spinner is truthful. A stage that actually computed (cache miss, judged
by wall time > _FRESH_S) auto-expands on completion and smooth-scrolls the
page so the NEXT stage's still-running header stays visible at the bottom
(style.scroll_to_anchor); instant cache hits leave scroll position and the
user's manual open/close state alone. The cache chain still means a weight
tweak only recomputes what actually changed (see pipeline_cache.py).

All plotly/streamlit calls live here; `figures.py`/`audio_bytes.py` stay pure
and streamlit-free (module CLAUDE.md hard rule).

MIDI-uploaded input degrades gracefully: there is no waveform/spectrogram/
frame-track to show (input_adapter.load_midi never touches audio) and no
"original" recording for the A/B player's loudness match -- both panels say
so explicitly rather than silently doing nothing.
"""
from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import streamlit as st

from .. import reducer, soundsim
from ..input_adapter import AudioBuffer
from ..reducer import StageConfig
from ..schema import NoteSequence
from ..soundsim import RenderConfig
from ..timbre import TimbreConfig
from ..transcriber import MonoConfig
from . import figures, pipeline_cache, style

_OUT_DIR = Path(__file__).resolve().parents[2] / "out" / "gui_paired_export"

_COST_FORMULA_MARKERS = ("== Costs (Viterbi minimizes total) ==", "== Decoding ==")

# A step whose body took longer than this actually computed something (cache
# miss); only those auto-expand + auto-scroll. Cache-hit reruns (widget
# tweaks) finish in milliseconds and must not fight the user's scroll
# position or manually-collapsed boxes.
_FRESH_S = 0.75

_N_STEPS = 6


def _reducer_cost_formula_text() -> str:
    """Pull the emission/transition cost formulas straight out of reducer.py's
    module docstring for display (never re-derive them by hand -- if the
    formula changes, this panel changes with it automatically)."""
    doc = reducer.__doc__ or ""
    start_marker, end_marker = _COST_FORMULA_MARKERS
    start = doc.find(start_marker)
    end = doc.find(end_marker)
    if start == -1 or end == -1:
        return "(cost formula section not found in reducer.py's docstring)"
    return doc[start:end].strip()


_CHARTS_OFF_MSG = "Chart hidden — enable 'Show charts' in the sidebar (Display)."


def _step_start(step: int, title: str):
    """Anchor + collapsed status box ("N. title") with a running spinner."""
    style.step_anchor(f"viostep-{step}")
    return st.status(f"{step}. {title}", expanded=False)


def _step_done(status, step: int, t0: float) -> None:
    """Complete a step. Fresh computation -> expand it and scroll so the next
    step's (still-running) header stays visible; cache hit -> touch nothing
    beyond the checkmark."""
    if time.perf_counter() - t0 > _FRESH_S:
        status.update(state="complete", expanded=True)
        next_anchor = f"viostep-{step + 1}" if step < _N_STEPS else "viostep-end"
        style.scroll_to_anchor(next_anchor)
    else:
        status.update(state="complete")


def render(digest: "str | None", mono_cfg: MonoConfig, timbre_cfg: TimbreConfig,
           stage_cfg: StageConfig, render_cfg: RenderConfig,
           show_charts: bool = True) -> None:
    if digest is None:
        st.info(
            "Select a fixture, upload an audio/MIDI file, or paste a URL in the "
            "sidebar to begin (on mobile: tap » in the top-left to open it)."
        )
        return

    # -----------------------------------------------------------------
    # 1. Input — input_adapter.load_audio / load_midi
    # -----------------------------------------------------------------
    t0 = time.perf_counter()
    with _step_start(1, "Input — input_adapter.load_audio / load_midi") as status:
        loaded = pipeline_cache.load(digest)
        is_midi = isinstance(loaded, NoteSequence)
        if is_midi:
            st.info("MIDI input: input_adapter.load_midi parses notes directly, bypassing "
                    "audio entirely, so there is no waveform/spectrogram here.")
        else:
            audio: AudioBuffer = loaded
            st.caption(f"{len(audio.pcm) / audio.sample_rate:.1f} s @ {audio.sample_rate} Hz")
            if not show_charts:
                st.caption(_CHARTS_OFF_MSG)
            else:
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(figures.waveform_figure(audio.pcm, audio.sample_rate),
                                     use_container_width=True, key="input_waveform_chart")
                with col2:
                    st.plotly_chart(figures.spectrogram_figure(audio.pcm, audio.sample_rate),
                                     use_container_width=True, key="input_spectrogram_chart")
    _step_done(status, 1, t0)

    # -----------------------------------------------------------------
    # 2. FrameTrack — transcriber.transcribe_mono
    # -----------------------------------------------------------------
    t0 = time.perf_counter()
    with _step_start(2, "FrameTrack — transcriber.transcribe_mono") as status:
        seq_transcribed = pipeline_cache.transcribe(digest, mono_cfg)
        track = seq_transcribed.features[0] if seq_transcribed.features else None
        if is_midi:
            st.info("MIDI input: no per-frame f0/voicing/amplitude track (nothing was transcribed).")
        elif track is None:
            st.info("No frame track produced for this input.")
        elif not show_charts:
            st.caption(f"{len(track.f0_hz)} frames @ {track.hop_s * 1000:.0f} ms hop. {_CHARTS_OFF_MSG}")
        else:
            st.plotly_chart(
                figures.frame_track_figure(track, mono_cfg.voicing_threshold),
                use_container_width=True, key="frame_track_chart",
            )
    _step_done(status, 2, t0)

    # -----------------------------------------------------------------
    # 3. Notes — schema.NoteSequence (post timbre pass)
    # -----------------------------------------------------------------
    t0 = time.perf_counter()
    selected_idx: "int | None" = None
    with _step_start(3, "Notes — schema.NoteSequence (post timbre.add_harmonics)") as status:
        seq_timbre = pipeline_cache.timbre(digest, mono_cfg, timbre_cfg)
        notes = seq_timbre.sorted().notes
        if not notes:
            st.info("No notes extracted.")
        else:
            options = list(range(len(notes)))

            def _fmt(i: int) -> str:
                n = notes[i]
                return f"#{i} — {n.pitch_hz:.1f} Hz @ {n.onset_s:.2f}s (dur {n.duration_s:.2f}s)"

            fallback_idx = st.selectbox(
                "Select note (fallback for chart click — AppTest + accessibility)",
                options, format_func=_fmt, key="note_select_fallback",
            )
            selected_idx = fallback_idx

            if not show_charts:
                st.caption(f"{len(notes)} notes. {_CHARTS_OFF_MSG}")
            else:
                pianoroll = figures.pianoroll_figure(seq_timbre, track=track, selected=fallback_idx)
                event = st.plotly_chart(
                    pianoroll, use_container_width=True, on_select="rerun", key="pianoroll_chart",
                )
                points = (event or {}).get("selection", {}).get("points", []) if event else []
                if points:
                    customdata = points[0].get("customdata")
                    if customdata:
                        try:
                            clicked = int(customdata[0])
                        except (TypeError, ValueError):
                            clicked = None
                        if clicked is not None and 0 <= clicked < len(notes):
                            selected_idx = clicked
    _step_done(status, 3, t0)

    # -----------------------------------------------------------------
    # 4. Timbre — timbre.add_harmonics (selected note)
    # -----------------------------------------------------------------
    t0 = time.perf_counter()
    with _step_start(4, "Timbre — timbre.add_harmonics (selected note)") as status:
        if selected_idx is None:
            st.info("No note selected.")
        else:
            note = notes[selected_idx]
            figs = figures.timbre_figures(note)
            col1, col2 = st.columns(2)
            col1.plotly_chart(figs["harmonics_bar"], use_container_width=True, key="timbre_harmonics_chart")
            col2.plotly_chart(figs["tristimulus_bar"], use_container_width=True, key="timbre_tristimulus_chart")

            m1, m2 = st.columns(2)
            odd_even = figs["odd_even_ratio"]
            inharm = figs["inharmonicity"]
            m1.metric("Odd/even ratio", f"{odd_even:.3f}" if odd_even is not None else "—")
            m2.metric("Inharmonicity", f"{inharm:.3f}" if inharm is not None else "—")

            if note.harmonics is None:
                st.info(
                    f"Note too short for one analysis frame "
                    f"(TimbreConfig.frame_size={timbre_cfg.frame_size} samples) — harmonics unavailable."
                )
    _step_done(status, 4, t0)

    # -----------------------------------------------------------------
    # 5. Reducer — reducer.reduce
    # -----------------------------------------------------------------
    t0 = time.perf_counter()
    with _step_start(5, "Reducer — reducer.reduce") as status:
        default_index = stage_cfg.max_voices - 1 if stage_cfg.max_voices in (1, 2, 3) else 0
        stage_n = st.radio(
            "Hardware stage to preview (overrides StageConfig.max_voices, like the CLI's --stage)",
            [1, 2, 3], index=default_index, horizontal=True, key="reducer_stage_radio",
        )
        effective_stage_cfg = replace(stage_cfg, max_voices=stage_n)

        reduced_seq = pipeline_cache.reduce(digest, mono_cfg, timbre_cfg, effective_stage_cfg)
        diff = figures.diff_reduction(seq_timbre, reduced_seq)
        if not show_charts:
            st.caption(
                f"kept {len(diff['kept'])} / trimmed {len(diff['trimmed'])} / "
                f"dropped {len(diff['dropped'])}. {_CHARTS_OFF_MSG}"
            )
        else:
            st.plotly_chart(
                figures.reduction_figure(seq_timbre, reduced_seq, diff),
                use_container_width=True, key="reduction_chart",
            )

        violations = reducer.playability_violations(reduced_seq, effective_stage_cfg)
        if violations:
            for v in violations:
                st.error(v)
        else:
            st.success("No playability violations.")

        with st.expander("Cost formula (reducer.py docstring)", expanded=False):
            st.code(_reducer_cost_formula_text(), language=None)

        st.caption("Importance breakdown for dropped notes (display-only recompute — never calls reducer internals):")
        rows = figures.importance_table(seq_timbre, diff, effective_stage_cfg)
        if rows:
            st.dataframe(rows, use_container_width=True)
        else:
            st.caption("No notes dropped at this stage.")

        stage_meta = reduced_seq.meta.stage or {}
        pruned = bool(reduced_seq.meta.extra.get("pruned"))
        st.caption(
            f"meta.stage.max_voices = {stage_meta.get('max_voices', '—')}  ·  "
            f"meta.extra.pruned = {pruned}"
        )
    _step_done(status, 5, t0)

    # -----------------------------------------------------------------
    # 6. A/B audio — soundsim.render / audio_bytes
    # -----------------------------------------------------------------
    t0 = time.perf_counter()
    with _step_start(6, "A/B audio — soundsim.render_to_array / audio_bytes.wav_bytes") as status:
        # A backend failure here (e.g. fluidsynth DLL/SoundFont problems) must
        # degrade to an inline error, not kill the whole page with a traceback.
        extracted_wav = reduced_wav = None
        try:
            extracted_wav = pipeline_cache.render(digest, mono_cfg, timbre_cfg, render_cfg, None)
            reduced_wav = pipeline_cache.render(digest, mono_cfg, timbre_cfg, render_cfg, effective_stage_cfg)
        except (ImportError, ValueError, FileNotFoundError, RuntimeError) as exc:
            st.error(f"Render failed ({render_cfg.backend} backend): {exc}")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.caption("Original")
            if is_midi:
                st.caption("(no source recording for MIDI input)")
            else:
                from . import audio_bytes as audio_bytes_mod
                audio = loaded
                st.audio(audio_bytes_mod.wav_bytes(audio.pcm, audio.sample_rate), format="audio/wav")
        with col2:
            st.caption("Extracted render (pre-reduction)")
            if extracted_wav is not None:
                st.audio(extracted_wav, format="audio/wav")
        with col3:
            st.caption(f"Reduced render (stage {stage_n})")
            if reduced_wav is not None:
                st.audio(reduced_wav, format="audio/wav")

        if not is_midi and st.button("Export paired WAVs (soundsim.render_paired)", key="export_paired_btn"):
            dest = _OUT_DIR / digest[:12]
            orig_path, render_path = soundsim.render_paired(loaded, seq_timbre, dest, render_cfg)
            st.success(f"Wrote {orig_path.name} and {render_path.name}")
            st.caption(str(dest))
    _step_done(status, 6, t0)
    style.step_anchor("viostep-end")
