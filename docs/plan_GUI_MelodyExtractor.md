# Plan — Streamlit GUI for MelodyExtractor

Status: **planned, not yet implemented** (2026-07-19). Execute in a fresh session; everything needed is in this document.

## Context & goals
The MelodyExtractor pipeline is built and validated (98 tests green, eval baseline in `MelodyExtractor/out/eval_baseline.json`). The GUI serves three goals, agreed with the user:
1. **Understand** the extraction process/code at a high level (stage-by-stage visualization, each panel captioned with the module/function it shows).
2. **Judge** extraction results — visually (plots) and by ear (A/B audio players).
3. **Adjust** parameters, exporting good ones as **named preset JSONs** consumed by a new CLI `--config` flag.

Decisions made with the user: **Streamlit + plotly** (new optional extra `[gui]`); **two tabs** (Pipeline Inspector / Eval Dashboard); **preset export** (code defaults never change; eval baseline stays CLI-governed and is never written by the GUI).

## Current state (already done — do not redo)
- `pyproject.toml` already has `gui = ["streamlit>=1.37", "plotly>=5.22"]` and the `melody-extractor-gui = "melody_extractor.gui.launch:main"` script entry.
- streamlit 1.59.2 + plotly 6.9.0 are installed in the venv `C:\Users\Luke\.venvs\viobot`.
- `melody_extractor/gui/__init__.py` exists (comment-only stub).
- Nothing else: `config_io.py`, `eval_harness.py`, all other `gui/` files, presets dir, and all tests below **do not exist yet**; `cli.py` is unrefactored.

## Architecture

New files under `MelodyExtractor/`:

```
melody_extractor/
  config_io.py       # core-level, stdlib+config-dataclasses only (no streamlit)
  eval_harness.py    # core-level: run_eval(...) extracted from cli.py
  gui/
    __init__.py      # exists (stub)
    app.py           # entry: st.tabs, sidebar, dispatch
    params.py        # sidebar widgets -> frozen config dataclasses
    pipeline_cache.py# st.cache_data stage chain
    figures.py       # PURE plotly builders (no streamlit import)
    audio_bytes.py   # PURE: in-memory WAV bytes via scipy (D-013!)
    inspector_view.py
    eval_view.py
    launch.py        # console-script wrapper (arg-list launch — path has spaces)
presets/             # named preset JSONs + README.md
```

Hard rules: nothing outside `gui/` imports streamlit/plotly; `cli.py` imports only `config_io` + `eval_harness`; every file the GUI writes (presets, report download) is deterministic (sorted keys, LF, no timestamps/abs paths); WAV bytes via `scipy.io.wavfile` only (D-013).

### config_io.py
- `PRESET_SCHEMA_VERSION = "1"`; frozen `Preset(name, comment, mono: MonoConfig, timbre: TimbreConfig, stage: StageConfig, render: RenderConfig)`.
- `config_from_dict(cls, d)`: keep only `dataclasses.fields(cls)` keys, coerce lists→tuples where defaults are tuples (`open_strings_hz`), drop unknown keys (must swallow `reducer_version` from `StageConfig.config_dict()`), missing keys → field defaults.
- `preset_to_dict/from_dict` (major-version gate), `save_preset/load_preset` (json sort_keys, indent=2, allow_nan=False, +"\n", utf-8 LF — byte-deterministic).
- Preset JSON: `{"preset_schema_version": "1", "name", "comment", "configs": {"mono": {...}, "timbre": {...}, "stage": {...}, "render": {...}}}`.

### eval_harness.py (behavior-preserving refactor OUT of cli.py)
Move `_is_monophonic`, `_gt_frame_f0`, `_to_intervals_pitches`, `_top_voice_notes`, retention matching, aggregate computation, tolerance constants. Provide:
- `run_eval(fixtures_dir, mono_config=MonoConfig(), timbre_config=TimbreConfig(), stage_configs=None → {n: StageConfig.stage(n)}, progress=None) -> dict` — parametrizes `transcribe_mono` (both determinism runs), `add_harmonics`, stage loop. **With defaults the serialized report must be byte-identical to today's `cmd_eval` output — regression-test this.**
- `compare_to_baseline(aggregate, baseline_aggregate) -> (rows, rpa_drop)` shared by CLI + GUI.
- `cmd_eval` keeps arg parsing, report writing, baseline handling, exit codes (0/1/3) unchanged.

### CLI --config PATH (extract | reduce | render | eval)
Precedence: explicit flag > preset > code default. `extract`: preset.mono, `--backend` (default=None sentinel) overrides via `replace`; timbre pass uses preset.timbre. `reduce`: `replace(preset.stage, max_voices=args.stage)`. `render`: preset.render. `eval`: `stage_configs = {n: replace(preset.stage, max_voices=n)}`; **guardrail** — with a non-default preset and no existing baseline, write the report but do NOT auto-create a baseline (print why).

### Tab 1 — Pipeline Inspector
Sidebar: `st.file_uploader` (AUDIO+MIDI extensions) **plus a fixture selectbox** (`MelodyExtractor/tests/fixtures/*.wav`) — the selectbox exists so AppTest can drive the app (file_uploader can't be simulated). Per-config `st.expander` groups with sliders; widget defaults read programmatically from `dataclasses.fields(...)` so GUI defaults can't drift from code; "reset" buttons clear widget keys; preset save/load UI at the bottom (no silent overwrite).

Main column, collapsible per-stage sections (each captioned with module.function):
- **Input**: decimated min/max waveform (`go.Scattergl`, ≤8k pts) + STFT heatmap (nperseg 1024, noverlap 768, dB floor −80, y ≤ ~2.6 kHz).
- **FrameTrack** (`transcriber.transcribe_mono`): 3-row shared-x subplot — f0 in MIDI units (unvoiced masked to None), voicing with dashed hline at `voicing_threshold` + grey vrects over merged sub-threshold runs (cap 200), amp_db. Moving the threshold slider re-shades; only segmentation downstream recomputes.
- **Notes**: piano-roll — one `go.Scatter` trace with None separators (x=[on,off,None…], y=[midi,midi,None…], width ~8, color by confidence), `customdata`=note index; `st.plotly_chart(on_select="rerun")` note picking + a selectbox fallback (AppTest + accessibility).
- **Timbre** (selected note): harmonic dB `go.Bar`, tristimulus 3-segment stacked bar, `st.metric` row (odd/even, inharmonicity); `st.info` naming `frame_size` when `harmonics is None`.
- **Reducer**: stage radio (1/2/3) + advanced expander (all StageConfig fields + the 5 weights; `open_strings_hz` NOT exposed — D-015). Before/after piano roll: input grey, survivors colored by voice (rolled dashed), dropped red, trimmed-away portions red; `playability_violations` panel (st.error lines / green success); cost-formula markdown from the reducer docstring + per-dropped-note importance table (**display-only recompute of the documented formula — never call reducer internals**); surface `meta.extra["pruned"]` and `meta.stage`.
- **A/B audio**: 3 `st.audio` players (original / extracted render / reduced render), renders RMS-matched to original; "Export paired WAVs" button → `soundsim.render_paired`.

### Caching chain (pipeline_cache.py)
Frozen configs are hashable → direct `st.cache_data` keys. AudioBuffer (ndarray) cached via `st.cache_resource`, keyed by sha256 of file bytes; downstream functions take the digest (not the buffer) and re-fetch via the cached loader:
`load(digest) → transcribe(digest, mono_cfg) → timbre(digest, mono, timbre_cfg) → reduce(+stage_cfg) → render(+render_cfg, stage_cfg|None)`.
Effect: weight tweak recomputes reduce+render only (~ms); threshold tweak recomputes transcribe onward; new file invalidates all. `st.spinner` on cold YIN runs; `max_entries=32` on render bytes.

### Tab 2 — Eval Dashboard
Read-only header (fixtures dir, baseline path). Button → `run_eval` with current sidebar configs (`st.status` streaming via `progress` callback; cached on configs + fixture mtime digest; runs the pipeline twice per fixture — tens of seconds, acceptable). Aggregate table vs baseline with colored Δ (+ REGRESSION flag at RPA drop > 1 pt, same wording as CLI); per-fixture table (poly fixtures show "—" for melody metrics); caption explaining knob→metric mapping (stage configs act on ground-truth MIDI ⇒ transcriber-independent). GUI never writes baseline/report files; only `st.download_button` for the deterministic report + a note: adopt via preset + `melody-extractor eval --config` + decisions.md entry.

## Tests
- `test_config_io.py`: round-trip equality; unknown keys (feed `config_dict()` output); missing sections → defaults; version 2.x rejected; byte-identical saves; no absolute paths in bytes.
- `test_eval_harness.py`: **byte-equality of default `run_eval` vs CLI cmd path** (refactor guard); modified StageConfig changes only reducer entries.
- `test_gui_helpers.py` (`importorskip("plotly")`): each figure builder returns `go.Figure` with expected trace counts (use `tests/synth_util.py`); `diff_reduction` kept/trimmed/dropped on a hand-built trio; `importance_table` rows only for dropped notes, total = sum of terms; `wav_bytes` RIFF prefix + round-trip + byte-identical; `rms_matched` incl. silence guard.
- `test_cli.py` additions: `--config` on extract/reduce (meta.stage shows preset weights + overridden max_voices)/precedence/eval-no-baseline guardrail.
- `test_gui_apptest.py` (`importorskip("streamlit")`, mark slow): `AppTest.from_file(app.py, default_timeout=120)`, select fixture via the sidebar selectbox, tweak `voicing_threshold`, assert no exception + key elements. Known AppTest limits: no file_uploader, no chart-selection simulation (that's what the selectbox fallbacks are for), no audio inspection.

## Execution order for the implementing session (delegation-ready)
1. **Agent A** (sonnet): `config_io.py` + `presets/README.md` + `eval_harness.py` + `cli.py` refactor + `test_config_io.py`/`test_eval_harness.py`/`test_cli.py` extensions. Owns `cli.py`.
2. **Agent B** (sonnet, parallel with A): `gui/figures.py` + `gui/audio_bytes.py` + `test_gui_helpers.py`. No `cli.py` contact. (Full figure-function specs: see git-less history — the spec list is reproduced in the test section above and the Tab 1 description; signatures: `waveform_figure(pcm, sr)`, `spectrogram_figure(pcm, sr, fmax_hz=2600)`, `frame_track_figure(track, voicing_threshold)`, `pianoroll_figure(seq, track=None, selected=None)`, `timbre_figures(note) -> dict`, `diff_reduction(seq_in, seq_out) -> {kept,trimmed,dropped}` (pitch-identical + span rules, 1e-6 tolerances, greedy canonical matching), `reduction_figure(seq_in, seq_out, diff)`, `importance_table(seq_in, diff, config)`; `wav_bytes(pcm, sr)`, `rms_matched(render, original)`.)
3. **Agent C** (sonnet, after A+B): `app.py`, `params.py`, `pipeline_cache.py`, `inspector_view.py`, `eval_view.py`, `launch.py` + `test_gui_apptest.py`.
4. **Integration** (orchestrator): full pytest (baseline: 98 passed + 1 skipped before this work); headless launch check (`streamlit run ... --server.headless true` starts without traceback); `MelodyExtractor/CLAUDE.md` layout tree + gotcha ("gui/ imports streamlit/plotly — never import gui from core; GUI writes only presets"); `docs/decisions.md` **D-016** entry (GUI + preset `--config`; alternatives Gradio/PySide6/Jupyter considered; baseline write-access explicitly excluded).

## Verification checklist
- Full suite green, no regressions; `test_eval_harness` byte-equality passes.
- `melody-extractor eval --fixtures MelodyExtractor/tests/fixtures --report <tmp>` output unchanged vs `MelodyExtractor/out/eval.json` (modulo none — should be identical).
- `streamlit run MelodyExtractor/melody_extractor/gui/app.py` renders both tabs on a fixture; A/B players audible; weight slider updates reducer panel without re-transcribing (watch spinner).
- Two `save_preset` calls byte-identical; `extract --config` respects preset; `reduce --stage 2 --config` shows preset weights + max_voices=2 in `meta.stage`.
