"""PURE plotly figure builders for the Pipeline Inspector tab (plan_GUI_MelodyExtractor.md).

Hard rule (module CLAUDE.md / plan "Architecture"): this module imports
plotly + numpy + core melody_extractor modules ONLY -- no streamlit import
anywhere here, so it stays unit-testable without a Streamlit runtime and
reusable from any future UI. `inspector_view.py` (Agent C) owns all
st.* calls and passes plain data in / plotly Figures out.

`importance_table` is a *display-only* recomputation of the cost formula
documented in reducer.py's module docstring (the emission-cost `importance`
term). It intentionally duplicates that small formula instead of importing
reducer's private helpers (`_pitch_ranks01`, `_change_points`, ...) per the
plan's explicit rule ("never call reducer internals/private functions") --
this keeps the reducer's DP internals a true implementation detail the GUI
cannot accidentally depend on, at the cost of the GUI's pitch-rank context
being "notes overlapping this note" rather than reducer's per-change-point
active set (documented on `_pitch_rank01` below).
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import stft

from ..reducer import StageConfig
from ..schema import FrameTrack, Note, NoteSequence, hz_to_midi

# ---------------------------------------------------------------------------
# Input panel: waveform + spectrogram
# ---------------------------------------------------------------------------

_MAX_WAVEFORM_POINTS = 8000
_DB_FLOOR = -80.0
_DB_EPS = 10.0 ** (_DB_FLOOR / 20.0)  # linear magnitude floor matching -80 dB

# Payload caps (D-018): every figure's per-trace point count is bounded so the
# websocket message stays small no matter how long the input audio is. A
# 5-minute song's raw STFT is ~3M cells (~50 MB of JSON) re-sent on EVERY
# Streamlit rerun -- enough to stall or drop the browser connection. Arrays
# are handed to plotly as numpy (not .tolist()) so plotly>=6 serializes them
# as compact base64 binary instead of JSON number text.
_MAX_SPEC_COLUMNS = 1000
_MAX_TRACK_POINTS = 4000


def waveform_figure(pcm: np.ndarray, sr: int) -> go.Figure:
    """Decimated min/max envelope waveform, one go.Scattergl trace, <= 8k points.

    For long signals, samples are grouped into up to 4000 time bins; each bin
    contributes two points (bin-center-time, min) and (bin-center-time, max),
    so the connected line traces a fast, faithful min/max envelope instead of
    plotting every sample.
    """
    pcm = np.asarray(pcm, dtype=np.float64)
    n = len(pcm)
    if n == 0:
        fig = go.Figure(data=[go.Scattergl(x=[], y=[], mode="lines", name="waveform")])
        fig.update_layout(xaxis_title="Time (s)", yaxis_title="Amplitude")
        return fig

    max_bins = _MAX_WAVEFORM_POINTS // 2
    if n <= _MAX_WAVEFORM_POINTS:
        x = np.arange(n, dtype=np.float64) / sr
        y = pcm
    else:
        n_bins = max_bins
        edges = np.linspace(0, n, n_bins + 1).astype(np.int64)
        xs = np.empty(n_bins * 2, dtype=np.float64)
        ys = np.empty(n_bins * 2, dtype=np.float64)
        for i in range(n_bins):
            lo, hi = int(edges[i]), max(int(edges[i + 1]), int(edges[i]) + 1)
            seg = pcm[lo:hi]
            center_sample = (lo + hi - 1) / 2.0
            xs[2 * i] = xs[2 * i + 1] = center_sample / sr
            ys[2 * i] = float(seg.min())
            ys[2 * i + 1] = float(seg.max())
        x = xs
        y = ys

    fig = go.Figure(data=[go.Scattergl(x=x, y=y, mode="lines", name="waveform")])
    fig.update_layout(xaxis_title="Time (s)", yaxis_title="Amplitude")
    return fig


def spectrogram_figure(pcm: np.ndarray, sr: int, fmax_hz: float = 2600.0) -> go.Figure:
    """STFT heatmap (nperseg=1024, noverlap=768), dB with floor -80, y <= fmax_hz.

    The time axis is max-pooled down to <= _MAX_SPEC_COLUMNS columns (max, not
    mean, so short note onsets stay visible) -- display-only decimation; the
    pipeline never reads this figure.
    """
    pcm = np.asarray(pcm, dtype=np.float64)
    nperseg = min(1024, max(2, len(pcm)))
    noverlap = min(768, nperseg - 1) if nperseg > 1 else 0
    freqs, times, Zxx = stft(pcm, fs=sr, nperseg=nperseg, noverlap=noverlap)
    mag = np.abs(Zxx)
    db = 20.0 * np.log10(np.maximum(mag, _DB_EPS))
    db = np.maximum(db, _DB_FLOOR)

    mask = freqs <= fmax_hz
    db = db[mask, :]
    freqs = freqs[mask]

    n_cols = db.shape[1]
    if n_cols > _MAX_SPEC_COLUMNS:
        pool = math.ceil(n_cols / _MAX_SPEC_COLUMNS)
        n_pad = (-n_cols) % pool
        if n_pad:
            db = np.concatenate([db, np.full((db.shape[0], n_pad), _DB_FLOOR)], axis=1)
            times = np.concatenate([times, np.full(n_pad, times[-1])])
        db = db.reshape(db.shape[0], -1, pool).max(axis=2)
        times = times.reshape(-1, pool).mean(axis=1)

    fig = go.Figure(data=[
        go.Heatmap(
            x=np.asarray(times, dtype=np.float64),
            y=np.asarray(freqs, dtype=np.float64),
            z=db.astype(np.float32),
            colorscale="Viridis",
            zmin=_DB_FLOOR,
            zmax=0.0,
            colorbar=dict(title="dB"),
        )
    ])
    fig.update_layout(xaxis_title="Time (s)", yaxis_title="Frequency (Hz)")
    fig.update_yaxes(range=[0.0, fmax_hz])
    return fig


# ---------------------------------------------------------------------------
# FrameTrack panel
# ---------------------------------------------------------------------------

_MAX_VREC_RUNS = 200


def frame_track_figure(track: FrameTrack, voicing_threshold: float) -> go.Figure:
    """3-row shared-x subplot: f0 (MIDI, unvoiced -> None) / voicing (+ dashed
    threshold hline + grey vrects over merged sub-threshold runs, capped at
    200) / amp_db.

    Long tracks are stride-decimated to <= _MAX_TRACK_POINTS plotted points
    per trace (display only); the vrect runs are still computed on the full-
    resolution voicing so no sub-threshold run is missed by decimation.
    """
    times = list(track.times_s())
    f0_midi = [hz_to_midi(f) if f > 0.0 else None for f in track.f0_hz]
    voicing = list(track.voicing)
    amp_db = list(track.amp_db)

    step = max(1, math.ceil(len(times) / _MAX_TRACK_POINTS))
    plot_times = times[::step]
    plot_f0 = f0_midi[::step]
    plot_voicing = voicing[::step]
    plot_amp = amp_db[::step]

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
        subplot_titles=("f0 (MIDI)", "voicing", "amp_db"),
    )
    fig.add_trace(go.Scatter(x=plot_times, y=plot_f0, mode="lines", name="f0 (MIDI)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_times, y=plot_voicing, mode="lines", name="voicing"), row=2, col=1)
    fig.add_trace(go.Scatter(x=plot_times, y=plot_amp, mode="lines", name="amp_db"), row=3, col=1)

    fig.add_hline(y=voicing_threshold, line_dash="dash", line_color="black", row=2, col=1)

    runs = _below_threshold_runs(voicing, voicing_threshold)[:_MAX_VREC_RUNS]
    hop_s = track.hop_s
    for i0, i1 in runs:
        x0 = times[i0]
        x1 = times[i1] + hop_s
        fig.add_vrect(x0=x0, x1=x1, fillcolor="grey", opacity=0.25, line_width=0, row=2, col=1)

    fig.update_xaxes(title_text="Time (s)", row=3, col=1)
    return fig


def _below_threshold_runs(values: "list[float]", threshold: float) -> "list[tuple[int, int]]":
    """Merge contiguous below-threshold frame indices into (start, end) runs."""
    runs: "list[tuple[int, int]]" = []
    start: "Optional[int]" = None
    for i, v in enumerate(values):
        below = v < threshold
        if below and start is None:
            start = i
        elif not below and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(values) - 1))
    return runs


# ---------------------------------------------------------------------------
# Piano roll (Notes panel + Reducer before/after panel)
# ---------------------------------------------------------------------------

def pianoroll_figure(seq: NoteSequence, track: Optional[FrameTrack] = None,
                      selected: Optional[int] = None) -> go.Figure:
    """Piano roll: ONE go.Scatter trace for all notes (None-separated segments,
    line width ~8, marker color = note confidence, customdata = note index),
    plus an optional f0-track overlay trace and an optional selected-note
    highlight trace.
    """
    notes = seq.sorted().notes
    xs: "list" = []
    ys: "list" = []
    customdata: "list" = []
    marker_colors: "list" = []
    for i, note in enumerate(notes):
        midi = hz_to_midi(note.pitch_hz)
        conf = note.confidence if note.confidence is not None else 1.0
        xs += [note.onset_s, note.offset_s, None]
        ys += [midi, midi, None]
        customdata += [i, i, None]
        # marker.color (unlike x/y) rejects None; NaN is the numeric
        # equivalent -- the point itself is invisible anyway since x/y is
        # None there, so the color value is moot.
        marker_colors += [conf, conf, float("nan")]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines+markers",
        line=dict(width=8, color="rgba(90,90,90,0.55)"),
        marker=dict(size=8, color=marker_colors, colorscale="Viridis", cmin=0.0, cmax=1.0,
                    showscale=True, colorbar=dict(title="confidence")),
        customdata=customdata,
        name="notes",
        hovertemplate="note %{customdata}<br>t=%{x:.3f}s<br>midi=%{y:.2f}<extra></extra>",
    ))

    if track is not None:
        times = list(track.times_s())
        f0_midi = [hz_to_midi(f) if f > 0.0 else None for f in track.f0_hz]
        step = max(1, math.ceil(len(times) / _MAX_TRACK_POINTS))
        fig.add_trace(go.Scatter(
            x=times[::step], y=f0_midi[::step], mode="lines",
            line=dict(width=1, color="rgba(0,0,0,0.5)"), name="f0",
        ))

    if selected is not None and 0 <= selected < len(notes):
        note = notes[selected]
        midi = hz_to_midi(note.pitch_hz)
        fig.add_trace(go.Scatter(
            x=[note.onset_s, note.offset_s], y=[midi, midi], mode="lines",
            line=dict(width=12, color="gold"), name="selected",
            customdata=[selected, selected],
        ))

    fig.update_layout(xaxis_title="Time (s)", yaxis_title="MIDI pitch")
    return fig


# ---------------------------------------------------------------------------
# Timbre panel
# ---------------------------------------------------------------------------

def timbre_figures(note: Note) -> dict:
    """Return {"harmonics_bar": go.Figure, "tristimulus_bar": go.Figure,
    "odd_even_ratio": float | None, "inharmonicity": float | None}.

    When note.harmonics is None (note too short for one analysis frame, per
    timbre.py), the figures are empty placeholders and the metrics are None;
    the caller (inspector_view.py) is expected to show an st.info naming
    frame_size in that case (plan Tab 1 "Timbre" bullet).
    """
    if note.harmonics is None:
        empty_harm = go.Figure()
        empty_harm.update_layout(xaxis_title="Harmonic #", yaxis_title="Amplitude (dB)")
        empty_tri = go.Figure()
        empty_tri.update_layout(yaxis_title="Energy share")
        return {
            "harmonics_bar": empty_harm,
            "tristimulus_bar": empty_tri,
            "odd_even_ratio": None,
            "inharmonicity": None,
        }

    h = note.harmonics
    ks = list(range(1, len(h.harmonic_amps_db) + 1))
    harmonics_fig = go.Figure(data=[go.Bar(x=ks, y=list(h.harmonic_amps_db), name="harmonic amplitude (dB)")])
    harmonics_fig.update_layout(xaxis_title="Harmonic #", yaxis_title="Amplitude (dB)")

    t1, t2, t3 = h.tristimulus
    tri_fig = go.Figure(data=[
        go.Bar(x=["tristimulus"], y=[t1], name="T1 (h1)"),
        go.Bar(x=["tristimulus"], y=[t2], name="T2 (h2-h4)"),
        go.Bar(x=["tristimulus"], y=[t3], name="T3 (h5+)"),
    ])
    tri_fig.update_layout(barmode="stack", yaxis_title="Energy share")

    return {
        "harmonics_bar": harmonics_fig,
        "tristimulus_bar": tri_fig,
        "odd_even_ratio": h.odd_even_ratio,
        "inharmonicity": h.inharmonicity,
    }


# ---------------------------------------------------------------------------
# Reducer before/after panel: diff classification + figure + importance table
# ---------------------------------------------------------------------------

_SPAN_TOL_S = 1e-6
_PITCH_REL_TOL = 1e-6  # the reducer never modifies pitch_hz; this only guards float noise


def diff_reduction(seq_in: NoteSequence, seq_out: NoteSequence) -> dict:
    """Classify each input note against the reduced output.

    {"kept": [...], "trimmed": [...], "dropped": [...]} where "kept" and
    "trimmed" entries are {"in_index": i, "out_index": j} (indices into
    seq_in.sorted().notes / seq_out.sorted().notes) and "dropped" entries are
    plain in_index ints (there is no surviving output note to reference).

    Matching rule: the reducer (reducer.py `reduce`) never changes a note's
    pitch_hz, only clips its onset/duration to a kept sub-run -- so a
    candidate match requires near-exact pitch equality AND the output note's
    [onset, offset) interval contained within the input note's interval
    (1e-6 s tolerance, matching reducer._EPS-scale comparisons). Same onset
    AND duration (within tolerance) -> "kept"; a strictly shorter span ->
    "trimmed"; no candidate at all -> "dropped".

    Matching is greedy over both note lists sorted canonically
    (onset_s, pitch_hz, duration_s), each output note claimed at most once,
    so ties between same-pitch notes resolve deterministically.
    """
    notes_in = list(seq_in.sorted().notes)
    notes_out = list(seq_out.sorted().notes)
    claimed = [False] * len(notes_out)

    kept: "list[dict]" = []
    trimmed: "list[dict]" = []
    dropped: "list[int]" = []

    for i, n_in in enumerate(notes_in):
        best_j = None
        for j, n_out in enumerate(notes_out):
            if claimed[j]:
                continue
            if abs(n_out.pitch_hz - n_in.pitch_hz) > _PITCH_REL_TOL * max(1.0, n_in.pitch_hz):
                continue
            if n_out.onset_s < n_in.onset_s - _SPAN_TOL_S or n_out.offset_s > n_in.offset_s + _SPAN_TOL_S:
                continue
            best_j = j
            break
        if best_j is None:
            dropped.append(i)
            continue
        claimed[best_j] = True
        n_out = notes_out[best_j]
        same_onset = abs(n_out.onset_s - n_in.onset_s) <= _SPAN_TOL_S
        same_duration = abs(n_out.duration_s - n_in.duration_s) <= _SPAN_TOL_S
        entry = {"in_index": i, "out_index": best_j}
        if same_onset and same_duration:
            kept.append(entry)
        else:
            trimmed.append(entry)

    return {"kept": kept, "trimmed": trimmed, "dropped": dropped}


_VOICE_PALETTE = ("#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b")


def reduction_figure(seq_in: NoteSequence, seq_out: NoteSequence, diff: dict) -> go.Figure:
    """Before/after piano roll: input grey, survivors colored by voice (rolled
    dashed), dropped red, trimmed-away portions red."""
    notes_in = list(seq_in.sorted().notes)
    notes_out = list(seq_out.sorted().notes)
    fig = go.Figure()

    bg_x: "list" = []
    bg_y: "list" = []
    for note in notes_in:
        midi = hz_to_midi(note.pitch_hz)
        bg_x += [note.onset_s, note.offset_s, None]
        bg_y += [midi, midi, None]
    fig.add_trace(go.Scatter(x=bg_x, y=bg_y, mode="lines",
                              line=dict(width=8, color="lightgrey"), name="input"))

    red_x: "list" = []
    red_y: "list" = []
    for i in diff["dropped"]:
        note = notes_in[i]
        midi = hz_to_midi(note.pitch_hz)
        red_x += [note.onset_s, note.offset_s, None]
        red_y += [midi, midi, None]
    for entry in diff["trimmed"]:
        note_in = notes_in[entry["in_index"]]
        note_out = notes_out[entry["out_index"]]
        midi = hz_to_midi(note_in.pitch_hz)
        if note_out.onset_s - note_in.onset_s > _SPAN_TOL_S:
            red_x += [note_in.onset_s, note_out.onset_s, None]
            red_y += [midi, midi, None]
        if note_in.offset_s - note_out.offset_s > _SPAN_TOL_S:
            red_x += [note_out.offset_s, note_in.offset_s, None]
            red_y += [midi, midi, None]
    if red_x:
        fig.add_trace(go.Scatter(x=red_x, y=red_y, mode="lines",
                                  line=dict(width=8, color="red"), name="dropped/trimmed"))

    groups: "dict[tuple[int, bool], list]" = {}
    for entry in list(diff["kept"]) + list(diff["trimmed"]):
        note_out = notes_out[entry["out_index"]]
        key = (note_out.voice if note_out.voice is not None else 0, note_out.rolled)
        groups.setdefault(key, []).append(note_out)

    for (voice, rolled), group_notes in sorted(groups.items()):
        gx: "list" = []
        gy: "list" = []
        for note in group_notes:
            midi = hz_to_midi(note.pitch_hz)
            gx += [note.onset_s, note.offset_s, None]
            gy += [midi, midi, None]
        color = _VOICE_PALETTE[voice % len(_VOICE_PALETTE)]
        fig.add_trace(go.Scatter(
            x=gx, y=gy, mode="lines",
            line=dict(width=8, color=color, dash="dash" if rolled else "solid"),
            name=f"voice {voice}" + (" (rolled)" if rolled else ""),
        ))

    fig.update_layout(xaxis_title="Time (s)", yaxis_title="MIDI pitch", title="Before / after reduction")
    return fig


def _overlaps(a: Note, b: Note) -> bool:
    """Half-open-interval overlap test for two notes."""
    return max(a.onset_s, b.onset_s) < min(a.offset_s, b.offset_s)


def _pitch_rank01(note: Note, pool: "list[Note]") -> float:
    """Percentile pitch rank (0=lowest .. 1=highest, 1.0 when alone) of `note`
    among the notes in `pool` that overlap it in time (including itself).

    This mirrors the *shape* of reducer.py's `_pitch_ranks01` (percentile
    rank among concurrently-active notes) without importing it. It is a
    display-only approximation: the reducer ranks per change-point interval
    (a note's rank can vary across its own span as neighbours start/stop),
    while this uses one rank for the note's whole span (its full-overlap
    group) -- adequate for an explanatory table, not for reproducing the
    reducer's DP bit-for-bit.
    """
    group = [n for n in pool if n is note or _overlaps(n, note)]
    if len(group) <= 1:
        return 1.0
    order = sorted(range(len(group)), key=lambda idx: (group[idx].pitch_hz, idx))
    ranks = {idx: rank / (len(order) - 1) for rank, idx in enumerate(order)}
    self_idx = next(idx for idx, n in enumerate(group) if n is note)
    return ranks[self_idx]


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def importance_table(seq_in: NoteSequence, diff: dict, config: StageConfig) -> "list[dict]":
    """Rows for dropped notes only: the reducer's documented importance
    formula (reducer.py module docstring), recomputed for display --
        importance(n) = w_amp * clip01((peak_db + 60) / 60)
                      + w_pitch * pitch_rank01(n)
                      + w_dur  * min(1.0, duration_s)
    Each row lists the individual terms plus "importance" == their sum.
    """
    notes = list(seq_in.sorted().notes)
    rows: "list[dict]" = []
    for i in diff["dropped"]:
        note = notes[i]
        amp_term = config.w_amp * _clip01((note.amp_db_envelope.peak_db() + 60.0) / 60.0)
        pitch_term = config.w_pitch * _pitch_rank01(note, notes)
        dur_term = config.w_dur * min(1.0, note.duration_s)
        rows.append({
            "note_index": i,
            "pitch_hz": note.pitch_hz,
            "onset_s": note.onset_s,
            "duration_s": note.duration_s,
            "amp_term": amp_term,
            "pitch_term": pitch_term,
            "dur_term": dur_term,
            "importance": amp_term + pitch_term + dur_term,
        })
    return rows
