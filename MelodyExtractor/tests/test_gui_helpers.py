"""Contract tests for the PURE gui helpers: figures.py (plotly builders,
diff/importance display logic) and audio_bytes.py (WAV bytes + RMS match).

These are Agent-B-owned files; this test module must not import cli.py,
config_io.py, eval_harness.py, or any other gui/ module (Agent A owns those,
concurrently).
"""
from __future__ import annotations

import io

import numpy as np
import pytest

plotly = pytest.importorskip("plotly")
import plotly.graph_objects as go  # noqa: E402
from scipy.io import wavfile  # noqa: E402

from melody_extractor.gui.audio_bytes import rms_matched, wav_bytes  # noqa: E402
from melody_extractor.gui.figures import (  # noqa: E402
    diff_reduction,
    frame_track_figure,
    importance_table,
    pianoroll_figure,
    reduction_figure,
    spectrogram_figure,
    timbre_figures,
    waveform_figure,
)
from melody_extractor.reducer import StageConfig  # noqa: E402
from melody_extractor.schema import (  # noqa: E402
    AmpEnvelope,
    FrameTrack,
    Harmonics,
    Meta,
    Note,
    NoteSequence,
)
from tests.synth_util import DEFAULT_SR, harmonic_tone  # noqa: E402

SR = DEFAULT_SR


def _note(pitch_hz=440.0, onset_s=0.0, duration_s=1.0, amp_db=-12.0, **kw):
    return Note(
        pitch_hz=pitch_hz,
        onset_s=onset_s,
        duration_s=duration_s,
        amp_db_envelope=AmpEnvelope.constant(amp_db, duration_s),
        **kw,
    )


def _seq(notes, source="x.wav"):
    meta = Meta(source=source, source_kind="audio", sample_rate=SR, backends={"transcriber": "yin-0.1.0"})
    return NoteSequence(notes=tuple(notes), meta=meta)


# ---------------------------------------------------------------------------
# waveform_figure / spectrogram_figure
# ---------------------------------------------------------------------------

def test_waveform_figure_short_signal_no_decimation():
    pcm = harmonic_tone(440.0, 0.05, SR)  # short -> under the 8k-point cap
    fig = waveform_figure(pcm, SR)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert isinstance(fig.data[0], go.Scattergl)
    assert len(fig.data[0].x) == len(pcm)


def test_waveform_figure_long_signal_decimated_under_cap():
    pcm = harmonic_tone(440.0, 5.0, SR)  # 80000 samples, well over the cap
    fig = waveform_figure(pcm, SR)
    assert len(fig.data) == 1
    assert isinstance(fig.data[0], go.Scattergl)
    assert len(fig.data[0].x) <= 8000
    # min/max envelope must bracket the true signal range
    assert min(fig.data[0].y) <= float(np.min(pcm)) + 1e-9
    assert max(fig.data[0].y) >= float(np.max(pcm)) - 1e-9


def test_spectrogram_figure_heatmap_respects_fmax():
    pcm = harmonic_tone(440.0, 1.0, SR)
    fmax = 2600.0
    fig = spectrogram_figure(pcm, SR, fmax_hz=fmax)
    assert len(fig.data) == 1
    assert isinstance(fig.data[0], go.Heatmap)
    assert max(fig.data[0].y) <= fmax + 1e-9
    # dB floor respected
    z = np.asarray(fig.data[0].z, dtype=np.float64)
    assert z.min() >= -80.0 - 1e-9


# ---------------------------------------------------------------------------
# frame_track_figure
# ---------------------------------------------------------------------------

def _make_track(n=40, hop_s=0.01, threshold=0.5):
    f0 = []
    voicing = []
    amp = []
    for i in range(n):
        if 10 <= i < 15:
            f0.append(0.0)
            voicing.append(0.1)
        else:
            f0.append(440.0)
            voicing.append(0.9)
        amp.append(-20.0)
    return FrameTrack(hop_s=hop_s, f0_hz=tuple(f0), voicing=tuple(voicing), amp_db=tuple(amp))


def test_frame_track_figure_three_rows_and_masking():
    track = _make_track()
    fig = frame_track_figure(track, voicing_threshold=0.5)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 3  # f0, voicing, amp_db

    f0_trace = fig.data[0]
    # unvoiced frames (indices 10..14) masked to None
    for i in range(10, 15):
        assert f0_trace.y[i] is None
    assert f0_trace.y[0] is not None

    # dashed hline present
    assert any(getattr(shp.line, "dash", None) == "dash" for shp in fig.layout.shapes)


def test_frame_track_figure_vrect_cap():
    # Alternate voiced/unvoiced single frames to build > 200 distinct runs.
    n = 500
    f0 = [440.0 if i % 2 == 0 else 0.0 for i in range(n)]
    voicing = [0.9 if i % 2 == 0 else 0.1 for i in range(n)]
    amp = [-20.0] * n
    track = FrameTrack(hop_s=0.01, f0_hz=tuple(f0), voicing=tuple(voicing), amp_db=tuple(amp))
    fig = frame_track_figure(track, voicing_threshold=0.5)
    # shapes = 1 hline + up to 200 vrects
    assert len(fig.layout.shapes) <= 201


# ---------------------------------------------------------------------------
# pianoroll_figure
# ---------------------------------------------------------------------------

def test_pianoroll_figure_single_trace_with_customdata():
    seq = _seq([
        _note(440.0, 0.0, 1.0, confidence=0.9),
        _note(523.25, 1.0, 0.5, confidence=0.4),
    ])
    fig = pianoroll_figure(seq)
    assert len(fig.data) == 1
    trace = fig.data[0]
    assert isinstance(trace, go.Scatter)
    # None-separated segments: on, off, None per note, x2 notes
    assert list(trace.x) == [0.0, 1.0, None, 1.0, 1.5, None]
    assert list(trace.customdata) == [0, 0, None, 1, 1, None]


def test_pianoroll_figure_with_track_and_selected_adds_traces():
    seq = _seq([_note(440.0, 0.0, 1.0, confidence=0.9)])
    track = _make_track()
    fig_with_track = pianoroll_figure(seq, track=track)
    assert len(fig_with_track.data) == 2

    fig_with_selection = pianoroll_figure(seq, track=track, selected=0)
    assert len(fig_with_selection.data) == 3


# ---------------------------------------------------------------------------
# timbre_figures
# ---------------------------------------------------------------------------

def test_timbre_figures_without_harmonics():
    note = _note()
    result = timbre_figures(note)
    assert set(result.keys()) == {"harmonics_bar", "tristimulus_bar", "odd_even_ratio", "inharmonicity"}
    assert isinstance(result["harmonics_bar"], go.Figure)
    assert len(result["harmonics_bar"].data) == 0
    assert result["odd_even_ratio"] is None
    assert result["inharmonicity"] is None


def test_timbre_figures_with_harmonics():
    harmonics = Harmonics(
        harmonic_amps_db=(-3.0, -9.0, -14.0, -20.0),
        odd_even_ratio=1.5,
        tristimulus=(0.6, 0.3, 0.1),
        inharmonicity=0.02,
    )
    note = _note(harmonics=harmonics)
    result = timbre_figures(note)
    assert len(result["harmonics_bar"].data) == 1
    assert isinstance(result["harmonics_bar"].data[0], go.Bar)
    assert list(result["harmonics_bar"].data[0].y) == list(harmonics.harmonic_amps_db)

    assert len(result["tristimulus_bar"].data) == 3
    stacked_values = [trace.y[0] for trace in result["tristimulus_bar"].data]
    assert stacked_values == pytest.approx(list(harmonics.tristimulus))

    assert result["odd_even_ratio"] == harmonics.odd_even_ratio
    assert result["inharmonicity"] == harmonics.inharmonicity


# ---------------------------------------------------------------------------
# diff_reduction
# ---------------------------------------------------------------------------

def _three_note_case():
    """A kept unchanged, B trimmed (shortened at the end), C dropped entirely."""
    note_a_in = _note(440.00, 0.0, 1.0)   # kept
    note_b_in = _note(523.25, 1.0, 1.0)   # trimmed to 0.5s
    note_c_in = _note(659.25, 2.0, 1.0)   # dropped

    note_a_out = _note(440.00, 0.0, 1.0)
    note_b_out = _note(523.25, 1.0, 0.5)

    seq_in = _seq([note_a_in, note_b_in, note_c_in])
    seq_out = _seq([note_a_out, note_b_out])
    return seq_in, seq_out


def test_diff_reduction_kept_trimmed_dropped_classification():
    seq_in, seq_out = _three_note_case()
    diff = diff_reduction(seq_in, seq_out)

    assert diff["kept"] == [{"in_index": 0, "out_index": 0}]
    assert diff["trimmed"] == [{"in_index": 1, "out_index": 1}]
    assert diff["dropped"] == [2]


def test_diff_reduction_span_tolerance_counts_as_kept():
    # onset/duration differ by less than the 1e-6s tolerance -> still "kept"
    note_in = _note(440.0, 0.0, 1.0)
    note_out = _note(440.0, 0.0, 1.0 - 1e-9)
    seq_in = _seq([note_in])
    seq_out = _seq([note_out])
    diff = diff_reduction(seq_in, seq_out)
    assert diff["kept"] == [{"in_index": 0, "out_index": 0}]
    assert diff["trimmed"] == []
    assert diff["dropped"] == []


# ---------------------------------------------------------------------------
# reduction_figure
# ---------------------------------------------------------------------------

def test_reduction_figure_builds_without_error():
    seq_in, seq_out = _three_note_case()
    diff = diff_reduction(seq_in, seq_out)
    fig = reduction_figure(seq_in, seq_out, diff)
    assert isinstance(fig, go.Figure)
    # background (input) + red (dropped/trimmed) + >=1 survivor group
    assert len(fig.data) >= 3


# ---------------------------------------------------------------------------
# importance_table
# ---------------------------------------------------------------------------

def test_importance_table_only_dropped_and_total_is_sum():
    seq_in, seq_out = _three_note_case()
    diff = diff_reduction(seq_in, seq_out)
    config = StageConfig()

    rows = importance_table(seq_in, diff, config)
    assert len(rows) == 1
    row = rows[0]
    assert row["note_index"] == 2
    assert row["importance"] == pytest.approx(row["amp_term"] + row["pitch_term"] + row["dur_term"])


def test_importance_table_empty_when_nothing_dropped():
    note_a_in = _note(440.0, 0.0, 1.0)
    note_a_out = _note(440.0, 0.0, 1.0)
    seq_in = _seq([note_a_in])
    seq_out = _seq([note_a_out])
    diff = diff_reduction(seq_in, seq_out)
    rows = importance_table(seq_in, diff, StageConfig())
    assert rows == []


# ---------------------------------------------------------------------------
# wav_bytes
# ---------------------------------------------------------------------------

def test_wav_bytes_riff_header_and_roundtrip():
    pcm = harmonic_tone(440.0, 0.2, SR, peak=0.5)
    data = wav_bytes(pcm, SR)
    assert data[:4] == b"RIFF"

    sr_read, arr = wavfile.read(io.BytesIO(data))
    assert sr_read == SR

    expected = np.round(np.clip(pcm, -1.0, 1.0) * 32767.0).astype(np.int16)
    assert np.array_equal(arr, expected)


def test_wav_bytes_byte_identical_across_calls():
    pcm = harmonic_tone(523.25, 0.15, SR, peak=0.3)
    data1 = wav_bytes(pcm, SR)
    data2 = wav_bytes(pcm, SR)
    assert data1 == data2


def test_wav_bytes_clips_out_of_range_samples():
    pcm = np.array([-2.0, 0.0, 2.0], dtype=np.float64)
    data = wav_bytes(pcm, SR)
    _, arr = wavfile.read(io.BytesIO(data))
    assert arr.tolist() == [-32767, 0, 32767]


# ---------------------------------------------------------------------------
# rms_matched
# ---------------------------------------------------------------------------

def test_rms_matched_scales_to_original_rms():
    rng_render = np.sin(np.linspace(0, 20 * np.pi, 4000)) * 0.1
    original = np.sin(np.linspace(0, 20 * np.pi, 4000)) * 0.8

    matched = rms_matched(rng_render, original)

    def _rms(x):
        return float(np.sqrt(np.mean(np.square(x))))

    assert _rms(matched) == pytest.approx(_rms(original), rel=1e-9)


def test_rms_matched_silence_guard_silent_original():
    render = np.sin(np.linspace(0, 20 * np.pi, 1000)) * 0.5
    original = np.zeros(1000)
    matched = rms_matched(render, original)
    assert np.array_equal(matched, render)


def test_rms_matched_silence_guard_silent_render():
    render = np.zeros(1000)
    original = np.sin(np.linspace(0, 20 * np.pi, 1000)) * 0.5
    matched = rms_matched(render, original)
    assert np.array_equal(matched, render)
    assert not np.any(np.isnan(matched))
