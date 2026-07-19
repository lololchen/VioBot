"""SoundSim contract tests: additive-backend synthesis correctness (pitch,
partial-amplitude ratios, glide/contour handling), byte-identical determinism,
the render_paired loudness-matching contract, empty-sequence handling, and the
guarded fluidsynth backend import.
"""
from __future__ import annotations

import hashlib
import time

import numpy as np
import pytest
from scipy.io import wavfile

from melody_extractor.input_adapter import AudioBuffer
from melody_extractor.schema import AmpEnvelope, F0Contour, Harmonics, Note, NoteSequence
from melody_extractor.soundsim import RenderConfig, render, render_paired, render_to_array

SR = 16000


def _note(pitch_hz=440.0, onset_s=0.0, duration_s=1.0, amp_db=-12.0, f0_contour=None, harmonics=None):
    return Note(
        pitch_hz=pitch_hz,
        onset_s=onset_s,
        duration_s=duration_s,
        amp_db_envelope=AmpEnvelope.constant(amp_db, duration_s),
        f0_contour=f0_contour,
        harmonics=harmonics,
    )


def _fft_mag(x: np.ndarray, sr: int):
    mags = np.abs(np.fft.rfft(x.astype(np.float64)))
    freqs = np.fft.rfftfreq(len(x), d=1.0 / sr)
    return freqs, mags


def _peak_near(freqs, mags, target_hz, tol_hz=3.0):
    window = (freqs >= target_hz - tol_hz) & (freqs <= target_hz + tol_hz)
    return float(np.max(mags[window]))


def _spectral_centroid(x: np.ndarray, sr: int) -> float:
    # Hann-window before the FFT: slicing a note into quarters cuts the
    # waveform at an arbitrary phase, and an unwindowed (rectangular) FFT of
    # that abrupt edge leaks broadband energy that dominates a full-band
    # magnitude-weighted centroid and swamps the real pitch-driven shift.
    # Windowing is the standard fix and makes the centroid track the actual
    # harmonic content instead of edge-truncation artifacts.
    window = np.hanning(len(x)) if len(x) > 1 else np.ones(len(x))
    freqs, mags = _fft_mag(x * window, sr)
    total = np.sum(mags)
    if total <= 0:
        return 0.0
    return float(np.sum(freqs * mags) / total)


# ---------------------------------------------------------------------------

def test_single_tone_pitch_and_amplitude(tmp_path):
    seq = NoteSequence(notes=(_note(pitch_hz=440.0, duration_s=1.0, amp_db=-12.0),))
    config = RenderConfig(sample_rate=SR, backend="additive")

    out = render(seq, tmp_path / "tone.wav", config)
    assert out.exists()
    sr_read, data = wavfile.read(str(out))
    assert sr_read == SR
    assert len(data) == round(1.0 * SR)

    arr = render_to_array(seq, config)
    assert len(arr) == round(1.0 * SR)

    freqs, mags = _fft_mag(arr, SR)
    dominant_freq = float(freqs[1:][np.argmax(mags[1:])])
    bin_width = SR / len(arr)
    assert abs(dominant_freq - 440.0) <= bin_width + 1e-9

    peak = float(np.max(np.abs(arr)))
    expected = 10.0 ** (-12.0 / 20.0)
    assert peak == pytest.approx(expected, rel=0.10)


def test_harmonics_partial_amplitude_ratios():
    harmonics = Harmonics(
        harmonic_amps_db=[-6.0, -12.0, -24.0],
        odd_even_ratio=1.0,
        tristimulus=(0.5, 0.3, 0.2),
        inharmonicity=0.0,
    )
    seq = NoteSequence(notes=(_note(pitch_hz=440.0, duration_s=1.0, amp_db=-6.0, harmonics=harmonics),))
    config = RenderConfig(sample_rate=SR, backend="additive")
    arr = render_to_array(seq, config)

    freqs, mags = _fft_mag(arr, SR)
    p1 = _peak_near(freqs, mags, 440.0)
    p2 = _peak_near(freqs, mags, 880.0)
    p3 = _peak_near(freqs, mags, 1320.0)

    ratio2_db = 20.0 * np.log10(p2 / p1)
    ratio3_db = 20.0 * np.log10(p3 / p1)
    assert ratio2_db == pytest.approx(-6.0, abs=2.0)
    assert ratio3_db == pytest.approx(-18.0, abs=2.0)


def test_f0_contour_glide_centroid_increases():
    contour = F0Contour(times_s=[0.0, 1.0], f0_hz=[440.0, 466.0])
    seq = NoteSequence(notes=(_note(pitch_hz=453.0, duration_s=1.0, amp_db=-6.0, f0_contour=contour),))
    config = RenderConfig(sample_rate=SR, backend="additive")
    arr = render_to_array(seq, config)

    n = len(arr)
    quarter = n // 4
    first_quarter = arr[:quarter]
    last_quarter = arr[-quarter:]

    centroid_first = _spectral_centroid(first_quarter, SR)
    centroid_last = _spectral_centroid(last_quarter, SR)
    assert centroid_last > centroid_first


def test_render_is_byte_identical_across_runs(tmp_path):
    """Two renders separated by >1s must produce byte-identical WAV files.

    The >1s gap specifically guards against libsndfile's PEAK-chunk timestamp
    (present when writing float WAVs via soundfile's subtype="FLOAT"): that
    chunk embeds a Unix timestamp, so two renders in different seconds would
    differ at that byte offset even though the audio samples are identical.
    soundsim writes via scipy.io.wavfile instead, which has no such chunk.
    """
    harmonics = Harmonics(
        harmonic_amps_db=[-3.0, -9.0, -15.0],
        odd_even_ratio=1.2,
        tristimulus=(0.6, 0.25, 0.15),
        inharmonicity=0.002,
    )
    contour = F0Contour(times_s=[0.0, 0.3, 0.6], f0_hz=[440.0, 445.0, 442.0])
    seq = NoteSequence(notes=(
        _note(pitch_hz=440.0, onset_s=0.0, duration_s=0.6, amp_db=-9.0, f0_contour=contour, harmonics=harmonics),
        _note(pitch_hz=293.66, onset_s=0.2, duration_s=0.5, amp_db=-15.0),
    ))
    config = RenderConfig(sample_rate=SR, backend="additive")

    path_a = render(seq, tmp_path / "a.wav", config)
    time.sleep(1.1)
    path_b = render(seq, tmp_path / "b.wav", config)

    bytes_a = path_a.read_bytes()
    bytes_b = path_b.read_bytes()
    assert bytes_a == bytes_b
    assert hashlib.sha256(bytes_a).hexdigest() == hashlib.sha256(bytes_b).hexdigest()

    # The in-memory arrays must also match exactly (no RNG / stateful drift).
    arr_a = render_to_array(seq, config)
    arr_b = render_to_array(seq, config)
    assert np.array_equal(arr_a, arr_b)


def test_render_paired_loudness_matched_and_original_sample_rate(tmp_path):
    orig_sr = 22050
    t = np.arange(int(0.8 * orig_sr), dtype=np.float64) / orig_sr
    orig_pcm = (0.3 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)
    original = AudioBuffer(pcm=orig_pcm, sample_rate=orig_sr, source="synthetic.wav")

    seq = NoteSequence(notes=(_note(pitch_hz=440.0, duration_s=0.8, amp_db=-6.0),))
    config = RenderConfig(sample_rate=44100, backend="additive")  # deliberately mismatched sr

    original_path, render_path = render_paired(original, seq, tmp_path / "pair", config)
    assert original_path.name == "original.wav"
    assert render_path.name == "extracted_render.wav"
    assert original_path.exists() and render_path.exists()

    sr_o, data_o = wavfile.read(str(original_path))
    sr_r, data_r = wavfile.read(str(render_path))
    assert sr_o == orig_sr
    assert sr_r == orig_sr  # render_paired must render at the ORIGINAL's rate, not config's

    rms_o = float(np.sqrt(np.mean(np.square(data_o.astype(np.float64)))))
    rms_r = float(np.sqrt(np.mean(np.square(data_r.astype(np.float64)))))
    assert rms_r == pytest.approx(rms_o, rel=0.01)


def test_render_paired_guards_silent_original(tmp_path):
    orig_sr = 16000
    original = AudioBuffer(pcm=np.zeros(1000, dtype=np.float32), sample_rate=orig_sr, source="silence.wav")
    seq = NoteSequence(notes=(_note(pitch_hz=440.0, duration_s=0.5, amp_db=-6.0),))
    config = RenderConfig(sample_rate=44100, backend="additive")

    # Must not raise (division by zero guarded) and must still produce both files.
    original_path, render_path = render_paired(original, seq, tmp_path / "pair_silent", config)
    assert original_path.exists() and render_path.exists()


def test_empty_note_sequence_renders_short_silent_file(tmp_path):
    seq = NoteSequence()
    config = RenderConfig(sample_rate=SR, backend="additive")

    out = render(seq, tmp_path / "empty.wav", config)
    assert out.exists()

    sr_read, data = wavfile.read(str(out))
    assert sr_read == SR
    assert 0 < len(data) <= round(0.5 * SR)  # short
    assert np.max(np.abs(data.astype(np.float64))) == 0.0

    arr = render_to_array(seq, config)
    assert arr.dtype == np.float32
    assert np.all(arr == 0.0)


def test_fluidsynth_backend_raises_import_error_with_hint():
    try:
        import fluidsynth  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("pyfluidsynth installed here: the ImportError path is unreachable")

    seq = NoteSequence(notes=(_note(),))
    config = RenderConfig(backend="fluidsynth", sample_rate=SR)

    with pytest.raises(ImportError, match="(?i)pyfluidsynth"):
        render_to_array(seq, config)

    with pytest.raises(ImportError, match="(?i)soundfont"):
        render_to_array(seq, config)


def test_fluidsynth_backend_requires_soundfont_path(monkeypatch, tmp_path):
    pytest.importorskip("fluidsynth")

    # Disable the default-soundfont discovery so the no-soundfont error path
    # is exercised even on machines that have one installed.
    monkeypatch.delenv("MELODY_EXTRACTOR_SF2", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))

    seq = NoteSequence(notes=(_note(),))
    config = RenderConfig(backend="fluidsynth", sample_rate=SR)  # soundfont_path=None

    with pytest.raises(ValueError, match="SoundFont"):
        render_to_array(seq, config)
