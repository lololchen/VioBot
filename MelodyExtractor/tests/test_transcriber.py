"""Transcriber contract tests: YIN pitch accuracy, segmentation, envelope/contour
shape, determinism, and guarded optional-dependency behavior.

AudioBuffer is constructed directly from synth_util-synthesized numpy arrays
in every test — never via input_adapter.load_audio (that function is owned
by a different in-flight change)."""
from __future__ import annotations

import numpy as np
import pytest
import synth_util

from melody_extractor.input_adapter import AudioBuffer
from melody_extractor.schema import midi_to_hz
from melody_extractor.transcriber import MonoConfig, PolyConfig, transcribe_mono, transcribe_poly, yin_track

SR = 16000


def _buffer(pcm: np.ndarray, sr: int = SR, source: str = "") -> AudioBuffer:
    return AudioBuffer(pcm=pcm.astype(np.float32), sample_rate=sr, source=source)


# ---------------------------------------------------------------------------
# yin_track
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("freq_hz", [196.0, 293.66, 440.0, 659.25, 1318.5])
def test_yin_track_pitch_accuracy(freq_hz):
    tone = synth_util.harmonic_tone(freq_hz, 1.0, sample_rate=SR)
    f0, voicing, amp_db = yin_track(tone, SR, MonoConfig())

    assert len(f0) == len(voicing) == len(amp_db)
    voiced_mask = f0 > 0.0
    assert voiced_mask.sum() > 0, "expected at least some voiced frames for a steady tone"

    median_f0 = float(np.median(f0[voiced_mask]))
    assert median_f0 == pytest.approx(freq_hz, rel=0.003)

    # Middle third of the tone avoids the onset/offset fades: voicing should
    # be consistently high there.
    n = len(voicing)
    mid = voicing[n // 3 : 2 * n // 3]
    assert mid.size > 0
    assert float(np.mean(mid)) > 0.7


def test_yin_track_silence_is_unvoiced():
    silence = np.zeros(int(0.5 * SR), dtype=np.float64)
    f0, voicing, amp_db = yin_track(silence, SR, MonoConfig())

    assert np.all(f0 == 0.0)
    assert float(np.mean(voicing)) < 0.1
    assert np.all(amp_db <= -80.0 + 1e-6)


def test_yin_track_handles_short_and_empty_input():
    # Shorter than one analysis frame: must not crash (last-partial-frame edge case).
    tiny = synth_util.harmonic_tone(440.0, 0.005, sample_rate=SR)
    f0, voicing, amp_db = yin_track(tiny, SR, MonoConfig())
    assert len(f0) >= 1
    assert np.all(np.isfinite(f0))
    assert np.all(np.isfinite(voicing))
    assert np.all(np.isfinite(amp_db))

    empty = np.zeros(0, dtype=np.float64)
    f0e, voicinge, ampe = yin_track(empty, SR, MonoConfig())
    assert len(f0e) == len(voicinge) == len(ampe) == 0


# ---------------------------------------------------------------------------
# transcribe_mono: segmentation
# ---------------------------------------------------------------------------

def _c_major_scale_notes():
    # C4 major scale, 8 notes (incl. octave), semitone offsets from MIDI 60.
    offsets = [0, 2, 4, 5, 7, 9, 11, 12]
    freqs = [midi_to_hz(60 + o) for o in offsets]
    return [(f, i * 0.5, 0.5) for i, f in enumerate(freqs)]


def test_transcribe_mono_segments_scale_into_eight_notes():
    notes_spec = _c_major_scale_notes()
    pcm = synth_util.sequence_audio(notes_spec, sample_rate=SR)
    audio = _buffer(pcm, source="scale.wav")

    seq = transcribe_mono(audio, MonoConfig(backend="yin"))
    seq = seq.sorted()

    assert len(seq.notes) == 8
    for note, (freq, onset, _dur) in zip(seq.notes, notes_spec):
        cents = 1200.0 * np.log2(note.pitch_hz / freq)
        assert abs(cents) < 20.0, f"pitch off by {cents} cents at onset {onset}"
        assert abs(note.onset_s - onset) < 0.04, f"onset off by {note.onset_s - onset}s"


def test_transcribe_mono_does_not_merge_across_silence():
    notes_spec = [(440.0, 0.0, 0.3), (659.25, 0.5, 0.3)]  # 0.2 s silence between
    pcm = synth_util.sequence_audio(notes_spec, sample_rate=SR)
    audio = _buffer(pcm, source="two_notes.wav")

    seq = transcribe_mono(audio, MonoConfig(backend="yin"))
    assert len(seq.notes) == 2


# ---------------------------------------------------------------------------
# transcribe_mono: envelope / contour / confidence
# ---------------------------------------------------------------------------

def test_transcribe_mono_note_carries_envelope_and_contour():
    tone = synth_util.harmonic_tone(440.0, 0.4, sample_rate=SR)
    audio = _buffer(tone, source="tone.wav")

    seq = transcribe_mono(audio, MonoConfig(backend="yin"))
    assert len(seq.notes) >= 1

    note = seq.notes[0]
    assert len(note.amp_db_envelope.times_s) >= 2
    assert note.amp_db_envelope.times_s[0] == pytest.approx(0.0)
    assert note.amp_db_envelope.times_s[-1] == pytest.approx(note.duration_s)

    assert note.f0_contour is not None
    assert len(note.f0_contour.times_s) >= 1
    assert note.f0_contour.times_s[0] >= 0.0
    assert note.f0_contour.times_s[-1] <= note.duration_s + 1e-9

    assert note.confidence is not None
    assert 0.0 <= note.confidence <= 1.0


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------

def test_transcribe_mono_is_deterministic():
    pcm = synth_util.harmonic_tone(440.0, 0.5, sample_rate=SR)
    audio = _buffer(pcm, source="det.wav")

    seq1 = transcribe_mono(audio)
    seq2 = transcribe_mono(audio)
    assert seq1.to_json() == seq2.to_json()


def test_transcribe_mono_auto_backend_uses_yin_tag_without_crepe():
    # In this environment crepe is not installed, so backend="auto" must
    # silently fall back to yin rather than raising.
    try:
        import crepe  # noqa: F401
        pytest.skip("crepe is installed in this environment")
    except ImportError:
        pass

    pcm = synth_util.harmonic_tone(440.0, 0.3, sample_rate=SR)
    audio = _buffer(pcm)
    seq = transcribe_mono(audio, MonoConfig(backend="auto"))
    assert seq.meta.backends["transcriber"] == "yin-0.1.0"


# ---------------------------------------------------------------------------
# CREPE (only runs if the optional dep is installed)
# ---------------------------------------------------------------------------

def test_transcribe_mono_crepe_backend():
    pytest.importorskip("crepe")
    pcm = synth_util.harmonic_tone(440.0, 0.5, sample_rate=SR)
    audio = _buffer(pcm)
    seq = transcribe_mono(audio, MonoConfig(backend="crepe"))
    assert seq.meta.backends["transcriber"].startswith("crepe-")


def test_transcribe_mono_crepe_explicit_backend_raises_when_missing():
    try:
        import crepe  # noqa: F401
        pytest.skip("crepe is installed in this environment")
    except ImportError:
        pass

    pcm = synth_util.harmonic_tone(440.0, 0.2, sample_rate=SR)
    audio = _buffer(pcm)
    with pytest.raises(ImportError, match=r"melody-extractor\[mono-dnn\]"):
        transcribe_mono(audio, MonoConfig(backend="crepe"))


# ---------------------------------------------------------------------------
# transcribe_poly: guarded optional dependency
# ---------------------------------------------------------------------------

def test_transcribe_poly_raises_import_error_with_extras_hint():
    try:
        import basic_pitch  # noqa: F401
        pytest.skip("basic-pitch is installed in this environment")
    except ImportError:
        pass

    pcm = synth_util.harmonic_tone(440.0, 0.3, sample_rate=SR)
    audio = _buffer(pcm)
    with pytest.raises(ImportError, match=r"melody-extractor\[poly\]"):
        transcribe_poly(audio, PolyConfig())
