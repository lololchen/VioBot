"""Timbre module tests: analytic harmonic-recovery accuracy, purity,
determinism, short-note handling, and the guarded essentia backend."""
from __future__ import annotations

import numpy as np
import pytest
from synth_util import DEFAULT_SR, harmonic_tone

from melody_extractor.input_adapter import AudioBuffer
from melody_extractor.schema import AmpEnvelope, FrameTrack, Meta, Note, NoteSequence
from melody_extractor.timbre import TimbreConfig, add_harmonics

FREQ = 440.0
DUR = 1.0
# Ground truth recipe from synth_util.harmonic_tone: linear amp of harmonic k is 1/k.
TRUE_AMPS = tuple(1.0 / k for k in range(1, 7))


def _note_sequence(note: Note, **meta_kwargs) -> NoteSequence:
    meta = Meta(source="synthetic", source_kind="synthetic", sample_rate=DEFAULT_SR,
                backends={"transcriber": "yin-0.1.0"}, extra={"note": "test"}, **meta_kwargs)
    track = FrameTrack(hop_s=0.01, f0_hz=[FREQ, FREQ], voicing=[0.9, 0.9], amp_db=[-20.0, -20.0])
    return NoteSequence(notes=(note,), features=(track,), meta=meta)


def _tone_audio(freq=FREQ, dur=DUR, harmonic_amps=TRUE_AMPS, sr=DEFAULT_SR) -> AudioBuffer:
    pcm = harmonic_tone(freq, dur, sample_rate=sr, harmonic_amps=harmonic_amps)
    return AudioBuffer(pcm=pcm.astype(np.float32), sample_rate=sr, source="synthetic")


def _linear_amps(harmonics) -> list[float]:
    return [10.0 ** (db / 20.0) for db in harmonics.harmonic_amps_db]


def test_analytic_harmonic_recovery_440hz():
    audio = _tone_audio()
    note = Note(pitch_hz=FREQ, onset_s=0.0, duration_s=DUR,
                amp_db_envelope=AmpEnvelope.constant(-6.0, DUR))
    seq = _note_sequence(note)

    out = add_harmonics(audio, seq, TimbreConfig())
    harmonics = out.notes[0].harmonics
    assert harmonics is not None

    a = _linear_amps(harmonics)
    assert a[1] / a[0] == pytest.approx(1.0 / 2.0, rel=0.10)
    assert a[2] / a[0] == pytest.approx(1.0 / 3.0, rel=0.10)

    sq = [(1.0 / k) ** 2 for k in range(1, 7)]
    total = sum(sq)
    t1_exp = sq[0] / total
    t2_exp = (sq[1] + sq[2] + sq[3]) / total
    t3_exp = (sq[4] + sq[5]) / total
    assert harmonics.tristimulus[0] == pytest.approx(t1_exp, abs=0.05)
    assert harmonics.tristimulus[1] == pytest.approx(t2_exp, abs=0.05)
    assert harmonics.tristimulus[2] == pytest.approx(t3_exp, abs=0.05)

    odd_even_exp = (sq[2] + sq[4]) / (sq[1] + sq[3] + sq[5])
    assert harmonics.odd_even_ratio == pytest.approx(odd_even_exp, rel=0.15)

    assert harmonics.inharmonicity < 0.01


def test_note_shorter_than_one_frame_keeps_harmonics_none():
    # frame_size=4096 samples @ 16 kHz = 0.256 s; give the note far less than that.
    dur = 0.1
    audio = _tone_audio(dur=dur)
    note = Note(pitch_hz=FREQ, onset_s=0.0, duration_s=dur,
                amp_db_envelope=AmpEnvelope.constant(-6.0, dur))
    seq = _note_sequence(note)

    out = add_harmonics(audio, seq, TimbreConfig())
    assert out.notes[0].harmonics is None


def test_pure_sine_gives_near_ideal_tristimulus_and_low_odd_even():
    audio = _tone_audio(harmonic_amps=(1.0,))
    note = Note(pitch_hz=FREQ, onset_s=0.0, duration_s=DUR,
                amp_db_envelope=AmpEnvelope.constant(-6.0, DUR))
    seq = _note_sequence(note)

    out = add_harmonics(audio, seq, TimbreConfig())
    harmonics = out.notes[0].harmonics
    assert harmonics is not None

    assert harmonics.tristimulus[0] == pytest.approx(1.0, abs=0.02)
    assert harmonics.tristimulus[1] == pytest.approx(0.0, abs=0.02)
    assert harmonics.tristimulus[2] == pytest.approx(0.0, abs=0.02)
    assert harmonics.odd_even_ratio < 0.01


def test_purity_input_sequence_unchanged():
    audio = _tone_audio()
    note = Note(pitch_hz=FREQ, onset_s=0.0, duration_s=DUR,
                amp_db_envelope=AmpEnvelope.constant(-6.0, DUR))
    seq = _note_sequence(note)
    before = seq.to_json()

    result = add_harmonics(audio, seq, TimbreConfig())

    assert seq.to_json() == before
    assert result.notes[0].harmonics is not None
    # everything else preserved
    assert result.meta.source == seq.meta.source
    assert result.meta.source_kind == seq.meta.source_kind
    assert result.meta.sample_rate == seq.meta.sample_rate
    assert result.meta.extra == seq.meta.extra
    assert result.meta.backends["transcriber"] == "yin-0.1.0"
    assert result.meta.backends["timbre"] == "numpy-0.1.0"
    assert result.features == seq.features
    assert result.notes[0].pitch_hz == seq.notes[0].pitch_hz
    assert result.notes[0].onset_s == seq.notes[0].onset_s
    assert result.notes[0].duration_s == seq.notes[0].duration_s


def test_determinism_two_runs_identical_json():
    audio = _tone_audio()
    note = Note(pitch_hz=FREQ, onset_s=0.0, duration_s=DUR,
                amp_db_envelope=AmpEnvelope.constant(-6.0, DUR))
    seq = _note_sequence(note)

    out1 = add_harmonics(audio, seq, TimbreConfig())
    out2 = add_harmonics(audio, seq, TimbreConfig())
    assert out1.to_json() == out2.to_json()


def test_essentia_backend_raises_clear_import_error():
    audio = _tone_audio()
    note = Note(pitch_hz=FREQ, onset_s=0.0, duration_s=DUR,
                amp_db_envelope=AmpEnvelope.constant(-6.0, DUR))
    seq = _note_sequence(note)

    with pytest.raises(ImportError, match="essentia"):
        add_harmonics(audio, seq, TimbreConfig(backend="essentia"))
