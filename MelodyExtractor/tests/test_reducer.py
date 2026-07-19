"""Reducer contract tests: Viterbi voice selection, playability auditing,
purity/determinism, and output invariants from reducer.py's module docstring."""
from __future__ import annotations

import pytest

from melody_extractor.reducer import (
    REDUCER_VERSION,
    StageConfig,
    playability_violations,
    reduce,
    subset_feasible,
)
from melody_extractor.schema import AmpEnvelope, Meta, Note, NoteSequence


def _note(pitch_hz, onset_s, duration_s, peak_db=-20.0, **kw):
    return Note(
        pitch_hz=pitch_hz,
        onset_s=onset_s,
        duration_s=duration_s,
        amp_db_envelope=AmpEnvelope.constant(peak_db, duration_s),
        **kw,
    )


def _seq(notes, source="x.wav", backends=None):
    meta = Meta(source=source, source_kind="audio", sample_rate=16000, backends=dict(backends or {"transcriber": "yin-0.1.0"}))
    return NoteSequence(notes=tuple(notes), meta=meta)


def _by_pitch(seq, pitch_hz, tol=1e-6):
    matches = [n for n in seq.notes if abs(n.pitch_hz - pitch_hz) < tol]
    assert len(matches) == 1, f"expected exactly one note near {pitch_hz} Hz, found {len(matches)}"
    return matches[0]


# ---------------------------------------------------------------------------
# Stage 1: melody-over-harmony voice selection
# ---------------------------------------------------------------------------


def test_stage1_keeps_melody_over_lower_harmony():
    melody = [
        _note(440.00, 0.0, 1.0),   # A4
        _note(493.88, 1.0, 1.0),   # B4
        _note(523.25, 2.0, 1.0),   # C5
        _note(587.33, 3.0, 1.0),   # D5
    ]
    harmony = [
        _note(220.00, 0.0, 1.0),   # A3
        _note(246.94, 1.0, 1.0),   # B3
        _note(261.63, 2.0, 1.0),   # C4
        _note(293.66, 3.0, 1.0),   # D4
    ]
    seq = _seq(melody + harmony)
    out = reduce(seq, StageConfig.stage(1))

    assert len(out.notes) == 4
    out_pitches = sorted(n.pitch_hz for n in out.notes)
    melody_pitches = sorted(n.pitch_hz for n in melody)
    assert out_pitches == pytest.approx(melody_pitches, rel=1e-6)
    for n in out.notes:
        assert n.voice == 0
    assert playability_violations(out, StageConfig.stage(1)) == []


def test_stage1_pitch_weight_beats_small_amplitude_advantage():
    """Accompaniment louder (-18 dB) but lower-pitched; melody quieter (-20 dB)
    but higher-pitched. Default weights (w_pitch=w_amp=1.0) must still keep
    the top line."""
    high = _note(659.25, 0.0, 1.0, peak_db=-20.0)   # E5, melody
    low = _note(440.00, 0.0, 1.0, peak_db=-18.0)    # A4, louder accompaniment
    seq = _seq([high, low])
    out = reduce(seq, StageConfig.stage(1))

    assert len(out.notes) == 1
    assert out.notes[0].pitch_hz == pytest.approx(659.25)
    assert playability_violations(out, StageConfig.stage(1)) == []


# ---------------------------------------------------------------------------
# Stage 2: adjacent-string double stops
# ---------------------------------------------------------------------------


def test_stage2_feasible_pair_survives_intact():
    cfg = StageConfig.stage(2)
    assert subset_feasible((523.25, 659.25), cfg)  # sanity: C5+E5 is feasible

    c5 = _note(523.25, 0.0, 1.0)
    e5 = _note(659.25, 0.0, 1.0)
    seq = _seq([c5, e5])
    out = reduce(seq, cfg)

    assert len(out.notes) == 2
    pitches = sorted(n.pitch_hz for n in out.notes)
    assert pitches == pytest.approx([523.25, 659.25])
    for n in out.notes:
        assert n.onset_s == pytest.approx(0.0)
        assert n.duration_s == pytest.approx(1.0)
        assert n.rolled is False
    assert playability_violations(out, cfg) == []


def test_stage2_infeasible_pair_reduced_to_one_note():
    cfg = StageConfig.stage(2)
    assert not subset_feasible((196.0, 207.65), cfg)  # sanity: G3+G#3 infeasible

    g3 = _note(196.0, 0.0, 1.0)
    gs3 = _note(207.65, 0.0, 1.0)
    seq = _seq([g3, gs3])
    out = reduce(seq, cfg)

    assert len(out.notes) == 1
    # Same amplitude/duration: default weights favor the higher-pitched note.
    assert out.notes[0].pitch_hz == pytest.approx(207.65)
    assert playability_violations(out, cfg) == []


# ---------------------------------------------------------------------------
# Stage 3: rolled triples
# ---------------------------------------------------------------------------


def test_stage3_feasible_triple_kept_and_rolled():
    cfg = StageConfig.stage(3)
    triple = (392.0, 587.33, 987.77)
    assert subset_feasible(triple, cfg)  # sanity

    notes = [_note(p, 0.0, 1.0) for p in triple]
    seq = _seq(notes)
    out = reduce(seq, cfg)

    assert len(out.notes) == 3
    assert all(n.rolled for n in out.notes)
    assert sorted(n.pitch_hz for n in out.notes) == pytest.approx(sorted(triple))
    assert playability_violations(out, cfg) == []


def test_same_triple_at_stage1_keeps_only_top_note_unrolled():
    triple = (392.0, 587.33, 987.77)
    notes = [_note(p, 0.0, 1.0) for p in triple]
    seq = _seq(notes)
    out = reduce(seq, StageConfig.stage(1))

    assert len(out.notes) == 1
    assert out.notes[0].pitch_hz == pytest.approx(987.77)
    assert out.notes[0].rolled is False


# ---------------------------------------------------------------------------
# Out-of-range pitches never survive
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stage", [1, 2, 3])
def test_out_of_range_pitches_never_survive(stage):
    cfg = StageConfig.stage(stage)
    too_low = _note(150.0, 0.0, 1.0)
    too_high = _note(3000.0, 0.0, 1.0)
    in_range = _note(440.0, 0.0, 1.0)

    seq = _seq([too_low, too_high, in_range])
    out = reduce(seq, cfg)

    out_pitches = [n.pitch_hz for n in out.notes]
    assert all(abs(p - 150.0) > 1.0 for p in out_pitches)
    assert all(abs(p - 3000.0) > 1.0 for p in out_pitches)
    assert playability_violations(out, cfg) == []

    # Alone, out-of-range notes reduce to nothing at all.
    lone_seq = _seq([_note(150.0, 0.0, 1.0)])
    lone_out = reduce(lone_seq, cfg)
    assert lone_out.notes == ()


# ---------------------------------------------------------------------------
# Mid-note fragmentation resistance
# ---------------------------------------------------------------------------


def test_melody_note_not_fragmented_by_short_loud_flash():
    melody = _note(440.0, 0.0, 3.0, peak_db=-30.0)
    flash = _note(880.0, 1.0, 0.2, peak_db=-5.0)
    seq = _seq([melody, flash])
    out = reduce(seq, StageConfig.stage(1))

    melody_notes = [n for n in out.notes if abs(n.pitch_hz - 440.0) < 1.0]
    assert len(melody_notes) == 1, "melody note must survive as a single, unfragmented note"
    kept = melody_notes[0]
    assert kept.onset_s == pytest.approx(0.0)
    assert kept.duration_s == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Purity + determinism
# ---------------------------------------------------------------------------


def test_reduce_is_pure_and_deterministic():
    notes = [
        _note(440.0, 0.0, 1.0),
        _note(220.0, 0.0, 1.0),
        _note(523.25, 1.0, 1.0),
        _note(261.63, 1.0, 1.0),
    ]
    seq = _seq(notes)
    before = seq.to_json()

    out1 = reduce(seq, StageConfig.stage(1))
    after = seq.to_json()
    assert after == before, "reduce() must not mutate its input"

    out2 = reduce(seq, StageConfig.stage(1))
    assert out1.to_json() == out2.to_json(), "reduce() must be deterministic"


# ---------------------------------------------------------------------------
# playability_violations
# ---------------------------------------------------------------------------


def test_playability_violations_flags_hand_built_two_voice_sequence():
    cfg = StageConfig.stage(1)
    seq = _seq([_note(440.0, 0.0, 1.0), _note(220.0, 0.0, 1.0)])
    violations = playability_violations(seq, cfg)
    assert violations != []


# ---------------------------------------------------------------------------
# Empty sequence
# ---------------------------------------------------------------------------


def test_empty_sequence_reduces_to_empty_with_stage_meta():
    seq = NoteSequence()
    cfg = StageConfig.stage(2)
    out = reduce(seq, cfg)

    assert out.notes == ()
    assert out.meta.stage == cfg.config_dict()
    assert out.meta.backends.get("reducer") == REDUCER_VERSION
    assert playability_violations(out, cfg) == []
