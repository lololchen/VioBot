"""Schema contract tests: round-trip fidelity, deterministic serialization,
validation, version gating, MIDI export."""
from __future__ import annotations

import json

import pytest

from melody_extractor.schema import (
    SCHEMA_VERSION,
    AmpEnvelope,
    F0Contour,
    FrameTrack,
    Harmonics,
    Meta,
    Note,
    NoteSequence,
    hz_to_midi,
    midi_to_hz,
)


def _rich_sequence() -> NoteSequence:
    notes = (
        Note(
            pitch_hz=442.1,
            onset_s=0.5,
            duration_s=1.0,
            amp_db_envelope=AmpEnvelope(times_s=[0.0, 0.2, 1.0], amp_db=[-30.0, -12.5, -40.0]),
            velocity=90,
            confidence=0.93,
            voice=0,
            f0_contour=F0Contour(times_s=[0.0, 0.5, 1.0], f0_hz=[441.0, 443.0, 442.0]),
            harmonics=Harmonics(
                harmonic_amps_db=[-12.0, -18.0, -24.0, -30.0],
                odd_even_ratio=1.4,
                tristimulus=(0.6, 0.3, 0.1),
                inharmonicity=0.001,
            ),
        ),
        Note(
            pitch_hz=196.0,
            onset_s=0.0,
            duration_s=0.4,
            amp_db_envelope=AmpEnvelope.constant(-20.0, 0.4),
        ),
    )
    track = FrameTrack(hop_s=0.01, f0_hz=[0.0, 196.0, 196.5], voicing=[0.1, 0.9, 0.95], amp_db=[-80.0, -20.0, -19.5])
    meta = Meta(source="x.wav", source_kind="audio", sample_rate=16000, backends={"transcriber": "yin-0.1.0"})
    return NoteSequence(notes=notes, features=(track,), meta=meta)


def test_pitch_conversions_roundtrip():
    assert hz_to_midi(440.0) == pytest.approx(69.0)
    assert midi_to_hz(69.0) == pytest.approx(440.0)
    for hz in (196.0, 293.66, 659.25, 2637.0):
        assert midi_to_hz(hz_to_midi(hz)) == pytest.approx(hz, rel=1e-12)


def test_json_roundtrip_preserves_everything():
    seq = _rich_sequence().validate()
    text = seq.to_json()
    back = NoteSequence.from_json(text)
    assert back.to_json() == text
    assert back.notes[0].onset_s == 0.0  # canonical order: sorted by onset
    n = back.sorted().notes[1]
    assert n.pitch_hz == pytest.approx(442.1)
    assert n.harmonics.tristimulus == pytest.approx((0.6, 0.3, 0.1))
    assert n.f0_contour.f0_hz == pytest.approx((441.0, 443.0, 442.0))
    assert back.features[0].voicing == pytest.approx((0.1, 0.9, 0.95))
    assert back.meta.backends["transcriber"] == "yin-0.1.0"


def test_serialization_is_deterministic_and_order_insensitive():
    seq = _rich_sequence()
    shuffled = NoteSequence(notes=tuple(reversed(seq.notes)), features=seq.features, meta=seq.meta)
    assert seq.to_json() == shuffled.to_json()
    assert seq.to_json() == NoteSequence.from_json(seq.to_json()).to_json()


def test_schema_version_major_gate():
    d = json.loads(_rich_sequence().to_json())
    assert d["schema_version"] == SCHEMA_VERSION
    d["schema_version"] = "99.0.0"
    with pytest.raises(ValueError, match="incompatible schema_version"):
        NoteSequence.from_json_dict(d)
    with pytest.raises(ValueError, match="schema_version"):
        NoteSequence.from_json_dict({"notes": []})


@pytest.mark.parametrize("bad, match", [
    (dict(pitch_hz=-5.0), "pitch_hz"),
    (dict(duration_s=0.0), "duration_s"),
    (dict(onset_s=-1.0), "onset_s"),
    (dict(velocity=300), "velocity"),
    (dict(confidence=1.5), "confidence"),
])
def test_note_validation_rejects_bad_fields(bad, match):
    base = dict(pitch_hz=440.0, onset_s=0.0, duration_s=0.5,
                amp_db_envelope=AmpEnvelope.constant(-20.0, 0.5))
    base.update(bad)
    with pytest.raises(ValueError, match=match):
        Note(**base).validate()


def test_envelope_validation():
    with pytest.raises(ValueError, match="not sorted"):
        AmpEnvelope(times_s=[0.5, 0.0], amp_db=[-10, -10]).validate()
    with pytest.raises(ValueError, match="length mismatch"):
        AmpEnvelope(times_s=[0.0], amp_db=[-10, -10]).validate()
    with pytest.raises(ValueError, match="NaN"):
        AmpEnvelope(times_s=[0.0, 1.0], amp_db=[float("nan"), -10]).validate()


def test_frozen_immutability():
    seq = _rich_sequence()
    with pytest.raises(AttributeError):
        seq.notes[0].pitch_hz = 100.0
    assert isinstance(seq.notes, tuple)
    assert isinstance(seq.notes[0].amp_db_envelope.times_s, tuple)


def test_midi_export(tmp_path):
    import pretty_midi

    seq = _rich_sequence()
    out = tmp_path / "out.mid"
    seq.to_midi(out)
    pm = pretty_midi.PrettyMIDI(str(out))
    all_notes = [n for inst in pm.instruments for n in inst.notes]
    assert len(all_notes) == 2
    pitches = sorted(n.pitch for n in all_notes)
    assert pitches == [55, 69]  # G3, A4 (nearest semitone)
    bends = [b for inst in pm.instruments for b in inst.pitch_bends]
    assert bends, "442.1 Hz is ~8 cents sharp of A4 — expected a pitch bend"


def test_empty_sequence_roundtrip():
    seq = NoteSequence()
    assert NoteSequence.from_json(seq.to_json()).notes == ()
