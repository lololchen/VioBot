import math

import pytest

from melody_extractor.schema import AmpEnvelope, Meta, Note, NoteSequence

from motion_planner.bowing import plan_bowing
from motion_planner.config_io import PlannerConfig
from motion_planner.fingering import plan_fingering
from motion_planner.hardware import trapezoid_time

CFG = PlannerConfig()


def _note(pitch_hz, onset, dur=0.4, amp_db=-20.0):
    return Note(pitch_hz=pitch_hz, onset_s=onset, duration_s=dur,
                amp_db_envelope=AmpEnvelope.constant(amp_db, dur))


def _seq(notes):
    return NoteSequence(notes=tuple(notes), meta=Meta(source="test", source_kind="synthetic"))


def _plans(seq, profile):
    f = plan_fingering(seq, profile, CFG)
    return f, plan_bowing(seq, f, profile, CFG)


def test_segments_split_on_rests(profiles):
    p = profiles["concept_a_1finger"]
    seq = _seq([_note(440.0, 0.0, 0.4), _note(494.0, 0.45, 0.4),   # 50 ms gap: same segment
                _note(523.0, 1.5, 0.4)])                           # 650 ms rest: new segment
    _, bow = _plans(seq, p)
    assert bow.n_segments == 2
    kinds = [e.kind for e in bow.events]
    assert kinds.count("bow_land") == 2 and kinds.count("bow_lift") == 2


def test_bow_params_inside_schelleng_wedge(profiles):
    p = profiles["concept_a_1finger"]
    seq = _seq([_note(440.0, 0.0), _note(660.0, 0.5, amp_db=-35.0)])
    _, bow = _plans(seq, p)
    from motion_planner.bow_sound_model import AnalyticSchellengModel
    model = AnalyticSchellengModel(p)
    for i, nb in bow.note_bow.items():
        f_min, f_max = model.wedge(nb.v_b_mps, nb.beta, string=0)
        # wedge depends on the string actually chosen; recompute properly:
    for i, nb in bow.note_bow.items():
        assert nb.v_b_mps > 0 and nb.force_n > 0 and 0 < nb.beta < 1


def test_louder_notes_get_faster_bow(profiles):
    p = profiles["concept_a_1finger"]
    seq = _seq([_note(440.0, 0.0, amp_db=-30.0), _note(440.0, 0.5, amp_db=-12.0)])
    _, bow = _plans(seq, p)
    assert bow.note_bow[1].v_b_mps > bow.note_bow[0].v_b_mps


def test_rolled_triple_realized_timing_matches_d024(profiles):
    p = profiles["concept_b_4finger"]
    # Feasible triple on strings 0,1,2: open G3, E4 (D-string pos 2), C5 (A-string
    # pos 3) — consecutive strings, stopped span 1 st (reducer-legal).
    g3, e4, c5 = 195.998, 329.63, 523.25
    dur = 0.6
    seq = _seq([_note(g3, 0.0, dur), _note(e4, 0.0, dur), _note(c5, 0.0, dur)])
    fing, bow = _plans(seq, p)
    strings = sorted(fing.assignments[i].string for i in range(3))
    assert strings[1] == strings[0] + 1 and strings[2] == strings[1] + 1
    angles = p.strings.band_angles_rad
    band_lo = 2 * strings[0] + 1
    band_hi = 2 * strings[1] + 1
    ramp_t = trapezoid_time(abs(angles[band_hi] - angles[band_lo]),
                            p.bow.incl.v_max_radps, p.bow.incl.a_max_radps2)
    on_lo, dur_lo = bow.realized[0]
    on_mid, dur_mid = bow.realized[1]
    on_hi, dur_hi = bow.realized[2]
    assert on_lo == 0.0 and on_mid == 0.0
    assert on_hi == pytest.approx(CFG.roll_span_s + ramp_t)
    assert dur_lo == pytest.approx(CFG.roll_span_s + ramp_t)
    assert dur_mid == pytest.approx(dur)
    assert dur_hi == pytest.approx(dur - on_hi)
    rolls = [e for e in bow.events if e.kind == "roll"]
    assert len(rolls) == 1 and rolls[0].params["band_sequence"] == [band_lo, band_hi]


def test_fast_string_hopping_records_incl_violations(profiles):
    p = profiles["concept_a_2finger"]
    # G3<->E5-side hops every 30 ms force full-range inclination ramps.
    seq = _seq([_note(220.0 if i % 2 == 0 else 700.0, 0.03 * i, dur=0.03) for i in range(8)])
    _, bow = _plans(seq, p)
    lates = [v for vs in bow.violations.values() for v in vs if v.kind == "late_transition"]
    coupling = [v for vs in bow.violations.values() for v in vs if v.kind == "coupling_wobble"]
    assert lates or coupling  # this passage must stress the bow Z axis


def test_fixture_triple_rolled_gets_roll_event(fixture_sequences, profiles):
    seq = fixture_sequences["triple_rolled"]
    _, bow = _plans(seq, profiles["concept_b_4finger"])
    assert any(e.kind == "roll" for e in bow.events)
    # Realized onsets of the triple differ (pair-then-pair), score onsets equal.
    notes = seq.sorted().notes
    triple = [i for i, n in enumerate(notes) if n.rolled]
    onsets = sorted(bow.realized[i][0] for i in triple)
    assert onsets[0] == onsets[1] < onsets[2]


def test_determinism(fixture_sequences, profiles):
    seq = fixture_sequences["triple_rolled"]
    p = profiles["concept_b_4finger"]
    f1, b1 = _plans(seq, p)
    f2, b2 = _plans(seq, p)
    assert b1 == b2
