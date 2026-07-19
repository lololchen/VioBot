import math

import pytest

from melody_extractor.reducer import OPEN_STRINGS_HZ
from melody_extractor.schema import AmpEnvelope, Meta, Note, NoteSequence

from motion_planner.config_io import PlannerConfig
from motion_planner.fingering import plan_fingering
from motion_planner.hardware import hz_to_position_st


def _note(pitch_hz, onset, dur=0.4):
    return Note(pitch_hz=pitch_hz, onset_s=onset, duration_s=dur,
                amp_db_envelope=AmpEnvelope.constant(-20.0, dur))


def _seq(notes):
    return NoteSequence(notes=tuple(notes), meta=Meta(source="test", source_kind="synthetic"))


def _st(hz, string):
    return hz_to_position_st(hz, OPEN_STRINGS_HZ[string])


CFG = PlannerConfig()


def test_isolated_open_string_pitch_stays_open(profiles):
    plan = plan_fingering(_seq([_note(440.0, 0.0)]), profiles["concept_a_1finger"], CFG)
    a = plan.assignments[0]
    assert a.finger is None and a.string == 2 and a.position_st == 0.0


def test_open_string_series_assignments_reproduce_pitches(profiles):
    # Ascending fifths (the open-string series). Both the all-open fingering and
    # the position-7 lane are valid — assert pitch consistency, not the lane.
    seq = _seq([_note(OPEN_STRINGS_HZ[i], 0.5 * i) for i in range(4)])
    plan = plan_fingering(seq, profiles["concept_a_1finger"], CFG)
    for i in range(4):
        a = plan.assignments[i]
        realized_hz = OPEN_STRINGS_HZ[a.string] * 2.0 ** (a.position_st / 12.0)
        assert realized_hz == pytest.approx(OPEN_STRINGS_HZ[i], rel=1e-3)


def test_scale_prefers_low_positions_single_string_runs(profiles):
    # A4..E5 diatonic run: A4 is open A (or pos 0), the rest low positions on A.
    pitches = [440.0, 493.88, 523.25, 587.33, 659.26]
    seq = _seq([_note(p, 0.5 * i) for i, p in enumerate(pitches)])
    plan = plan_fingering(seq, profiles["concept_a_1finger"], CFG)
    for i in range(len(pitches)):
        a = plan.assignments[i]
        assert a.position_st <= 7.5, f"note {i} put at position {a.position_st}"
    # A4 opens the run: open A must beat a cold 7th-position D-string start.
    assert plan.assignments[0].finger is None and plan.assignments[0].string == 2


def test_double_stop_uses_adjacent_strings_distinct_fingers(profiles):
    # D5 + F#5 (a third): feasible on (A, E) strings.
    seq = _seq([_note(587.33, 0.0, 0.6), _note(739.99, 0.0, 0.6)])
    plan = plan_fingering(seq, profiles["concept_a_2finger"], CFG)
    a0, a1 = plan.assignments[0], plan.assignments[1]
    assert a1.string == a0.string + 1
    stopped = [a for a in (a0, a1) if a.finger is not None]
    assert len({a.finger for a in stopped}) == len(stopped)
    assert not a0.violations and not a1.violations


def test_concept_b_pins_finger_to_string(profiles):
    seq = _seq([_note(493.88, 0.0)])  # B4 -> A string pos 2 (or E string below range)
    plan = plan_fingering(seq, profiles["concept_b_4finger"], CFG)
    a = plan.assignments[0]
    assert a.finger == a.string


def test_fast_wide_leaps_get_late_transition_violations(profiles):
    # 8 alternating leaps G3<->high position at 25 ms spacing: one roaming
    # finger cannot lift-travel-press in time.
    lo, hi = 220.0, 660.0  # G-string pos ~2 vs ~19 (or other strings, still far)
    notes = [_note(lo if i % 2 == 0 else hi, 0.025 * i, dur=0.025) for i in range(8)]
    plan = plan_fingering(_seq(notes), profiles["concept_a_1finger"], CFG)
    lates = [v for a in plan.assignments.values() for v in a.violations
             if v.kind == "late_transition"]
    assert lates, "expected late_transition violations at 25 ms leap spacing"
    assert all(v.late_by_s > 0 for v in lates)


def test_two_fingers_beat_one_on_fast_alternation(profiles):
    # Same fast alternation: a second roaming finger can pre-position, so the
    # total lateness must drop (this is the Concept A-2 selling point).
    lo, hi = 246.94, 392.0
    notes = [_note(lo if i % 2 == 0 else hi, 0.06 * i, dur=0.06) for i in range(10)]
    seq = _seq(notes)
    late_1 = sum(a.transit.late_by_s for a in
                 plan_fingering(seq, profiles["concept_a_1finger"], CFG).assignments.values())
    late_2 = sum(a.transit.late_by_s for a in
                 plan_fingering(seq, profiles["concept_a_2finger"], CFG).assignments.values())
    assert late_2 <= late_1


def test_all_fixtures_get_complete_assignments(fixture_sequences, profiles):
    for name, seq in fixture_sequences.items():
        for pname, profile in profiles.items():
            plan = plan_fingering(seq, profile, CFG)
            assert set(plan.assignments) == set(range(len(seq.sorted().notes))), \
                f"{name} × {pname}: incomplete assignment"
            assert math.isfinite(plan.total_cost)
            for a in plan.assignments.values():
                pos = hz_to_position_st(seq.sorted().notes[a.note_index].pitch_hz,
                                        OPEN_STRINGS_HZ[a.string])
                assert a.position_st == pytest.approx(max(0.0, pos), abs=0.35)


def test_determinism_repeated_runs(fixture_sequences, profiles):
    seq = fixture_sequences["two_voice_thirds"]
    p = profiles["concept_a_2finger"]
    r1 = plan_fingering(seq, p, CFG)
    r2 = plan_fingering(seq, p, CFG)
    assert r1 == r2
