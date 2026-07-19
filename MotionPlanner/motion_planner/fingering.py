"""Fingering planner — Viterbi DP over (string, finger) states per sonority (D-026).

Time steps are ONSET CLUSTERS (notes with equal onsets, eps-deduped). Notes
sustained from earlier clusters keep their assignment and constrain the new
one (fingers/strings occupied). The reducer already guarantees every sonority
is acoustically feasible on consecutive strings within the hand span, so
candidate enumeration here is a superset of its feasibility (same open-string
constants, same pitch tolerance — no plan-time dead ends).

Costs (weights in PlannerConfig; all PROVISIONAL until SysID/kinematics data):
- travel-time hinge  w_time · max(0, T_req − T_avail)²   per moved finger,
  T_req = max(trapezoid(Δz), trapezoid(Δx)) + t_lift + t_press,
  T_avail = (onset − finger release) + steal_fraction · previous duration;
- shift continuity   w_shift · (Δposition_st / 7)²        (Maezawa 2012, D-014);
- bow travel proxy   w_string · |Δband_index|;
- open-string bias   w_open per open-string note.

Deviations from a textbook Viterbi, both deliberate:
- Each DP node carries per-finger (position, string, release time) OF THE BEST
  PATH into it, so an idle finger can pre-position during intervening notes
  and is charged its true waiting time. This makes costs path-dependent (not
  strictly optimal); exact DP would need continuous finger positions in the
  state. Deterministic: ties break on the lexicographically smallest state key.
- Planner never hard-fails: when a cluster has no collision-free state
  (concept A shared rail, e.g. perfect fifths needing two fingers at one z),
  the first candidate ignoring the collision rule is used and an
  `infeasible_assignment` violation is recorded (D-026).

Concept dispatch happens in candidate enumeration only:
- A (roaming): any finger may take any string; engaged fingers on the shared
  rail must keep index order == z order with ≥ min_finger_separation_mm.
- B (per string): the string's dedicated finger is the only candidate; units
  sit on separate rails, so no cross-unit separation constraint.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from melody_extractor.schema import NoteSequence

from .config_io import PlannerConfig
from .hardware import (
    HardwareProfile,
    band_index,
    hz_to_position_st,
    st_to_mm,
    trapezoid_time,
)
from .schema import Violation

FINGERING_VERSION = "viterbi-fingering-0.1.0"

_EPS = 1e-6
_BIG = 1e6   # finite "infeasible" cost — keeps the DP total ordering meaningful


@dataclass(frozen=True)
class Candidate:
    """One way to play one note: string + finger (None = open) + position."""

    string: int
    finger: Optional[int]
    position_st: float


@dataclass(frozen=True)
class Transit:
    """Timing analysis of the finger move that starts this note."""

    t_req_s: float = 0.0
    t_avail_s: float = 0.0
    late_by_s: float = 0.0


@dataclass(frozen=True)
class FingerAssignment:
    note_index: int
    string: int
    finger: Optional[int]
    position_st: float
    position_mm: float
    transit: Transit = field(default_factory=Transit)
    violations: tuple = ()


@dataclass(frozen=True)
class FingeringPlan:
    assignments: dict            # note_index -> FingerAssignment
    total_cost: float
    version: str = FINGERING_VERSION


@dataclass(frozen=True)
class _FingerState:
    """Carried along the best path: where each finger last played."""

    position_st: float
    string: int
    release_t: float
    last_duration_s: float


def _spacing_mm(profile: HardwareProfile, position_st: float) -> float:
    """Cross-string spacing at a fingerboard position (linear nut→bridge)."""
    s = profile.strings
    frac = st_to_mm(position_st, s.scale_length_mm) / s.scale_length_mm
    return s.spacing_nut_mm + (s.spacing_bridge_mm - s.spacing_nut_mm) * frac


def _candidates_for_note(pitch_hz: float, profile: HardwareProfile,
                         config: PlannerConfig) -> "tuple[tuple[Candidate, ...], tuple[Violation, ...]]":
    """All (string, finger, position) options for one pitch. Falls back to a
    clamped assignment + violation when the pitch is out of range (D-026)."""
    tol = config.pitch_tolerance_semitones
    max_pos = profile.fingerboard.max_position_st
    out: "list[Candidate]" = []
    for s, open_hz in enumerate(profile.strings.open_hz):
        pos = hz_to_position_st(pitch_hz, open_hz)
        if pos < -tol or pos > max_pos + tol:
            continue
        if pos <= tol:
            out.append(Candidate(string=s, finger=None, position_st=0.0))
            continue
        clamped = min(max(pos, 0.0), max_pos)
        for f in profile.candidate_fingers_for_string(s):
            out.append(Candidate(string=s, finger=f, position_st=clamped))
    if out:
        return tuple(out), ()
    # Out of range for every string: clamp onto the outermost string.
    lo = hz_to_position_st(pitch_hz, profile.strings.open_hz[0])
    if lo < 0.0:
        cand = Candidate(string=0, finger=None, position_st=0.0)
        needed = lo
    else:
        s = len(profile.strings.open_hz) - 1
        pos = min(hz_to_position_st(pitch_hz, profile.strings.open_hz[s]), max_pos)
        f = profile.candidate_fingers_for_string(s)
        cand = Candidate(string=s, finger=f[0] if f else None, position_st=pos)
        needed = pos
    violation = Violation(kind="position_out_of_range", axis="fingerboard",
                          needed=float(needed), available=float(max_pos))
    return (cand,), (violation,)


def onset_clusters(seq: NoteSequence) -> "list[tuple[float, tuple[int, ...]]]":
    """Onset clusters over canonical note order: [(onset, note_indices)].
    Shared with bowing.py — both planners must slice time identically."""
    notes = seq.sorted().notes
    clusters: "list[tuple[float, list[int]]]" = []
    for i, n in enumerate(notes):
        if clusters and abs(n.onset_s - clusters[-1][0]) <= _EPS:
            clusters[-1][1].append(i)
        else:
            clusters.append((n.onset_s, [i]))
    return [(t, tuple(idx)) for t, idx in clusters]


def _rail_ok(engaged: "list[tuple[int, float]]", profile: HardwareProfile) -> bool:
    """Concept-A shared-rail check: engaged [(finger, z_mm)] must keep index
    order == z order with the minimum separation. Concept B always passes."""
    if profile.topology.concept != "A" or len(engaged) < 2:
        return True
    if profile.topology.fingers_can_cross:
        return True
    engaged = sorted(engaged)
    for (f1, z1), (f2, z2) in zip(engaged, engaged[1:]):
        if z2 - z1 < profile.topology.min_finger_separation_mm:
            return False
    return True


def _enumerate_states(note_idxs, notes, held_cands, profile, config):
    """All joint candidate tuples for a cluster, honoring held notes.

    held_cands: [(finger|None, string, z_mm)] for sustained notes. Returns
    (states, fallback): each state is a tuple of Candidate per cluster note in
    note_idxs order; fallback is the collision-ignoring first state (used with
    an infeasible_assignment violation when states is empty).
    """
    per_note: "list[tuple[tuple[Candidate, ...], tuple[Violation, ...]]]" = [
        _candidates_for_note(notes[i].pitch_hz, profile, config) for i in note_idxs]

    held_fingers = {f for f, _, _ in held_cands if f is not None}
    held_strings = {s for _, s, _ in held_cands}

    def joint(relax: int):
        """relax levels: 0 = all constraints; 1 = ignore rail separation;
        2 = + allow duplicate fingers (hardware short on fingers); 3 = + ignore
        held-note conflicts. Level 3 is guaranteed non-empty."""
        results = []

        def rec(k: int, chosen: "list[Candidate]"):
            if k == len(per_note):
                if relax < 1:
                    engaged = [(c.finger, st_to_mm(c.position_st, profile.strings.scale_length_mm))
                               for c in chosen if c.finger is not None]
                    engaged += [(f, z) for f, _, z in held_cands if f is not None]
                    if not _rail_ok(engaged, profile):
                        return
                results.append(tuple(chosen))
                return
            for cand in per_note[k][0]:
                if relax < 3:
                    if cand.string in held_strings or any(c.string == cand.string for c in chosen):
                        continue
                    if cand.finger is not None and relax < 2:
                        if cand.finger in held_fingers or any(c.finger == cand.finger for c in chosen):
                            continue
                rec(k + 1, chosen + [cand])

        rec(0, [])
        return results

    states = joint(relax=0)
    fallback = None
    for relax in (1, 2, 3):
        if states:
            break
        pool = joint(relax)
        if pool:
            fallback = min(pool, key=_state_key)
            break
    range_violations = {note_idxs[k]: vs for k, (_, vs) in enumerate(per_note) if vs}
    return states, fallback, range_violations


def _state_key(state: "tuple[Candidate, ...]") -> tuple:
    return tuple((c.string, -1 if c.finger is None else c.finger, c.position_st) for c in state)


def _cluster_band(state, held_cands) -> "int | None":
    """Inclination band proxy for the sounding strings (triples use the lower
    pair's band — the roll starts there, D-024)."""
    strings = sorted({c.string for c in state} | {s for _, s, _ in held_cands})
    if not strings:
        return None
    if len(strings) == 1:
        return band_index((strings[0],))
    return band_index((strings[0], strings[0] + 1))


def plan_fingering(seq: NoteSequence, profile: HardwareProfile,
                   config: PlannerConfig) -> FingeringPlan:
    """Assign (string, finger, position) to every note. Pure and deterministic."""
    notes = seq.sorted().notes
    if not notes:
        return FingeringPlan(assignments={}, total_cost=0.0)
    clusters = onset_clusters(seq)

    # DP nodes: state_key -> dict(cost, state, finger_state, band, back, details)
    # details: per note_index (Transit, violations) chosen on the edge into the node.
    layers: "list[dict]" = []
    prev_layer: "dict[tuple, dict]" = {
        (): {"cost": 0.0, "state": (), "fingers": {}, "band": None, "back": None, "details": {}}}

    for onset, note_idxs in clusters:
        # Held notes: sounding across this onset, assigned in earlier clusters.
        held_idx = [j for j in range(len(notes))
                    if notes[j].onset_s < onset - _EPS and notes[j].offset_s > onset + _EPS]
        layer: "dict[tuple, dict]" = {}
        for prev_key in sorted(prev_layer):
            prev = prev_layer[prev_key]
            held_cands = []
            for j in held_idx:
                a = _find_assignment(prev, j)
                if a is not None:
                    held_cands.append((a[1], a[0], st_to_mm(a[2], profile.strings.scale_length_mm)))
            states, fallback, range_violations = _enumerate_states(
                note_idxs, notes, held_cands, profile, config)
            forced_violation: tuple = ()
            if not states:
                if fallback is None:
                    continue
                states = [fallback]
                forced_violation = (Violation(kind="infeasible_assignment", axis="fingers"),)
            for state in sorted(states, key=_state_key):
                cost, fingers, details = _transition(
                    prev, state, note_idxs, notes, onset, profile, config)
                for note_i in note_idxs:
                    t, vs = details[note_i]
                    details[note_i] = (t, vs + forced_violation + range_violations.get(note_i, ()))
                band = _cluster_band(state, held_cands)
                if band is not None and prev["band"] is not None:
                    cost += config.w_string * abs(band - prev["band"])
                total = prev["cost"] + cost + (_BIG if forced_violation else 0.0)
                key = _state_key(state)
                node = {"cost": total, "state": state, "fingers": fingers,
                        "band": band if band is not None else prev["band"],
                        "back": (prev_key, prev), "details": details,
                        "order": note_idxs}
                if key not in layer or (total, key) < (layer[key]["cost"], key):
                    layer[key] = node
        layers.append({"note_idxs": note_idxs, "nodes": layer})
        prev_layer = layer

    # Backtrack from the cheapest final node (deterministic tie-break on key).
    final_key = min(prev_layer, key=lambda k: (prev_layer[k]["cost"], k))
    assignments: "dict[int, FingerAssignment]" = {}
    node = prev_layer[final_key]
    total_cost = node["cost"]
    for layer in reversed(layers):
        note_idxs = layer["note_idxs"]
        for i_pos, note_i in enumerate(note_idxs):
            cand = node["state"][i_pos]
            transit, violations = node["details"][note_i]
            assignments[note_i] = FingerAssignment(
                note_index=note_i, string=cand.string, finger=cand.finger,
                position_st=cand.position_st,
                position_mm=st_to_mm(cand.position_st, profile.strings.scale_length_mm),
                transit=transit, violations=violations)
        back = node["back"]
        node = back[1] if back else None
        if node is None:
            break
    return FingeringPlan(assignments=assignments, total_cost=total_cost)


def _find_assignment(node: dict, note_index: int):
    """Walk backpointers to find (string, finger, position_st) of a held note."""
    while node is not None:
        order = node.get("order", ())
        if note_index in order:
            cand = node["state"][order.index(note_index)]
            return (cand.string, cand.finger, cand.position_st)
        back = node.get("back")
        node = back[1] if back else None
    return None


def _transition(prev: dict, state, note_idxs, notes, onset,
                profile: HardwareProfile, config: PlannerConfig):
    """Edge cost prev-node -> state, plus updated finger states and per-note
    (Transit, violations) details."""
    cost = 0.0
    fingers: "dict[int, _FingerState]" = dict(prev["fingers"])
    details: "dict[int, tuple]" = {}
    for i_pos, note_i in enumerate(note_idxs):
        cand = state[i_pos]
        note = notes[note_i]
        violations: "list[Violation]" = []
        transit = Transit()
        if cand.finger is None:
            cost += config.w_open
        else:
            unit = profile.fingers[cand.finger]
            last = fingers.get(cand.finger)
            if last is None:
                # First use: the finger pre-positions from home (nut) before the
                # piece starts — no timing constraint, but the shift cost keeps
                # ties musical (open strings beat cold high positions).
                cost += config.w_shift * (cand.position_st / 7.0) ** 2
            else:
                dz_mm = abs(st_to_mm(cand.position_st, profile.strings.scale_length_mm)
                            - st_to_mm(last.position_st, profile.strings.scale_length_mm))
                dx_mm = abs(cand.string - last.string) * _spacing_mm(profile, cand.position_st)
                t_req = max(trapezoid_time(dz_mm / 1000.0, unit.z.v_max_mps, unit.z.a_max_mps2),
                            trapezoid_time(dx_mm / 1000.0, unit.x.v_max_mps, unit.x.a_max_mps2))
                t_req += unit.press.t_lift_s + unit.press.t_press_s
                t_avail = max(0.0, onset - last.release_t) \
                    + config.steal_fraction * last.last_duration_s
                late = max(0.0, t_req - t_avail)
                if late > _EPS:
                    cost += config.w_time * late * late
                    violations.append(Violation(
                        kind="late_transition", axis=f"f{cand.finger}",
                        needed=t_req, available=t_avail, late_by_s=late))
                shift_st = abs(cand.position_st - last.position_st)
                cost += config.w_shift * (shift_st / 7.0) ** 2
                transit = Transit(t_req_s=t_req, t_avail_s=t_avail, late_by_s=late)
            fingers[cand.finger] = _FingerState(
                position_st=cand.position_st, string=cand.string,
                release_t=note.offset_s, last_duration_s=note.duration_s)
        details[note_i] = (transit, tuple(violations))
    return cost, fingers, details
