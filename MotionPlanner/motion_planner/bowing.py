"""Bow planner — contact segments, inclination trajectory, Schelleng speed/force,
rolled-chord timing (D-024).

Pipeline position: after fingering (needs string assignments), before
trajectory (emits events + per-note bow params the track builder samples).

Event vocabulary emitted here (schema.Event params):
- bow_land   {segment_id, y_to_m: 0.0}            at each segment start
- bow_lift   {segment_id, y_to_m}                 at each segment end
- bow_contact{segment_id, t_end_s, strings, band, band_angle_rad}  per band hold
- bow_incline{from_rad, to_rad, t_end_s}          ramps between holds
- roll       {segment_id, band_sequence, span_s}  D-024 pair-then-pair

Rolled-triple realization (D-024): dwell on the (low,mid) double-stop band for
config.roll_span_s, then sweep to (mid,high). Realized timing recorded per
note: low+mid onset at score onset; high onset delayed by roll_span_s + ramp;
low sounds until the sweep leaves its band; mid sustains throughout.

Inclination ramps are placed to END at the next cluster's onset, starting in
the inter-note gap, else stealing up to steal_fraction of the previous
cluster's shortest note. A ramp that cannot fit records a late_transition
violation on axis "bow.incl" (trajectory shifts realized onsets by it, D-026
soft-fail rule). During a ramp at rate ω the hair-string relative speed gains
±ω·r_contact; when that exceeds coupling_warn_fraction · v_b a
coupling_wobble violation is recorded on the following note(s).
"""
from __future__ import annotations

from dataclasses import dataclass

from melody_extractor.schema import NoteSequence

from .bow_sound_model import AnalyticSchellengModel
from .config_io import PlannerConfig
from .fingering import FingeringPlan, onset_clusters
from .hardware import HardwareProfile, band_index, trapezoid_time
from .schema import BowNotePlan, Event, VibratoPlan, Violation  # noqa: F401  (VibratoPlan for callers)

BOWING_VERSION = "schelleng-bow-0.1.0"

_EPS = 1e-6


@dataclass(frozen=True)
class BowingPlan:
    note_bow: dict                  # note_index -> BowNotePlan
    realized: dict                  # note_index -> (realized_onset_s, realized_duration_s)
    events: tuple                   # schema.Event, unsorted (MotionScore sorts canonically)
    violations: dict                # note_index -> tuple[Violation, ...]
    n_segments: int
    version: str = BOWING_VERSION


@dataclass
class _Hold:
    """One band-hold interval within a segment (mutable while building)."""

    t_start: float
    t_end: float
    band: int
    strings: tuple
    note_idxs: tuple


def _target_amp_db(note) -> float:
    """Representative loudness target: median of the envelope samples."""
    amps = sorted(note.amp_db_envelope.amp_db)
    return amps[len(amps) // 2]


def _brightness_u(note, config: PlannerConfig) -> float:
    if note.harmonics is not None:
        t3 = note.harmonics.tristimulus[2]
        return config.u_min + (config.u_max - config.u_min) * min(max(t3, 0.0), 1.0)
    return config.u_default


def _cluster_band_for(strings: "tuple[int, ...]") -> int:
    """Band for the sounding strings of a cluster; triples use the lower pair
    (the roll starts there, D-024). Non-adjacent fallback: lowest string."""
    ss = tuple(sorted(set(strings)))
    if len(ss) == 1:
        return band_index(ss)
    if len(ss) >= 2 and ss[1] == ss[0] + 1:
        return band_index((ss[0], ss[0] + 1))
    return band_index((ss[0],))


def plan_bowing(seq: NoteSequence, fingering: FingeringPlan, profile: HardwareProfile,
                config: PlannerConfig) -> BowingPlan:
    """Pure and deterministic. Returns per-note bow params, realized timing
    (D-024 rolls), the bow event list and per-note violations."""
    notes = seq.sorted().notes
    model = AnalyticSchellengModel(profile)
    note_bow: dict = {}
    realized: dict = {}
    violations: "dict[int, list]" = {i: [] for i in range(len(notes))}
    events: "list[Event]" = []
    if not notes:
        return BowingPlan(note_bow={}, realized={}, events=(), violations={}, n_segments=0)

    clusters = onset_clusters(seq)
    angles = profile.strings.band_angles_rad

    # ---- segment split: gap > lift_gap_s between consecutive sounding spans ----
    segment_of_cluster: "list[int]" = []
    seg = 0
    for k, (onset, idxs) in enumerate(clusters):
        if k > 0:
            prev_end = max(notes[i].offset_s for _, cl in clusters[:k] for i in cl)
            if onset - prev_end > config.lift_gap_s + _EPS:
                seg += 1
        segment_of_cluster.append(seg)
    n_segments = seg + 1

    # ---- per-note bow params + realized timing; build holds per cluster ----
    holds: "list[_Hold]" = []          # includes roll sub-holds, time-ordered
    rolls: "list[tuple]" = []          # (segment_id, cluster_k, band_lo, band_hi, onset)
    for k, (onset, idxs) in enumerate(clusters):
        seg_id = segment_of_cluster[k]
        cluster_strings = tuple(fingering.assignments[i].string for i in idxs)
        cluster_end = max(notes[i].offset_s for i in idxs)
        is_triple = len(idxs) == 3
        for i in idxs:
            a = fingering.assignments[i]
            beta = profile.beta_eff(a.position_st)
            u = _brightness_u(notes[i], config)
            inv = model.inverse(_target_amp_db(notes[i]), u, beta, a.string)
            if inv.speed_clipped:
                violations[i].append(Violation(
                    kind="speed_clipped", axis="bow.belt",
                    needed=inv.v_b_mps, available=profile.bow.belt.v_max_mps))
            if inv.force_clipped:
                violations[i].append(Violation(
                    kind="force_out_of_wedge", axis="bow.force",
                    needed=inv.force_n, available=profile.bow.force.f_max_n))
            note_bow[i] = BowNotePlan(segment_id=seg_id, v_b_mps=inv.v_b_mps,
                                      force_n=inv.force_n, beta=beta, u_brightness=u)
            realized[i] = (notes[i].onset_s, notes[i].duration_s)

        if is_triple:
            ss = sorted(set(cluster_strings))
            by_pitch = sorted(idxs, key=lambda i: notes[i].pitch_hz)
            if len(ss) == 3 and ss[1] == ss[0] + 1 and ss[2] == ss[1] + 1:
                lo_pair, hi_pair = (ss[0], ss[1]), (ss[1], ss[2])
                band_lo, band_hi = band_index(lo_pair), band_index(hi_pair)
            else:  # degenerate fallback assignment — roll collapses onto one band
                band_lo = band_hi = _cluster_band_for(cluster_strings)
                lo_pair = hi_pair = tuple(ss[:2]) if len(ss) >= 2 else (ss[0],)
            ramp_t = trapezoid_time(abs(angles[band_hi] - angles[band_lo]),
                                    profile.bow.incl.v_max_radps, profile.bow.incl.a_max_radps2)
            span = config.roll_span_s
            i_lo, i_mid, i_hi = by_pitch
            realized[i_lo] = (onset, max(span + ramp_t, 0.02))
            realized[i_mid] = (onset, notes[i_mid].duration_s)
            hi_onset = onset + span + ramp_t
            hi_dur = max(notes[i_hi].offset_s - hi_onset, 0.02)
            realized[i_hi] = (hi_onset, hi_dur)
            holds.append(_Hold(onset, onset + span, band_lo, lo_pair, (i_lo, i_mid)))
            if band_hi != band_lo:
                holds.append(_Hold(hi_onset, cluster_end, band_hi, hi_pair, (i_mid, i_hi)))
            rolls.append((seg_id, k, band_lo, band_hi, onset))
            events.append(Event(t_s=onset, kind="roll", params={
                "segment_id": seg_id, "band_sequence": [band_lo, band_hi],
                "span_s": span}))
        else:
            band = _cluster_band_for(cluster_strings)
            holds.append(_Hold(onset, cluster_end, band, tuple(sorted(set(cluster_strings))), idxs))

    # ---- inclination ramps between consecutive holds + lift/land at segment edges ----
    cluster_of_note = {}
    for k, (_, idxs) in enumerate(clusters):
        for i in idxs:
            cluster_of_note[i] = k

    def hold_segment(hold: _Hold) -> int:
        return segment_of_cluster[cluster_of_note[hold.note_idxs[0]]]

    for h in range(len(holds)):
        hold = holds[h]
        seg_id = hold_segment(hold)
        events.append(Event(t_s=hold.t_start, kind="bow_contact", params={
            "segment_id": seg_id, "t_end_s": hold.t_end,
            "strings": list(hold.strings), "band": hold.band,
            "band_angle_rad": angles[hold.band]}))
        if h == 0 or hold_segment(holds[h - 1]) != seg_id:
            events.append(Event(t_s=hold.t_start, kind="bow_land",
                                params={"segment_id": seg_id, "y_to_m": 0.0}))
        if h == len(holds) - 1 or hold_segment(holds[h + 1]) != seg_id:
            seg_end = max(hold.t_end, max(notes[i].offset_s for i in hold.note_idxs))
            events.append(Event(t_s=seg_end, kind="bow_lift",
                                params={"segment_id": seg_id,
                                        "y_to_m": profile.bow.y.travel_m / 4.0}))
        if h == 0:
            continue
        prev = holds[h - 1]
        if hold.band == prev.band:
            continue
        d_angle = angles[hold.band] - angles[prev.band]
        ramp_t = trapezoid_time(abs(d_angle), profile.bow.incl.v_max_radps,
                                profile.bow.incl.a_max_radps2)
        gap = hold.t_start - prev.t_end
        prev_min_dur = min(notes[i].duration_s for i in prev.note_idxs)
        window = max(gap, 0.0) + config.steal_fraction * prev_min_dur
        if ramp_t > window + _EPS:
            late = ramp_t - window
            for i in hold.note_idxs:
                violations[i].append(Violation(
                    kind="late_transition", axis="bow.incl",
                    needed=ramp_t, available=window, late_by_s=late))
        start = hold.t_start - min(ramp_t, window)
        events.append(Event(t_s=max(start, 0.0), kind="bow_incline", params={
            "from_rad": angles[prev.band], "to_rad": angles[hold.band],
            "t_end_s": hold.t_start}))
        # Inclination-rate/speed coupling: warn when ω·r spoils the note's v_b.
        if ramp_t > _EPS:
            omega = abs(d_angle) / ramp_t
            dv = omega * profile.bow.r_contact_m
            for i in hold.note_idxs:
                v_b = note_bow[i].v_b_mps
                if v_b > _EPS and dv > config.coupling_warn_fraction * v_b:
                    violations[i].append(Violation(
                        kind="coupling_wobble", axis="bow.incl",
                        needed=dv, available=config.coupling_warn_fraction * v_b))

    return BowingPlan(
        note_bow=note_bow,
        realized=realized,
        events=tuple(events),
        violations={i: tuple(v) for i, v in violations.items() if v},
        n_segments=n_segments,
    )
