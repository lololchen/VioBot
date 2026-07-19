"""Trajectory generation — events + assignments → fixed-hop task-space tracks +
FeasibilityReport (F4). This is the engine of the topology comparison: every
"can this hardware play this passage" number originates here.

Principles (D-023/D-026):
- Limits are never violated in the tracks; deadlines are. A move that cannot
  fit its window starts as early as allowed and ARRIVES LATE — the realized
  onset in note_plan shifts by the lateness, and the sim hears the delay.
- Sampling is closed-form (piecewise-linear breakpoints from trapezoid-timed
  moves; no ODE integration, no solver, no RNG) — float64, byte-deterministic.
- Realized onsets: bowing's D-024 roll timing, further shifted per note by the
  max late_by_s across its fingering/bowing violations.

Channel construction (v0 fidelity notes):
- bow.inclination_rad: band holds + ramps with their PHYSICAL trapezoid
  duration — an over-long ramp visibly overshoots its deadline in the track.
- bow.force_n / bow.speed_mps: per-note targets over realized intervals
  (overlaps take the max — one bow, one hair). Force rises over
  t_attack = max(guettler.t_attack_min_s, F/df_dt_max) at each note start.
- f{i}.x_m: string lane = string · spacing_bridge_mm (v0 constant spacing);
  f{i}.z_m: stopped position in metres from the nut + vibrato sinusoid
  (phase 0 at onset+delay); f{i}.press_n: trapezoidal press/lift ramps, press
  completes AT the realized onset.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from melody_extractor.schema import NoteSequence

from .bowing import BowingPlan
from .config_io import PlannerConfig
from .fingering import FingeringPlan
from .hardware import HardwareProfile, trapezoid_time
from .schema import Event, FeasibilityReport, Tracks

TRAJECTORY_VERSION = "trapezoid-0.1.0"

_EPS = 1e-9
_PRESS_LEVEL = 0.8       # press force as a fraction of the unit's f_max_n
_TAIL_S = 0.25           # track tail beyond the last release


@dataclass(frozen=True)
class TrajectoryResult:
    tracks: Tracks
    realized: dict               # note_index -> (realized_onset_s, realized_duration_s)
    finger_events: tuple         # press/lift/move events (bow events come from bowing)
    report: FeasibilityReport


class _Breakpoints:
    """Piecewise-linear channel: append (t, v) in time order, then sample."""

    def __init__(self, initial: float):
        self.ts: "list[float]" = [0.0]
        self.vs: "list[float]" = [initial]

    def add(self, t: float, v: float) -> None:
        t = max(t, self.ts[-1] + _EPS)
        self.ts.append(t)
        self.vs.append(v)

    def hold_until(self, t: float) -> None:
        self.add(t, self.vs[-1])

    def sample(self, times: "list[float]") -> "tuple[float, ...]":
        out = []
        k = 0
        for t in times:
            while k + 1 < len(self.ts) and self.ts[k + 1] <= t:
                k += 1
            if k + 1 >= len(self.ts) or t <= self.ts[0]:
                out.append(self.vs[min(k, len(self.vs) - 1)] if t >= self.ts[0] else self.vs[0])
                continue
            t0, t1 = self.ts[k], self.ts[k + 1]
            v0, v1 = self.vs[k], self.vs[k + 1]
            frac = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
            out.append(v0 + frac * (v1 - v0))
        return tuple(out)


def _note_lateness(i: int, fingering: FingeringPlan, bowing: BowingPlan) -> float:
    lates = [v.late_by_s for v in fingering.assignments[i].violations]
    lates += [v.late_by_s for v in bowing.violations.get(i, ())]
    return max(lates, default=0.0)


def build_trajectory(seq: NoteSequence, fingering: FingeringPlan, bowing: BowingPlan,
                     vibratos: dict, profile: HardwareProfile,
                     config: PlannerConfig, extra_violations: "dict | None" = None,
                     ) -> TrajectoryResult:
    """vibratos: note_index -> Optional[VibratoPlan] (already axis-clipped).
    extra_violations: note_index -> tuple[Violation] merged into the report
    (vibrato clips and other planner-level findings; they add no lateness)."""
    notes = seq.sorted().notes
    hop = profile.timing.control_hop_s

    # ---- realized timing: bowing (D-024) + lateness shifts ----
    realized: dict = {}
    for i in range(len(notes)):
        onset_r, dur_r = bowing.realized[i]
        late = _note_lateness(i, fingering, bowing)
        offset_r = onset_r + dur_r
        new_onset = onset_r + late
        realized[i] = (new_onset, max(offset_r - new_onset, 0.02))

    t_end = max((on + du for on, du in realized.values()), default=0.0) + _TAIL_S
    n_samples = int(math.ceil(t_end / hop)) + 1
    times = [k * hop for k in range(n_samples)]

    channels: dict = {}
    lifted_y = profile.bow.y.travel_m / 4.0

    # ---- bow inclination + y from bowing events ----
    incl = _Breakpoints(initial=_first_band_angle(bowing, profile))
    y_move_t = trapezoid_time(lifted_y, profile.bow.y.v_max_mps, profile.bow.y.a_max_mps2)
    # Pre-positioning rule: if the first landing has no room for its approach
    # ramp, the bow starts on the string (set up before the piece, like fingers).
    lands = [e.t_s for e in bowing.events if e.kind == "bow_land"]
    y0 = 0.0 if lands and min(lands) < y_move_t else lifted_y
    y = _Breakpoints(initial=y0)
    for e in sorted(bowing.events, key=lambda e: e.sort_key()):
        if e.kind == "bow_incline":
            phys = trapezoid_time(abs(e.params["to_rad"] - e.params["from_rad"]),
                                  profile.bow.incl.v_max_radps, profile.bow.incl.a_max_radps2)
            incl.hold_until(e.t_s)
            incl.add(e.t_s + phys, e.params["to_rad"])
        elif e.kind == "bow_contact":
            incl.hold_until(max(e.t_s - _EPS, 0.0))
            incl.add(e.t_s, e.params["band_angle_rad"])
            incl.hold_until(e.params["t_end_s"])
        elif e.kind == "bow_land":
            y.hold_until(max(e.t_s - y_move_t, 0.0))
            y.add(e.t_s, 0.0)
        elif e.kind == "bow_lift":
            y.hold_until(e.t_s)
            y.add(e.t_s + y_move_t, e.params.get("y_to_m", lifted_y))
    channels["bow.inclination_rad"] = incl.sample(times)
    channels["bow.y_m"] = y.sample(times)

    # ---- bow force / speed / beta per note interval (max-combine overlaps) ----
    force = [0.0] * n_samples
    speed = [0.0] * n_samples
    beta = [profile.bow.beta_default] * n_samples
    for i in sorted(realized):
        nb = bowing.note_bow[i]
        on, du = realized[i]
        k0 = min(max(int(math.ceil(on / hop)), 0), n_samples - 1)
        k1 = min(int(math.floor((on + du) / hop)), n_samples - 1)
        t_attack = max(profile.bow.guettler.t_attack_min_s,
                       nb.force_n / max(profile.bow.force.df_dt_max_nps, _EPS))
        for k in range(k0, k1 + 1):
            t_rel = times[k] - on
            rise = min(t_rel / t_attack, 1.0) if t_attack > 0 else 1.0
            force[k] = max(force[k], nb.force_n * rise)
            speed[k] = max(speed[k], nb.v_b_mps)
            beta[k] = nb.beta
    channels["bow.force_n"] = tuple(force)
    channels["bow.speed_mps"] = tuple(speed)
    channels["bow.beta"] = tuple(beta)

    # ---- fingers ----
    finger_events: "list[Event]" = []
    spacing_m = profile.strings.spacing_bridge_mm / 1000.0
    for fi, unit in enumerate(profile.fingers):
        assigned = sorted((i for i, a in fingering.assignments.items() if a.finger == fi),
                          key=lambda i: realized[i][0])
        if not assigned:
            continue
        first = fingering.assignments[assigned[0]]
        zbp = _Breakpoints(initial=first.position_mm / 1000.0)
        xbp = _Breakpoints(initial=first.string * spacing_m)
        press = _Breakpoints(initial=0.0)
        press_force = _PRESS_LEVEL * unit.press.f_max_n
        prev_release = 0.0
        prev = None
        for i in assigned:
            a = fingering.assignments[i]
            on, du = realized[i]
            z_target = a.position_mm / 1000.0
            x_target = a.string * spacing_m
            if prev is not None:
                move_start = max(prev_release, 0.0)
                z_time = trapezoid_time(abs(z_target - zbp.vs[-1]), unit.z.v_max_mps, unit.z.a_max_mps2)
                x_time = trapezoid_time(abs(x_target - xbp.vs[-1]), unit.x.v_max_mps, unit.x.a_max_mps2)
                move_t = max(z_time, x_time)
                start = max(move_start, on - unit.press.t_press_s - move_t)
                zbp.hold_until(start)
                zbp.add(start + z_time, z_target)
                xbp.hold_until(start)
                xbp.add(start + x_time, x_target)
                if abs(z_target - zbp.vs[-2]) > 1e-6:
                    finger_events.append(Event(t_s=start, kind="finger_move", params={
                        "finger": fi, "axis": "z", "from_m": zbp.vs[-2], "to_m": z_target,
                        "t_end_s": start + z_time}))
                if abs(x_target - xbp.vs[-2]) > 1e-6:
                    finger_events.append(Event(t_s=start, kind="finger_move", params={
                        "finger": fi, "axis": "x", "from_m": xbp.vs[-2], "to_m": x_target,
                        "t_end_s": start + x_time}))
            press_start = max(on - unit.press.t_press_s, prev_release, 0.0)
            press.hold_until(press_start)
            press.add(on, press_force)
            press.hold_until(on + du)
            press.add(on + du + unit.press.t_lift_s, 0.0)
            finger_events.append(Event(t_s=press_start, kind="finger_press", params={
                "finger": fi, "string": a.string, "position_st": a.position_st,
                "force_n": press_force, "t_ramp_s": unit.press.t_press_s}))
            finger_events.append(Event(t_s=on + du, kind="finger_lift", params={"finger": fi}))
            prev_release = on + du + unit.press.t_lift_s
            prev = a
        z_samples = list(zbp.sample(times))
        # Vibrato overlay: sinusoid in z during each vibrato note's sounding span.
        for i in assigned:
            vib = vibratos.get(i)
            if vib is None:
                continue
            a = fingering.assignments[i]
            on, du = realized[i]
            from .hardware import mm_per_st
            dz_m = (vib.depth_cents / 100.0) * mm_per_st(a.position_st,
                                                         profile.strings.scale_length_mm) / 1000.0
            t0 = on + vib.delay_s
            for k in range(n_samples):
                t = times[k]
                if t0 <= t <= on + du:
                    z_samples[k] += dz_m * math.sin(2.0 * math.pi * vib.rate_hz * (t - t0))
        channels[f"f{fi}.z_m"] = tuple(z_samples)
        channels[f"f{fi}.x_m"] = xbp.sample(times)
        channels[f"f{fi}.press_n"] = press.sample(times)

    # Fingers with no notes still get flat channels (sim + firmware want them).
    for fi in range(len(profile.fingers)):
        if f"f{fi}.z_m" not in channels:
            channels[f"f{fi}.z_m"] = tuple([0.0] * n_samples)
            channels[f"f{fi}.x_m"] = tuple([0.0] * n_samples)
            channels[f"f{fi}.press_n"] = tuple([0.0] * n_samples)

    tracks = Tracks(hop_s=hop, start_s=0.0, channels=channels)
    report = _feasibility_report(seq, fingering, bowing, extra_violations or {},
                                 realized, tracks, profile)
    return TrajectoryResult(tracks=tracks, realized=realized,
                            finger_events=tuple(finger_events), report=report)


def _first_band_angle(bowing: BowingPlan, profile: HardwareProfile) -> float:
    contacts = [e for e in bowing.events if e.kind == "bow_contact"]
    if not contacts:
        return profile.strings.band_angles_rad[len(profile.strings.band_angles_rad) // 2]
    first = min(contacts, key=lambda e: e.sort_key())
    return first.params["band_angle_rad"]


def _percentile(sorted_vals: "list[float]", q: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(int(q * (len(sorted_vals) - 1)), len(sorted_vals) - 1)
    return sorted_vals[idx]


def _axis_stats(values, hop: float, v_limit: float, a_limit: float) -> dict:
    v = [abs(values[k + 1] - values[k]) / hop for k in range(len(values) - 1)]
    a = [abs(v[k + 1] - v[k]) / hop for k in range(len(v) - 1)] if len(v) > 1 else [0.0]
    sv, sa = sorted(v), sorted(a)
    peak_v, peak_a = (sv[-1] if sv else 0.0), (sa[-1] if sa else 0.0)
    return {
        "peak_v": peak_v, "p95_v": _percentile(sv, 0.95),
        "peak_a": peak_a, "p95_a": _percentile(sa, 0.95),
        "v_limit": v_limit, "a_limit": a_limit,
        "utilization_v": peak_v / v_limit if v_limit > 0 else 0.0,
        "utilization_a": peak_a / a_limit if a_limit > 0 else 0.0,
    }


def _feasibility_report(seq, fingering: FingeringPlan, bowing: BowingPlan, extra_violations,
                        realized, tracks: Tracks, profile: HardwareProfile) -> FeasibilityReport:
    notes = seq.sorted().notes
    entries = []
    late_per_note = {}
    for i in range(len(notes)):
        vs = list(fingering.assignments[i].violations) + list(bowing.violations.get(i, ())) \
            + list(extra_violations.get(i, ()))
        for v in vs:
            entries.append({"note_index": i, "kind": v.kind, "axis": v.axis,
                            "needed": v.needed, "available": v.available,
                            "late_by_s": v.late_by_s})
        late_per_note[i] = max((v.late_by_s for v in vs), default=0.0)

    hop = tracks.hop_s
    axis_map = {
        "bow.inclination_rad": (profile.bow.incl.v_max_radps, profile.bow.incl.a_max_radps2),
        "bow.y_m": (profile.bow.y.v_max_mps, profile.bow.y.a_max_mps2),
    }
    for fi, unit in enumerate(profile.fingers):
        axis_map[f"f{fi}.z_m"] = (unit.z.v_max_mps, unit.z.a_max_mps2)
        axis_map[f"f{fi}.x_m"] = (unit.x.v_max_mps, unit.x.a_max_mps2)
    utilization = {name: _axis_stats(tracks.channels[name], hop, vlim, alim)
                   for name, (vlim, alim) in axis_map.items() if name in tracks.channels}
    # Value-limited (not derivative-limited) channels:
    for name, limit in (("bow.speed_mps", profile.bow.belt.v_max_mps),
                        ("bow.force_n", profile.bow.force.f_max_n)):
        vals = tracks.channels.get(name, ())
        peak = max(vals) if vals else 0.0
        utilization[name] = {"peak_v": peak, "p95_v": _percentile(sorted(vals), 0.95),
                             "peak_a": 0.0, "p95_a": 0.0, "v_limit": limit, "a_limit": 0.0,
                             "utilization_v": peak / limit if limit > 0 else 0.0,
                             "utilization_a": 0.0}

    n_notes = len(notes)
    bad_notes = {e["note_index"] for e in entries}
    summary = {
        "n_notes": n_notes,
        "n_violations": len(entries),
        "feasibility_pct": 100.0 * (n_notes - len(bad_notes)) / n_notes if n_notes else 100.0,
        "total_late_s": sum(late_per_note.values()),
        "worst_late_s": max(late_per_note.values(), default=0.0),
    }
    entries.sort(key=lambda e: (e["note_index"], e["kind"], e["axis"]))
    return FeasibilityReport(summary=summary, violations=tuple(entries),
                             axis_utilization=utilization)
