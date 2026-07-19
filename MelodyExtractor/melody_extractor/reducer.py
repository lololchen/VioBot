"""Playability reducer: (NoteSequence, StageConfig) -> NoteSequence. Pure — no I/O, no model calls.

Algorithm (D-004): input-output HMM decoded with Viterbi, after Hori 2013;
violin state-space and continuity costs after Maezawa 2012. Rolled-triple
semantics per D-009 (single-bow bridge-curvature physics; note: Kamatani 2022
does NOT quantify this — see decisions.md correction entry). The reducer
reasons about string feasibility to decide WHICH notes survive, but never
emits string/finger assignments (D-002).

== Time grid ==
Change points = sorted unique {onset, offset} of all notes -> half-open
intervals I_k = [t_k, t_{k+1}). Note n is "active" in I_k iff it overlaps it.

== States ==
For each interval: all subsets S of active notes with |S| <= max_voices that
pass subset_feasible(). If more than max_active notes are active, keep only
the top max_active by importance() before enumerating subsets (deterministic
pruning; record "pruned": true in meta.extra). The empty subset is always a
valid state (silence).

== Costs (Viterbi minimizes total) ==
Emission E(S, k)  = sum over active notes n NOT in S of importance(n) * len(I_k)
  importance(n)   = w_amp  * clip01((peak_db + 60) / 60)
                  + w_pitch * pitch_rank01(n)      # percentile of pitch among
                                                   # concurrently active notes:
                                                   # top-voice bias
                  + w_dur  * min(1.0, duration_s)
Transition T(S_prev -> S, at boundary t_k) =
    w_frag * |{n in S_prev : n still active in I_k, n not in S}|   # mid-note drop
  + w_frag * |{n in S : n already active in I_{k-1}, n not in S_prev}|  # late entry
  + w_jump * (|semitones(top_pitch(S)) - semitones(top_pitch(S_prev))| / 7)²
             (0 when either side is empty)
The /7-squared jump term follows Maezawa 2012's horizontal-vertical transition
model p(Si|Sj; v) ∝ exp(−v((Δp/7)² + Δs²)) (p. 63: intervals up to 7 semitones
are typically fingered without changing hand position). Weights are
PROVISIONAL until hardware kinematics exist (PRD open question); they are
config fields so tuning never touches code.

Known quirk (v0.1, keep in mind when tuning weights): because entering or
leaving the empty state never pays the jump term, an extreme NON-RETURNING
register switch can make the optimal path pass through silence to "reset" the
jump cost — occasionally dropping a sole-active note outright instead of
truncating it. Harmless at default weights for realistic material; if it
bites, cap the jump term (min(jump, cap)) or charge jumps across short
silences — revisit alongside the SysID-driven weight tuning.

== Decoding ==
Viterbi over intervals; DP tie-break must be deterministic: on equal cost
prefer the state whose key (tuple of canonical note indices, ascending) is
lexicographically smallest. Canonical note index = position in
NoteSequence.sorted().notes.

== Reconstruction ==
For each input note, its kept intervals form runs; keep the longest contiguous
run (earliest on tie). New onset/duration = run bounds; drop the note if the
kept duration < min_keep_s. Clip f0_contour and amp_db_envelope to the kept
span (times stay onset-relative). Voice index: at every instant, sounding
output notes ranked by pitch descending -> rank 0,1,2; a note's voice = the
maximum rank it ever holds (stable, deterministic). When a state holds 3 notes
they are marked rolled=True (single-bow triple stops are infeasible, D-009).

== Output ==
NoteSequence.sorted() with:
  meta.stage    = config.config_dict()
  meta.backends = input backends + {"reducer": "hmm-viterbi-0.1.0"}
  features      = passed through unchanged
Playability invariants the eval harness gates on (algorithm-validation skill):
no instant has > max_voices sounding notes; every simultaneous pair/triple
passes subset_feasible(); zero notes outside [open G3, max_pitch_hz].
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from itertools import combinations

from .schema import AmpEnvelope, F0Contour, Meta, NoteSequence, hz_to_midi

_EPS = 1e-9

REDUCER_VERSION = "hmm-viterbi-0.1.0"

# Violin open strings G3, D4, A4, E5 — EXACT equal temperament (A4=440),
# i.e. midi_to_hz(55/62/69/76). Rounded literals here once made the range
# check reject exact-ET G3 notes (195.9977 Hz < "196.0"), silently dropping
# them — caught by the eval harness, see decisions.md D-015.
OPEN_STRINGS_HZ = (195.99771799087463, 293.6647679174076, 440.0, 659.2551138257398)


@dataclass(frozen=True)
class StageConfig:
    """Hardware stage gates (CLAUDE.md): only this config changes per stage."""

    max_voices: int = 1                      # N in {1, 2, 3}
    open_strings_hz: tuple = OPEN_STRINGS_HZ
    max_pitch_hz: float = 2637.0             # ~E7, practical fingerboard top
    max_fingerboard_semitones: float = 19.0  # highest stopped position per string (PROVISIONAL — not from papers)
    max_position_span_semitones: float = 5.0 # cross-string finger span; Maezawa 2012 Pnat(index,little)=5 st (Pmax=6)
    rolled_triples: bool = True              # 3-note chords only as rolled (D-009; see decisions.md correction note)
    pitch_tolerance_semitones: float = 0.3   # jitter allowance at range/position boundaries (D-015)
    max_active: int = 8                      # prune bound for subset enumeration
    min_keep_s: float = 0.05                 # drop notes shorter than this after clipping
    # Cost weights — PROVISIONAL until SysID/kinematics data exists.
    w_amp: float = 1.0
    w_pitch: float = 1.0
    w_dur: float = 0.5
    w_frag: float = 2.0
    w_jump: float = 0.3

    @classmethod
    def stage(cls, n: int) -> "StageConfig":
        """Presets for the hardware roadmap: 1=mono, 2=adjacent-string double
        stops, 3=+rolled triples. (Stage 4 / two-bow is future work.)"""
        if n not in (1, 2, 3):
            raise ValueError(f"stage must be 1, 2 or 3 (got {n})")
        return cls(max_voices=n)

    def config_dict(self) -> dict:
        d = asdict(self)
        d["open_strings_hz"] = list(self.open_strings_hz)
        d["reducer_version"] = REDUCER_VERSION
        return d


def subset_feasible(pitches_hz: "tuple[float, ...]", config: StageConfig) -> bool:
    """Acoustic feasibility of sounding these pitches simultaneously on one bow.

    Rules (hardware-agnostic — no assignment is returned):
    - every pitch within [lowest open string, max_pitch_hz]
    - k pitches (sorted ascending) must sit on k CONSECUTIVE strings, lower
      pitch on lower string; on each string the stopped position
      pos = semitones above that open string must satisfy
      0 <= pos <= max_fingerboard_semitones
    - hand span: max(pos) - min(pos) over STOPPED notes only (pos > 0.01;
      open strings need no finger) must be <= max_position_span_semitones
    - k == 3 requires config.rolled_triples (single bow cannot sustain them)
    - k > 3 is infeasible (two-bow hardware is out of scope for v0)
    All boundary comparisons happen in the semitone domain with a
    pitch_tolerance_semitones slack: transcription jitter of a few cents must
    never disqualify a nominally playable pitch (D-015).
    """
    k = len(pitches_hz)
    if k == 0:
        return True
    if k > 3 or k > config.max_voices:
        return False
    tol = config.pitch_tolerance_semitones
    lo_st = hz_to_midi(config.open_strings_hz[0]) - tol
    hi_st = hz_to_midi(config.max_pitch_hz) + tol
    if any(not (lo_st <= hz_to_midi(p) <= hi_st) for p in pitches_hz):
        return False
    if k == 1:
        return True
    if k == 3 and not config.rolled_triples:
        return False
    pitches = sorted(pitches_hz)
    n_strings = len(config.open_strings_hz)
    for s in range(n_strings - k + 1):
        positions = []
        ok = True
        for i, p in enumerate(pitches):
            pos = hz_to_midi(p) - hz_to_midi(config.open_strings_hz[s + i])
            if pos < -tol or pos > config.max_fingerboard_semitones + tol:
                ok = False
                break
            positions.append(max(0.0, pos))
        if not ok:
            continue
        stopped = [p for p in positions if p > 0.01]
        if len(stopped) >= 2 and (max(stopped) - min(stopped)) > config.max_position_span_semitones + 1e-6:
            continue
        return True
    return False


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _change_points(notes, eps: float = _EPS) -> "list[float]":
    """Sorted, epsilon-deduped {onset, offset} times across `notes`.

    `notes` is any sequence of objects exposing .onset_s / .offset_s (works
    for Note tuples straight from NoteSequence.sorted().notes as well as for
    reducer-internal output-note lists).
    """
    raw = sorted({n.onset_s for n in notes} | {n.offset_s for n in notes})
    points: "list[float]" = []
    for t in raw:
        if not points or t - points[-1] > eps:
            points.append(t)
    return points


def _active_indices(notes, t0: float, t1: float, eps: float = _EPS) -> "list[int]":
    """Canonical indices of notes fully covering half-open interval [t0, t1)."""
    return [i for i, n in enumerate(notes) if n.onset_s <= t0 + eps and n.offset_s >= t1 - eps]


def _pitch_ranks01(idxs, pitch) -> "dict[int, float]":
    """Percentile rank (0=lowest pitch .. 1=highest) among `idxs`; 1.0 when alone."""
    if len(idxs) <= 1:
        return {i: 1.0 for i in idxs}
    order = sorted(idxs, key=lambda i: (pitch[i], i))
    m = len(order)
    return {idx: rank / (m - 1) for rank, idx in enumerate(order)}


def _interp_series(times, values, t: float) -> float:
    """Linear interpolation of a (times, values) series at time t (clamped at ends)."""
    if t <= times[0]:
        return values[0]
    if t >= times[-1]:
        return values[-1]
    for i in range(len(times) - 1):
        t0, t1 = times[i], times[i + 1]
        if t0 <= t <= t1:
            if t1 == t0:
                return values[i]
            frac = (t - t0) / (t1 - t0)
            return values[i] + frac * (values[i + 1] - values[i])
    return values[-1]


def _clip_series(times, values, lo: float, hi: float, eps: float = _EPS):
    """Clip a (times, values) series to [lo, hi] and re-baseline times to lo.

    Interpolates exact values at the lo/hi boundaries so at least the two
    endpoints are always present (existing interior points are kept as-is).
    """
    pts = [(t, v) for t, v in zip(times, values) if lo - eps <= t <= hi + eps]
    if not pts or pts[0][0] > lo + eps:
        pts.insert(0, (lo, _interp_series(times, values, lo)))
    if pts[-1][0] < hi - eps:
        pts.append((hi, _interp_series(times, values, hi)))
    new_times = tuple(max(0.0, t - lo) for t, _ in pts)
    new_values = tuple(v for _, v in pts)
    return new_times, new_values


def _clip_envelope(env: AmpEnvelope, lo: float, hi: float) -> AmpEnvelope:
    times, values = _clip_series(env.times_s, env.amp_db, lo, hi)
    return AmpEnvelope(times_s=times, amp_db=values)


def _clip_contour(contour: F0Contour, lo: float, hi: float) -> F0Contour:
    times, values = _clip_series(contour.times_s, contour.f0_hz, lo, hi)
    return F0Contour(times_s=times, f0_hz=values)


def _transition_cost(sp, s, prev_active: set, cur_active: set, pitch, config: StageConfig) -> float:
    """T(S_prev -> S) per the module docstring: mid-note drop + late entry + jump."""
    sp_set = set(sp)
    s_set = set(s)
    mid_note_drop = sum(1 for i in sp_set if i in cur_active and i not in s_set)
    late_entry = sum(1 for i in s_set if i in prev_active and i not in sp_set)
    frag = config.w_frag * (mid_note_drop + late_entry)
    if sp and s:
        top_prev = max(sp, key=lambda i: pitch[i])
        top_cur = max(s, key=lambda i: pitch[i])
        delta = hz_to_midi(pitch[top_cur]) - hz_to_midi(pitch[top_prev])
        jump = config.w_jump * (abs(delta) / 7.0) ** 2
    else:
        jump = 0.0
    return frag + jump


def _assign_voices(recs) -> "list[int]":
    """Voice = max concurrent pitch-descending rank a note ever holds (rank 0 = top)."""
    n = len(recs)
    if n == 0:
        return []
    max_rank = [0] * n
    points = _change_points(recs)
    for t0, t1 in zip(points, points[1:]):
        active = _active_indices(recs, t0, t1)
        if not active:
            continue
        order = sorted(active, key=lambda i: (-recs[i].pitch_hz, i))
        for r, i in enumerate(order):
            if r > max_rank[i]:
                max_rank[i] = r
    return max_rank


def reduce(seq: NoteSequence, config: StageConfig) -> NoteSequence:
    """Reduce to <= max_voices playable voices. Pure function; deterministic.

    Implementation spec is in the module docstring. Must use subset_feasible()
    for state enumeration and StageConfig weights for costs.
    """
    seq_sorted = seq.sorted()
    notes = seq_sorted.notes
    n = len(notes)

    if n == 0:
        new_backends = dict(seq_sorted.meta.backends)
        new_backends["reducer"] = REDUCER_VERSION
        new_meta = Meta(
            source=seq_sorted.meta.source,
            source_kind=seq_sorted.meta.source_kind,
            sample_rate=seq_sorted.meta.sample_rate,
            backends=new_backends,
            stage=config.config_dict(),
            extra=dict(seq_sorted.meta.extra),
        )
        return NoteSequence(notes=(), features=seq_sorted.features, meta=new_meta)

    pitch = [note.pitch_hz for note in notes]
    points = _change_points(notes)
    intervals = list(zip(points, points[1:]))

    amp_term = [config.w_amp * _clip01((note.amp_db_envelope.peak_db() + 60.0) / 60.0) for note in notes]
    dur_term = [config.w_dur * min(1.0, note.duration_s) for note in notes]

    feas_cache: "dict[tuple, bool]" = {}

    def feasible(idxs) -> bool:
        key = tuple(sorted(pitch[i] for i in idxs))
        cached = feas_cache.get(key)
        if cached is None:
            cached = subset_feasible(key, config)
            feas_cache[key] = cached
        return cached

    pruned_any = False
    dp_history: "list[dict[tuple, tuple[float, tuple]]]" = []

    prev_dp = {(): (0.0, None)}
    prev_active: set = set()

    for t0, t1 in intervals:
        length = t1 - t0
        active = _active_indices(notes, t0, t1)
        active_set = set(active)

        ranks = _pitch_ranks01(active, pitch)
        importance_map = {i: amp_term[i] + config.w_pitch * ranks[i] + dur_term[i] for i in active}

        if len(active) > config.max_active:
            candidates = sorted(active, key=lambda i: (-importance_map[i], i))[: config.max_active]
            pruned_any = True
        else:
            candidates = list(active)
        candidates_sorted = sorted(candidates)

        max_size = min(config.max_voices, len(candidates_sorted))
        states = []
        for size in range(0, max_size + 1):
            for combo in combinations(candidates_sorted, size):
                if size > 0 and not feasible(combo):
                    continue
                states.append(combo)
        states.sort()

        cur_dp: "dict[tuple, tuple[float, tuple]]" = {}
        for s in states:
            s_set = set(s)
            best_cost = None
            best_prev = None
            for sp in sorted(prev_dp.keys()):
                prev_cost = prev_dp[sp][0]
                trans = _transition_cost(sp, s, prev_active, active_set, pitch, config)
                total = prev_cost + trans
                if best_cost is None or total < best_cost:
                    best_cost = total
                    best_prev = sp
            emission = length * sum(importance_map[i] for i in active if i not in s_set)
            cur_dp[s] = (best_cost + emission, best_prev)

        dp_history.append(cur_dp)
        prev_dp = cur_dp
        prev_active = active_set

    last_dp = dp_history[-1]
    best_final = None
    best_final_cost = None
    for s in sorted(last_dp.keys()):
        cost, _ = last_dp[s]
        if best_final_cost is None or cost < best_final_cost:
            best_final_cost = cost
            best_final = s

    chosen: "list[tuple]" = [None] * len(intervals)  # type: ignore[list-item]
    cur = best_final
    for k in range(len(intervals) - 1, -1, -1):
        chosen[k] = cur
        cur = dp_history[k][cur][1]

    kept_by_note: "dict[int, list[int]]" = {i: [] for i in range(n)}
    for k, s in enumerate(chosen):
        for i in s:
            kept_by_note[i].append(k)

    output_notes = []
    for i, note in enumerate(notes):
        ks = kept_by_note[i]
        if not ks:
            continue
        runs = []
        run_start = ks[0]
        run_prev = ks[0]
        for k in ks[1:]:
            if k == run_prev + 1:
                run_prev = k
            else:
                runs.append((run_start, run_prev))
                run_start = k
                run_prev = k
        runs.append((run_start, run_prev))

        def _run_duration(r):
            k0, k1 = r
            return intervals[k1][1] - intervals[k0][0]

        best_run = min(runs, key=lambda r: (-_run_duration(r), r[0]))
        k0, k1 = best_run
        new_onset = intervals[k0][0]
        new_offset = intervals[k1][1]
        new_duration = new_offset - new_onset
        if new_duration < config.min_keep_s:
            continue

        rolled = any(len(chosen[k]) == 3 for k in range(k0, k1 + 1))

        if abs(new_onset - note.onset_s) > _EPS or abs(new_duration - note.duration_s) > _EPS:
            lo = new_onset - note.onset_s
            hi = new_offset - note.onset_s
            new_env = _clip_envelope(note.amp_db_envelope, lo, hi)
            new_contour = _clip_contour(note.f0_contour, lo, hi) if note.f0_contour is not None else None
        else:
            new_env = note.amp_db_envelope
            new_contour = note.f0_contour

        output_notes.append(
            replace(
                note,
                onset_s=new_onset,
                duration_s=new_duration,
                amp_db_envelope=new_env,
                f0_contour=new_contour,
                rolled=rolled,
                voice=None,
            )
        )

    voice_ranks = _assign_voices(output_notes)
    final_notes = tuple(replace(note, voice=voice_ranks[idx]) for idx, note in enumerate(output_notes))

    new_backends = dict(seq_sorted.meta.backends)
    new_backends["reducer"] = REDUCER_VERSION
    new_extra = dict(seq_sorted.meta.extra)
    if pruned_any:
        new_extra["pruned"] = True
    new_meta = Meta(
        source=seq_sorted.meta.source,
        source_kind=seq_sorted.meta.source_kind,
        sample_rate=seq_sorted.meta.sample_rate,
        backends=new_backends,
        stage=config.config_dict(),
        extra=new_extra,
    )

    return NoteSequence(notes=final_notes, features=seq_sorted.features, meta=new_meta).sorted().validate()


def playability_violations(seq: NoteSequence, config: StageConfig) -> "list[str]":
    """Audit helper for the eval harness: return a list of human-readable
    violation strings (empty == playable). Checks the three output invariants
    from the module docstring at every change point."""
    s = seq.sorted()
    notes = s.notes
    violations: "list[str]" = []
    if not notes:
        return violations

    tol = config.pitch_tolerance_semitones
    lo_st = hz_to_midi(config.open_strings_hz[0]) - tol
    hi_st = hz_to_midi(config.max_pitch_hz) + tol
    for i, note in enumerate(notes):
        st = hz_to_midi(note.pitch_hz)
        if not (lo_st <= st <= hi_st):
            violations.append(
                f"note {i} (onset={note.onset_s:.6g}s, pitch={note.pitch_hz:.3f} Hz) "
                f"outside playable range [{config.open_strings_hz[0]:.3f}, "
                f"{config.max_pitch_hz:.3f}] Hz (±{tol} st)"
            )

    points = _change_points(notes)
    for t0, t1 in zip(points, points[1:]):
        sounding = _active_indices(notes, t0, t1)
        if not sounding:
            continue
        if len(sounding) > config.max_voices:
            violations.append(
                f"t=[{t0:.6g},{t1:.6g}): {len(sounding)} voices sounding exceeds max_voices={config.max_voices}"
            )
        pitches = tuple(notes[i].pitch_hz for i in sounding)
        if not subset_feasible(pitches, config):
            violations.append(f"t=[{t0:.6g},{t1:.6g}): pitch set {pitches} fails subset_feasible")
        if len(sounding) == 3 and config.rolled_triples:
            unrolled = [i for i in sounding if not notes[i].rolled]
            if unrolled:
                violations.append(
                    f"t=[{t0:.6g},{t1:.6g}): 3-note state has non-rolled note(s) at index {unrolled}"
                )
    return violations
