# PRD — MotionPlanner (Sound2Motion)

Status: v0 in build. Read this before touching `MotionPlanner/` — it encodes decisions already debated
(decisions.md D-022…D-030). Companion docs: `CONCEPT_AudioFeedback.md`, `SysID_Protocol.md`.

## Overview

MotionPlanner is the planning mid-layer of the violin robot: it converts MelodyExtractor's
hardware-agnostic `NoteSequence` (pitch/onset/duration/dynamics/harmonics — deliberately free of
string/finger/bow information, D-002) into a **MotionScore**: fingering and string assignments,
bow speed/pressure/inclination trajectories, and vibrato commands, all expressed in physical
task-space units against a declared **HardwareProfile**. It owns the SysID acoustics→actuation
mapper (CLAUDE.md module map).

Because no mechanics exist yet, MotionPlanner v0 is **simulation-first**: every plan is validated by
a forward model (MotionScore → predicted NoteSequence → audio render → mir_eval against the target),
and its headline deliverable is the **topology comparison** — quantifying which finger concept
(A: 1–2 roaming 3-DoF fingers; B: GhostPlay-style one finger per string) can play which repertoire
at what tempo, *before* hardware is built.

Design principle (mirrors MelodyExtractor's): *plan against an abstract, swappable hardware profile;
put every hardware-specific number in config, never in code.*

## Hardware concept being parameterized (user spec, 2026-07-19)

**Bow** — a continuously spinning bow-hair belt (pulley-belt rubbing the string):
1. Belt drive: bow "speed" is belt surface speed; single direction in MVP; no bow changes; unlimited
   bow length. (Acoustic difference between 100→0→100% and 100→0→−100% speed profiles is a future
   evaluation.)
2. Linear Y axis: touch/leave the strings.
3. Rotational Z axis: bow inclination = string selection — single-string bands and double/triple-stop
   bands between them.
4. Bow tilt: TBD (GhostPlay used tilt qualitatively; note D-014 — Kamatani 2022 publishes **no**
   numeric inclination band, so no constant may be cited from it). Reserved profile key, unused v0.
- Mechanism: DoFs 2+3 are realized by two identical coupled motors (differential); after contact the
  common mode yields **bow pressure**. Skewness fixed at 90° to the strings in MVP.
- Control inputs: bow speed [0, v_max] m/s; bow pressure (stored as physical force N in MotionScore —
  torque/FOC-amps conversion is a profile/bridge concern, D-023); bow–bridge distance (see β physics
  below); tilt TBD.

**Fingers** — each finger unit has 3 DoF:
- Two SimpleFOCMini-driven QDDs (1503 BLDC, KV 2000–2700 TBD, DRV8313 driver, AS5048A 14-bit
  encoder) + capstan drive + four-bar linkage → XY planar motion: switch fingertip between the four
  strings, and press. Fingertip force sensing TBD (DRV8313 has no integrated current sense).
- One QDD + igus dryspin high-helix lead screw → linear Z: traverse along the fingerboard and
  oscillate for **vibrato**.
- Control inputs: position, speed, fingertip pressure.

**Topologies under comparison** (the crucial parametric question):
- **Concept A**: 1 roaming finger (POC) → 2 fingers (theoretically the whole fingerboard with
  6 motors; evaluates note transitions across neighboring strings/fingers) → 3–4 fingers later.
- **Concept B**: one finger per string (4 traverse axes + 4 pressing units), GhostPlay-style.
The `motion-planner compare` CLI answers this with feasibility %, **tempo headroom** (max tempo
scale at 100% feasibility), per-axis utilization (informs the KV choice), timing slack, vibrato
coverage, and motor count.

## β physics — what bow–bridge distance does (user question, answered)

β = bow–bridge distance / **sounding** string length (Schelleng 1973 contact point).
- Small β (near bridge, *sul ponticello*): the playable force wedge narrows — `F_min ∝ v_b/β²` grows
  faster than `F_max ∝ v_b/β` — so both higher force and speed are required; reward is a higher
  loudness ceiling (∝ `v_b/β`) and a brighter spectrum (slower harmonic rolloff).
- Large β (toward fingerboard, *sul tasto*): wide forgiving wedge, low force suffices, softer and
  darker sound.
- **Fixed-bow-placement consequence**: sounding length shrinks as notes are stopped higher
  (`L_s = L·2^(−p/12)`), so `β_eff = d_bridge/L_s` **rises automatically with position** — high
  passages drift toward sul tasto unless β gets at least a slow setup axis. The forward sim
  quantifies this drift; it is the evidence on which the "β actuated?" decision will be made.

## Functional requirements

- **F1 Fingering/string assignment**: deterministic Viterbi DP over (string, finger) states per
  sonority; feasibility a superset of the reducer's (same open-string constants, same 0.3 st
  tolerance, D-015) so any reduced NoteSequence gets a complete finite-cost assignment — soft costs
  and recorded violations, never a hard failure (the comparison must be able to *measure*
  infeasibility). Costs: travel-time hinge (trapezoid kinematics), Maezawa shift cost `(Δst/7)²`
  (D-014 precedent), band-distance bow cost, open-string bias, geometric collision exclusion
  (Concept A).
- **F2 Bow planning**: contact segments, lift/land, inclination ramps (in gaps, else stealing a
  configurable tail fraction); speed/force from the analytic Schelleng prior via `BowSoundModel`
  (D-027); **rolled-triple timing** = pair-then-pair band dwell (D-024, resolves D-009's TBD);
  inclination-rate coupling `±ω·r_contact` added to relative hair speed with a warning threshold.
- **F3 Vibrato**: rate/depth/delay extracted from `f0_contour` (autocorrelation, 3–9 Hz); mapped to
  finger-Z oscillation `Δz = (depth_cents/100)·mm_per_st(p, L)`; depth (never rate) clipped to
  actuator accel/bandwidth, clip recorded.
- **F4 Trajectory generation**: events → per-axis sampled tracks via closed-form trapezoidal
  profiles fitted backward from deadlines; deadline misses arrive late (never violate limits),
  `late_by_s` recorded and realized onsets shifted; emits the `FeasibilityReport`.
- **F5 Forward sim + round-trip**: kinematic sounding gate + `BowSoundModel.forward` → predicted
  `NoteSequence` → same-renderer A/B listening pairs + mir_eval (onset F1, onset+pitch F1, RPA).
  This is the module's `algorithm-validation` path.
- **F6 Topology comparison**: corpus × profiles → deterministic JSON + markdown report;
  `compare_baseline.json` is decision-governed like `eval_baseline.json`.
- **F7 Firmware bridge**: Native Protocol v2 encoder/decoder + Mock/Serial transports + 100 Hz
  target streamer with heartbeat/lease + single-motor bench utility. All command constants carry
  `# CONFIRM-WITH-FIRMWARE` markers (the firmware repo is days old, D-029).
- **F8 SysID scaffolding**: sweep protocol + `SweepDataset` whose `measured` block is exactly the
  D-008 timbre feature set (measurement = MelodyExtractor's transcriber+timbre on bench recordings);
  `BowSoundModel` interface with the analytic model as v0 and a learned mapper later.

## Non-functional requirements

- Determinism: `(NoteSequence, HardwareProfile, PlannerConfig) → (MotionScore, FeasibilityReport)`
  is pure; byte-identical JSON on identical input (house rule; sha256 checks in tests/CLI).
- Schema governance: MotionScore v0.1.0 mirrors NoteSequence's rules — frozen dataclasses,
  deterministic serialization, major-version gate, changes need a version bump + decisions entry.
- Python ≥3.10; heavy/optional deps behind extras (`[serial]`, `[gui]`); core importable bare.
- Module boundary: MelodyExtractor stays acoustic (no edits except sanctioned gui-only retrofits,
  D-030); anything motor-protocol-specific stays in `firmware_bridge/`; closed-loop listening is
  AudioFeedback's (concept only for now).

## Architecture

```
NoteSequence ─▶ fingering ─▶ bowing ─▶ vibrato ─▶ trajectory ─▶ MotionScore + FeasibilityReport
                   ▲            ▲                                   │
             HardwareProfile  BowSoundModel.inverse                 ├─▶ simulate ─▶ predicted NoteSequence
                                                                    │        └─▶ roundtrip (mir_eval + A/B render)
                                                                    ├─▶ compare (corpus × profiles)
                                                                    └─▶ firmware_bridge.streamer ─▶ UART / mock
```

MotionScore has three layers (D-023): `note_plan[]` (per-note assignment + realized timing +
violations), `events[]` (typed discrete plan — what firmware and the GUI consume), `tracks{}`
(fixed-hop task-space samples — what the sim consumes). Task-space only; joint space (differential
bow pair, four-bar, lead screw) is profile/bridge territory.

## Algorithm decisions (citations)

- Fingering DP: Viterbi over sonority states; costs shaped by **Maezawa et al. 2012** (project PDF;
  shift cost `(Δp/7)²`, span constants — already adopted in the reducer per D-014). Chosen over GA
  (Tuohy 2005 — stochastic, determinism-hostile) and MILP (solver dependency, nondeterministic
  tie-breaking) — D-026.
- Bow force/speed: **Schelleng 1973** playable-region wedge (`F_max = k_max·v_b/β`,
  `F_min = k_min·v_b/β²`), force placed at `F = F_min^(1−u)·F_max^u` with brightness `u` from
  tristimulus T3; loudness prior `amp_db ≈ a0 + 20·log10(v_b/β)`. **Guettler 2002** cited for
  attacks, with an honest caveat: a constant-speed belt never accelerates from rest, so the classic
  (acceleration, force) diagram does not directly apply — v0 shapes force rise time at Y-engage and
  logs `(v_b, F, dF/dt, β)` per onset as SysID priority #1 — D-027.
- Feasibility bounds: reducer constants (D-014/D-015) reused verbatim; **Kamatani 2022 GhostPlay**
  cited only qualitatively (D-014 correction: no numeric bands published).

## Firmware gaps to negotiate (AachenBQ/Motor_Architecture)

1. Multi-device addressing (M2–M8 reserved but unimplemented) + bus scheduling — Concept A-2 needs
   ~9 motors (6 finger + 2 bow differential + 1 belt).
2. A trajectory-streaming command (buffered timestamped waypoints) instead of 100 Hz SET_TARGET spam
   per axis over one UART.
3. Synchronized start / shared timebase across devices.
4. Fingertip/bow force estimation without DRV8313 current sense (load cell on the bench? encoder
   deflection observer?).
5. Ownership of SI scaling and the bow differential mixing matrix (host vs MCU).
6. Telemetry for SysID: position + current at ≥100 Hz.

## Open questions

1. Finger-unit crossing/min-separation (mech TBD) — profile defaults conservative
   (`fingers_can_cross=false`, 25 mm).
2. β actuated or setup-static? Decide on sim evidence (sul-tasto drift, above).
3. Same-string legato: v0 always lift-move-press (audible micro-gap); portamento mode later.
4. Bow tilt axis (reserved key).
5. Belt-bow attack regime (novel hardware — SysID priority #1).
6. Motor KV 2000 vs 2700 — decide from compare's axis-utilization reports.
7. Track JSON size for multi-minute songs (~5–10 MB) — accepted v0; `.npz` sidecar is the revisit
   trigger.
8. Rolled-chord `roll_anticipate` (top-note-on-beat) — deferred to listening tests (D-024).

## Explicitly out of scope

- Transcription/reduction quality — MelodyExtractor's.
- Closed-loop listening/correction — AudioFeedback (`CONCEPT_AudioFeedback.md`).
- Motor control, FOC tuning, electronics — Firmware repo; this module stops at protocol frames.
- Absolute sound-quality claims from the sim: the analytic model's constants are literature-shaped
  priors, so sim conclusions are **relative** (topology A vs B under the same model), not absolute.

## Stage gates

Same roadmap as the reducer (mono → double stops → rolled triples → 4-finger/2-bow): only
`HardwareProfile`/`PlannerConfig` change per stage; stage-4 two-bow planning is out of scope for v0.
