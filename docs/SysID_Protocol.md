# SysID Protocol — bench sweeps for the acoustics↔actuation mapper

Owner: MotionPlanner (`motion_planner/sysid.py`). Prerequisite for replacing the analytic
Schelleng model (D-027) with a learned mapper. **SysID priority #1 is the belt-bow attack
regime** — constant-speed hair engaging via Y is not covered by the bowed-string literature
(PRD caveat), so no analytic prior exists to fall back on there.

## What gets measured

Per grid point: steady-tone acoustics = **exactly the D-008 feature set** — f0, amplitude (dBFS),
per-harmonic amplitudes, odd/even ratio, tristimulus, inharmonicity. Measurement = run
MelodyExtractor's transcriber + timbre on the bench recording (`sysid.measured_from_note`);
no new DSP exists or should exist for this.

## Grid (defaults in `SweepConfig`)

| Axis | Values | Note |
|---|---|---|
| string | G, D, A, E | per-string impedance differs |
| position_st | 0, 2, 5, 9, 14, 19 | 0 = open; spans the fingerboard |
| v_b (m/s) | 0.1, 0.2, 0.4 | extend upward once belt limits are known |
| force (N) | 0.3, 0.8, 1.5 | bracket the predicted Schelleng wedge |
| β | 0.10 (+ more if the axis is actuated) | PRD open Q2 decides |

Each point: **0.25 s engage/settle → 2 s steady tone → release**. Additionally record the
Y-engage attack transient of every point (first 100 ms at ≥16 kHz) and log the attack tuple
(v_b, F, dF/dt, β) — this is the Guettler-replacement dataset.

## Procedure

1. `sysid.sweep_points(config)` enumerates the grid deterministically; `point_to_score`
   turns each point into a tiny MotionScore with hand-set control tracks.
2. Execute through `firmware_bridge` (bench.py / streamer.py) once multi-axis firmware exists;
   until then single-axis subsets via device 0x01.
3. Record WAV per point. **Filename = `{point_id}__{motionscore_sha256[:12]}__{profile_sha256[:12]}.wav`**
   so every sample is traceable to exact commanded controls and hardware profile.
4. Extract with MelodyExtractor (mono path, timbre on), fill the `measured` block of each
   `SweepPoint`, save the `SweepDataset` JSON (versioned, deterministic).
5. Fit `LearnedBowSoundModel` (deterministic ridge on log-features; dataset hash pinned into the
   model version string). The analytic model stays as fallback + sanity prior.

## Recording chain (to fix before the first sweep)

- Microphone position repeatable (mount to the frame, not the room); note distance/angle in
  the dataset meta. Motor self-noise: record a silent-motion take per session for a noise floor
  reference (spectral subtraction decision deferred to AudioFeedback).
- Levels calibrated once per session against a reference tone; all amp_db values are dBFS of
  the fixed chain — comparisons are valid within a session, absolute SPL is out of scope.

## Synergy

Every AudioFeedback v0 take is a free sweep contribution: MotionScore hash → commanded
controls, extraction → measured block (CONCEPT_AudioFeedback.md).
