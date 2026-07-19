# Concept — AudioFeedback (closed-loop listening)

Status: **concept only** — no code. This doc exists so MotionPlanner's interfaces leave the right
hooks. Building anything beyond v0 requires a decisions.md entry revisiting D-003 (batch-only rule).

## Role

Listen to the robot playing, compare against what was *planned*, and feed corrections back —
first offline between takes, later online during play. Consumes three existing artifacts:
the target `NoteSequence` (MelodyExtractor), the executed `MotionScore` (MotionPlanner — its
meta hashes pin exactly which plan and hardware profile produced the take), and the microphone
recording.

## v0 — offline post-mortem (pure reuse, D-003 intact)

1. Record the robot take (WAV) alongside the `MotionScore` hash.
2. Run MelodyExtractor's extraction on the recording → measured `NoteSequence`.
3. **Align before judging**: offline DTW on f0/chroma tracks removes global latency, servo lag and
   slow drift (raw mir_eval against the target would count alignment as error).
4. Diff aligned-measured vs target: mir_eval scores + per-note error table (cents error, onset
   error, missing/extra notes, dynamics error in dB).
5. Emit a **CorrectionSet JSON** (schema v1, versioned like all repo artifacts):
   - per-(string, position-band) intonation offsets in cents → finger-Z offsets via
     `mm_per_st(p, L)` — applied by MotionPlanner as a calibration overlay on the next plan;
   - per-segment dynamics gain → bow force/speed trims;
   - flagged notes (silenced, squealed, late) for human review.
6. Every take doubles as SysID data: (known controls from MotionScore) × (measured D-008 acoustic
   features) pairs convert mechanically into `SweepDataset` rows — the learned `BowSoundModel`
   trains for free while the robot practices.

v0 needs no new DSP beyond offline DTW; everything else is existing MelodyExtractor + MotionPlanner
machinery. It is also the acceptance harness for first hardware bring-up.

## v1 — online score follower + slow corrections (requires D-003 revisit entry)

- Low-latency pitch tracking (YIN on short frames, ~50 ms hop) + online DTW/HMM score follower
  against the target NoteSequence.
- Corrections limited to *slow* servo trims (heavily smoothed intonation offsets → finger Z;
  dynamics trims → bow force), pushed as target offsets through `firmware_bridge`.
- Latency budget to analyze up front: mic → f0 (~50 ms) → alignment (~50 ms) → command (~10 ms)
  → mechanical response (finger/bow time constants). Corrections slower than ~1 s time constant
  are safe against instability; anything faster needs v2 thinking.
- Open questions: mic placement, motor self-noise in the recording (spectral gating? contact mic?),
  room compensation.

## v2 — fast reflex loops (sketch only)

Sub-100 ms reactions (bow-force servo on audio envelope, string-crossing squeal detection and
back-off). At these timescales the loop likely belongs firmware-side with the host supplying
setpoints; requires firmware gaps (telemetry, streaming) to be closed first.

## Non-goals

- Re-planning fingering mid-performance.
- Any learned model beyond the SysID mapper already owned by MotionPlanner.
- Audio-quality judgments beyond the D-008 feature set.
