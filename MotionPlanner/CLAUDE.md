# MotionPlanner (Sound2Motion) — Module Guide

Planning mid-layer: reduced `NoteSequence` → `MotionScore` (fingering, bow, vibrato, trajectories)
against a declared `HardwareProfile`, plus forward sim, topology comparison, SysID scaffolding and
the firmware bridge. Full spec: `../docs/PRD_MotionPlanner.md` (read it first — decisions
D-022…D-030 are encoded there).

## Layout
```
motion_planner/
  schema.py          # MotionScore v0.1.0 + FeasibilityReport (NoteSequence governance clone)
  hardware.py        # HardwareProfile dataclasses + geometry helpers (st↔mm, trapezoid_time, bands)
  profile_io.py      # profile JSON v1 load/save + sha256 pinning (config_io pattern)
  config_io.py       # PlannerConfig + preset JSONs (reuses melody_extractor.config_io.config_from_dict)
  fingering.py       # Viterbi DP (string, finger) assignment; topology A/B behind one interface
  bowing.py          # contact segments, inclination ramps, Schelleng speed/force, roll timing (D-024)
  vibrato.py         # f0_contour → rate/depth/delay → finger-Z oscillation (clipped, flagged)
  trajectory.py      # events → sampled tracks (closed-form trapezoids) + FeasibilityReport
  planner.py         # pure: (NoteSequence, HardwareProfile, PlannerConfig) → (MotionScore, Report)
  bow_sound_model.py # BowSoundModel interface; AnalyticSchellengModel v0
  simulate.py        # MotionScore → predicted NoteSequence (kinematic gate + forward model)
  roundtrip.py       # target vs predicted: mir_eval + melody_extractor.soundsim renders
  compare.py         # corpus × profiles → topology report (tempo-headroom bisection)
  sysid.py           # SweepConfig/SweepDataset v1 + sweep→MotionScore generator
  cli.py             # plan | simulate | roundtrip | compare | bench
  firmware_bridge/   # Native Protocol v2 frames, Mock/Serial transports, 100 Hz streamer, bench
  gui/               # [gui] extra — Streamlit app (see gui_hub/ at repo root for the 4-tab shell)
profiles/            # HardwareProfile JSONs (concept_a_1finger / concept_a_2finger / concept_b_4finger)
tests/               # pytest; fixtures/ = reduced NoteSequence JSONs generated from MelodyExtractor's MIDI fixtures
```

## Contracts (do not break)
- `MotionScore` JSON is the interface consumed by the sim, GUI, firmware bridge and (future)
  AudioFeedback — schema changes require a version bump + `docs/decisions.md` entry (D-023).
- `planner.plan()` is **pure and deterministic**: no I/O, no RNG, byte-identical output for
  identical `(NoteSequence, HardwareProfile, PlannerConfig)`. Same rule as the reducer.
- Planners **never hard-fail on infeasibility** — soft costs + recorded violations; the topology
  comparison must be able to measure *how* infeasible a profile is (D-026).
- Fingering feasibility must stay a superset of the reducer's (same `OPEN_STRINGS_HZ`, same 0.3 st
  tolerance): every reduced NoteSequence gets a complete assignment, no plan-time dead ends.
- MotionScore stores task-space physical units (m, m/s, N, rad). Joint space (bow differential,
  four-bar, lead screw) and motor units live only in `HardwareProfile` + `firmware_bridge`.
- All firmware command constants carry `# CONFIRM-WITH-FIRMWARE` — the firmware repo is a days-old
  skeleton; pin nothing silently (D-029).
- `motion_planner` imports `melody_extractor`; never the reverse (D-022).
- `gui/` is the only place allowed to import streamlit/plotly; GUI writes only profiles/presets,
  never baselines. Custom CSS only in `gui/style.py`. (Mirrors D-016/D-017.)

## Gotchas
- Rolled triples: realized onsets follow D-024 (pair-then-pair, `roll_span_s` dwell) — sim and
  mir_eval must compare against **realized** timing from `note_plan`, not raw score onsets.
- β (bow–bridge) uses **sounding** length: `β_eff = d_bridge / (L·2^(−p/12))` — it drifts sul tasto
  as positions climb; don't "fix" this in code, it's physics (PRD β section).
- The belt bow has no bow changes and one direction: never plan direction reversals; attack quality
  is force-rise-time at Y-engage, not Guettler acceleration (PRD caveat).
- Lengths are suffixed: `_mm` fields are millimetres (fingerboard geometry), `_m` metres (axis
  ranges, tracks). Convert only at the suffix boundary; never guess.
- WAV writes go through `melody_extractor.soundsim` (scipy.io.wavfile — D-013). Don't import
  soundfile for writing.

## Definition of done for any planner change
Round-trip on the four fixtures (`motion-planner roundtrip`): mir_eval numbers + paired listening
files + byte-determinism (run twice, sha256) + a `docs/decisions.md` entry if behavior changed.
Run the repo `algorithm-validation` skill before claiming improvement.
