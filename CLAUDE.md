# Violin Robot — Repo Guide

A robot that listens to arbitrary music and replays it on a real violin.

## Module map
| Module | Role | Status |
|---|---|---|
| `MelodyExtractor/` | Audio/MIDI → note sequence + timbre features + playability reduction + sound sim | **Active** |
| `MotionPlanner/` (a.k.a. Sound2Motion) | NoteSequence → MotionScore: fingering, bow speed/pressure/inclination, vibrato; forward sim + topology comparison; owns SysID mapper + firmware bridge | **Active** |
| `AudioFeedback/` | Closed-loop listening during play | Concept — `docs/CONCEPT_AudioFeedback.md` |
| `Firmware/` | Motor control — see https://github.com/AachenBQ/Motor_Architecture | External repo |

## Ground rules
- **Read `docs/PRD_MelodyExtractor.md` before touching MelodyExtractor, and `docs/PRD_MotionPlanner.md` before touching MotionPlanner.** They encode decisions already debated; don't silently re-litigate them.
- **Log every architectural decision in `docs/decisions.md`** (append-only, dated, with alternatives considered). If you change an approach, add a new entry superseding the old one — never rewrite history.
- Module boundary discipline: MelodyExtractor outputs hardware-agnostic acoustics (`NoteSequence` JSON schema). Anything about fingers, bows, or motors belongs in MotionPlanner (`MotionScore` JSON schema); motor-protocol specifics stay in its `firmware_bridge/`. Closed-loop listening is AudioFeedback's. Reject scope creep in all directions. Dependency direction: `motion_planner` imports `melody_extractor`, never the reverse (D-022).
- Determinism matters: same input file ⇒ byte-identical output (fix seeds, pin model versions). The robot is debugged against these outputs.
- Hardware stage gates (from PRD): mono → 2-note adjacent-string double stops → 3-note rolled → 4-finger/2-bow. Only `reducer` config (MelodyExtractor) and `HardwareProfile`/`PlannerConfig` (MotionPlanner) change per stage.
- GUIs: `run_gui.bat` opens one browser window with all module tabs via `gui_hub/` (port registry + workspace manifest); each module's GUI also runs standalone (D-030, once built).

## Environment
- Python ≥3.10, `uv` or `pip` with `requirements.txt`; ffmpeg required on PATH.
- Heavy deps (tensorflow for basic-pitch/CREPE, essentia) are optional extras — keep core importable without them; guard imports.
- Tests: `pytest`; transcription accuracy via `mir_eval` against fixtures in `MelodyExtractor/tests/fixtures/`.

## Validation
Before claiming any extraction/reduction algorithm works, use the `algorithm-validation` skill (`.claude/skills/algorithm-validation/`). Listening tests + mir_eval numbers, not vibes.

## Model routing
Model-per-task guidance for Claude Code lives in `.claude/skills/model-routing/SKILL.md` and `.claude/agents/`. Default session model: sonnet.
