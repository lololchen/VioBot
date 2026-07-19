# PRD — MelodyExtractor

## Overview
MelodyExtractor is the perception front-end of a violin-playing robot ("music → violin translator"). It converts arbitrary music audio into a hardware-agnostic acoustic feature sequence — per-note pitch (f0, plus 2nd/3rd concurrent f0s at later hardware stages), onset, duration, amplitude envelope, and harmonic-distribution descriptors — that downstream modules (MotionPlanner/Sound2Motion, AudioFeedback, Firmware) map onto fingering and bowing actions. It also includes a Sound Simulation submodule that renders extracted sequences back to audio, enabling a fully software-based proof of concept, decoupled from hardware. Extraction fidelity is scoped to what the robot roadmap can play: monophonic first, then 2-note double stops on adjacent strings, then 3-note (rolled), then 4-finger/2-bow. Design principle: extract everything faithfully once; reduce to hardware constraints in a separate, swappable stage.

## Functional & Non-Functional Requirements
**Functional:** (F1) Accept audio files (WAV/MP3/MP4-audio/FLAC via ffmpeg) and MIDI through two input adapters converging on one internal PCM/note representation. (F2) Extract per-frame f0, voicing confidence, amplitude; segment into note events (pitch, onset, duration, dynamics). (F3) Extract harmonic-distribution features per note (harmonic peak amplitudes, odd/even ratio, tristimulus, inharmonicity). (F4) Reduce polyphonic transcription to N≤{1,2,3} playable voices under string-adjacency constraints (stage-gated). (F5) Render any note sequence to audio for A/B listening. (F6) Export JSON + MIDI.
**Non-functional:** Offline/batch processing (no real-time requirement); deterministic output for identical input; runs on a laptop CPU (GPU optional); Python ≥3.10; every stage independently testable with mir_eval metrics; module boundaries stable so hardware generations only touch the reducer config.

## Chosen Architecture & Rationale
Pipeline of four decoupled stages: **InputAdapter → Transcriber → Reducer → Exporter**, with **SoundSim** attached to any stage output. Rationale (from design discussions): (1) Transcription and playability-reduction are different problems — one perception, one constrained optimization — so hardware evolution must not force transcriber rewrites. (2) A DDSP-style "transfer first, extract after" pipeline (Idea 2) was rejected: DDSP conditions only on f0+loudness, discarding the source's real harmonic distribution and re-hallucinating a training-set violin unrelated to our hardware; it is also monophonic-only, breaking the chord requirement. Direct feature extraction (Idea 1) is a strict subset of Idea 2's effort. (3) Acoustic-feature output stays instrument/hardware-agnostic; the future SysID-based mapper (in MotionPlanner) owns "acoustics → fingering/bowing". (4) Deterministic algorithms preferred over stochastic ones for robot debuggability.

## Submodule Breakdown
**extraction/** — `input_adapter` (ffmpeg decode → 16 kHz mono PCM; MIDI parser → note events directly, skipping DSP); `transcriber` (monophonic path: CREPE f0 + confidence; polyphonic path: basic-pitch note events + pitch bends + amplitude-derived velocity); `timbre` (Essentia SpectralPeaks → HarmonicPeaks → odd/even, tristimulus, inharmonicity per note); `reducer` (HMM/Viterbi voice selection under playability costs; stage-configurable N and adjacency rules).
**soundsim/** — renders note sequences via FluidSynth + solo-violin SoundFont (baseline) with optional DDSP Tone Transfer violin checkpoint for realistic previews; produces paired original-vs-rendered WAVs for listening tests.
**interfaces/** — versioned JSON schema `NoteSequence{notes[], features[], meta}` consumed by MotionPlanner and AudioFeedback; MIDI export; CLI (`extract`, `reduce`, `render`, `eval`); mir_eval-based evaluation harness with ground-truth MIDI fixtures.

## Algorithm Decisions (with source papers)
- **Monophonic f0:** CREPE deep pitch tracker — Kim et al., ICASSP 2018. Frame-level f0 + voicing confidence, Viterbi smoothing.
- **Polyphonic transcription:** basic-pitch — Bittner et al., "A Lightweight Instrument-Agnostic Model for Polyphonic Note Transcription," ICASSP 2022. Note events, pitch bends, amplitude→velocity.
- **Melody-from-mix fallback:** Melodia salience-based melody extraction — Salamon & Gómez, 2012; source separation pre-step via Demucs (Défossez et al.) when accompaniment drowns melody.
- **Playability reduction:** input-output HMM with Viterbi decoding — Hori, Kameoka & Sagayama, 2013 (project PDF); violin state-space precedent from Maezawa et al., 2012 (project PDF). Chosen over Tuohy & Potter's 2005 GA (stochastic, guitar-idiom fitness) and Matos 2025 AutoTab graph+CNN (discrete-fret assumption, weaker accuracy, code unavailable) per the three-paper comparison.
- **Hardware feasibility bounds:** GhostPlay — Kamatani et al., 2022 (project PDF): bow-inclination double-stop band, rolled triple stops.
- **Timbre descriptors:** Essentia (MTG-UPF) HarmonicPeaks/Tristimulus/Inharmonicity algorithms.

## Open Questions / Risks
- **SysID dependency (TBD):** the acoustics→motion mapper lives in MotionPlanner and needs the future [finger position × bow speed × pressure] → [pitch × amplitude × harmonics] sweep; MelodyExtractor only guarantees a stable feature schema. Guettler/Schelleng diagrams as prior until real data exists.
- **Reducer cost weights (TBD):** string-crossing and position-shift costs need tuning against real hardware kinematics from Firmware/MotionPlanner.
- **3-note chords:** GhostPlay indicates true triple stops are infeasible with one bow; "rolled" semantics for onset/duration must be agreed with MotionPlanner.
- **Polyphonic accuracy risk:** chord-level transcription accuracy (60–78 % in comparable systems) may bottleneck stages 2–3; mitigation: Demucs pre-separation, confidence gating, human-in-the-loop review.
- **Dynamics calibration:** mapping amplitude to bow parameters is undefined until SysID; export raw dB envelopes, not abstract velocity only.
- **Vinyl/streaming capture chain** (phono preamp, Bluetooth sink) deferred to a hardware I/O module.

## Explicitly Out of Scope
- Fingering/bowing choice, bow pressure/speed/vibrato planning → MotionPlanner/Sound2Motion.
- SysID sweeps and the acoustics→actuation inverse model → MotionPlanner + hardware.
- Closed-loop listening/correction during robot play → AudioFeedback.
- Motor control, firmware, electronics → Firmware (AachenBQ/Motor_Architecture).
- Real-time/streaming extraction; live latency guarantees.
- Streaming-service integration (Spotify/YouTube clients), Bluetooth/AUX/phono capture electronics.
- Score-level musicology (key detection, expressive phrasing models, MIDI-DDSP-style expression synthesis).
- Training custom neural transcription models; only pretrained/off-the-shelf models are used.
