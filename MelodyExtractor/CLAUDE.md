# MelodyExtractor — Module Guide

Perception front-end: arbitrary music → hardware-agnostic note sequence with timbre features → playability-reduced voices → (optional) audio preview. Full spec: `../docs/PRD_MelodyExtractor.md`.

## Layout
```
melody_extractor/
  input_adapter.py   # ffmpeg decode → 16 kHz mono float32 PCM; MIDI → NoteSequence directly
  transcriber.py     # mono: CREPE (f0, confidence); poly: basic-pitch (notes, bends, velocity)
  timbre.py          # Essentia: HarmonicPeaks, odd/even ratio, tristimulus, inharmonicity per note
  reducer.py         # HMM/Viterbi voice selection; N + adjacency from StageConfig
  schema.py          # NoteSequence dataclasses + versioned JSON (de)serialization
  soundsim.py        # FluidSynth+violin SoundFont render; optional DDSP tone-transfer preview
  cli.py             # extract | reduce | render | eval  (--config PATH loads a preset)
  config_io.py       # preset JSON (schema v1) ⇄ frozen config dataclasses; streamlit-free
  url_fetch.py       # [url] extra: YouTube/SoundCloud via yt-dlp, Spotify via oEmbed→ytsearch1; streamlit-free (D-017)
  eval_harness.py    # run_eval/compare_to_baseline — eval computation shared by CLI + GUI
  gui/               # [gui] extra: Streamlit app (app.py), figures, caching, views, launch
tests/
  fixtures/          # ground-truth MIDI + rendered WAV pairs
presets/             # named preset JSONs exported from the GUI, fed to --config
```

## Contracts (do not break)
- `NoteSequence` JSON is the interface consumed by MotionPlanner/AudioFeedback — schema changes require a version bump + `docs/decisions.md` entry.
- Every note: `pitch_hz` (float, not just MIDI int — violin is continuous-pitch), `onset_s`, `duration_s`, `amp_db_envelope`, optional `f0_contour` (for bends/vibrato), optional `harmonics` block.
- Reducer is pure: `(NoteSequence, StageConfig) → NoteSequence`. No I/O, no model calls inside.
- MIDI input path must not round-trip through audio.

## Algorithm anchors (see PRD for citations)
- CREPE = mono f0; basic-pitch = poly notes; Demucs pre-separation only when melody is buried; Melodia as salience fallback.
- Reducer follows Hori 2013 input-output HMM (Viterbi), violin state space per Maezawa 2012. Papers are in the project PDFs — read them before modifying cost functions.
- Physical feasibility limits (double-stop inclination band, rolled triples) come from Kamatani 2022 GhostPlay.

## Gotchas
- CREPE/basic-pitch resample internally — always pass 16 kHz mono to CREPE, and let basic-pitch use 22050 Hz; never assume shared sample rate.
- basic-pitch velocity is amplitude-scaled 0–127; keep the raw dB envelope alongside it (SysID will need real levels).
- pip installs here need TensorFlow only for the poly path — keep `[poly]` extra separate so mono POC installs stay light.
- Determinism: CREPE Viterbi is deterministic; basic-pitch is deterministic; anything stochastic (future) must take an explicit seed.
- Rendered previews must be exported as *pairs* (original excerpt + rendered) with matched loudness for fair listening.
- Write WAVs with `scipy.io.wavfile`, never soundfile/libsndfile: float WAVs get a timestamped PEAK chunk that breaks byte-determinism (D-013). Reading with soundfile is fine.
- `gui/` is the only place allowed to import streamlit/plotly — never import `gui` from core modules. The GUI writes only preset JSONs (and user-triggered report downloads); it must never write `eval_baseline.json` or eval reports (D-016).
- Custom CSS lives ONLY in `gui/style.py` (D-017) — never inline `st.markdown(<style>)` elsewhere. URL downloads go to `%TEMP%` + session memory, never under the (OneDrive-synced) repo.

## Definition of done for any pipeline change
Run `algorithm-validation` skill: mir_eval metrics vs fixtures + paired listening files + a `docs/decisions.md` entry if behavior changed.
