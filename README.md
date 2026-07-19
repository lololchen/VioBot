# Viobot — a robot that plays the violin

Viobot listens to arbitrary music (audio file, MIDI, or a YouTube/SoundCloud/Spotify link) and replays it on a real violin. This repo holds the software pipeline; motor firmware lives in a separate repo.

## Modules

| Module | Role | Status |
|---|---|---|
| [MelodyExtractor/](MelodyExtractor/) | Music → hardware-agnostic `NoteSequence` (pitch, timing, dynamics, timbre) → playability-reduced voices → audio preview | **Active** |
| `MotionPlanner/` | `NoteSequence` → fingering, bow speed/pressure/position, vibrato | Planned |
| `AudioFeedback/` | Closed-loop listening while the robot plays | Planned |
| Firmware | Motor control — [AachenBQ/Motor_Architecture](https://github.com/AachenBQ/Motor_Architecture) | External repo |

The module boundary is strict: MelodyExtractor outputs only acoustics (a versioned `NoteSequence` JSON schema). Anything involving fingers, bows, or motors belongs downstream in MotionPlanner.

## MelodyExtractor pipeline

```
audio / MIDI / URL
      │
      ▼
input_adapter ──► transcriber ──► timbre ──► reducer ──► NoteSequence JSON
 (ffmpeg decode)   (CREPE mono /   (harmonic  (HMM/Viterbi              │
                    basic-pitch     features    playability             ▼
                    poly; YIN       per note)   reduction)          soundsim
                    fallback)                                (FluidSynth preview)
```

- **Transcription:** CREPE for monophonic f0, basic-pitch for polyphony; a dependency-free YIN fallback runs when the DNN extras aren't installed.
- **Reduction:** an input-output HMM (Viterbi) selects which voices a violin can physically play, following Hori 2013 with the violin state space of Maezawa 2012 and feasibility limits from Kamatani 2022.
- **Determinism:** same input file ⇒ byte-identical output. The robot is debugged against these outputs.

Hardware stage gates (only the `reducer` config changes per stage):
mono → 2-note adjacent-string double stops → 3-note rolled chords → 4-finger/2-bow.

## Quickstart (Windows)

Double-click **`run_gui.bat`** — it creates a local `.venv`, installs the package with GUI + ffmpeg extras on first run, and starts the Streamlit GUI in your browser.

Manual setup (any OS, Python ≥ 3.10):

```bash
python -m venv .venv
.venv/Scripts/activate          # Linux/macOS: source .venv/bin/activate
pip install -e "MelodyExtractor[gui,ffmpeg]"

melody-extractor-gui            # parameter-tuning / inspection GUI
melody-extractor extract song.mp3 -o out.json    # CLI: extract | reduce | render | eval
melody-extractor extract song.mp3 --config MelodyExtractor/presets/<preset>.json
```

ffmpeg must be on PATH (or use the `[ffmpeg]` extra, which bundles a static build).

### Optional extras

Core stays importable without any of these — install only what you need:

| Extra | Enables | Pulls in |
|---|---|---|
| `gui` | Streamlit GUI (`melody-extractor-gui`) | streamlit, plotly |
| `ffmpeg` | Bundled ffmpeg when none on PATH | imageio-ffmpeg |
| `mono-dnn` | CREPE monophonic transcription | TensorFlow |
| `poly` | basic-pitch polyphonic transcription | TensorFlow |
| `render-fluidsynth` | Audio preview rendering | pyfluidsynth + FluidSynth DLL + a violin SoundFont |
| `url` | YouTube/SoundCloud/Spotify input | yt-dlp |
| `timbre-essentia` | Essentia timbre backend (numpy backend is the default) | essentia (no Windows wheels) |
| `dev` | Test suite | pytest |

## Tests

```bash
cd MelodyExtractor
pytest                                   # fast suite
pytest -m slow                           # full Streamlit AppTest runs
MELODY_EXTRACTOR_NETWORK_TESTS=1 pytest -m network   # real yt-dlp downloads
```

Transcription accuracy is measured with `mir_eval` against ground-truth fixtures in `MelodyExtractor/tests/fixtures/`; the regression baseline lives at `MelodyExtractor/out/eval_baseline.json` and is only updated deliberately (see `docs/decisions.md`).

## Documentation

- [docs/PRD_MelodyExtractor.md](docs/PRD_MelodyExtractor.md) — product requirements; read before changing MelodyExtractor.
- [docs/decisions.md](docs/decisions.md) — append-only architectural decision log.
- [docs/plan_GUI_MelodyExtractor.md](docs/plan_GUI_MelodyExtractor.md) — GUI design plan.
- [MelodyExtractor/CLAUDE.md](MelodyExtractor/CLAUDE.md) — module contracts and gotchas.

## Repo layout

```
run_gui.bat            # one-click GUI launcher (Windows)
CLAUDE.md              # repo guide for AI-assisted development
docs/                  # PRD, decision log, plans (reference-paper PDFs are local-only, not versioned)
MelodyExtractor/
  melody_extractor/    # the package (adapter, transcriber, timbre, reducer, schema, soundsim, cli, gui/)
  presets/             # named preset JSONs exported from the GUI, fed to --config
  tests/               # pytest suite + ground-truth MIDI/WAV fixtures
```
