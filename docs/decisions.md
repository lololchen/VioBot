# Decision Log

Append-only. Newest at bottom. Format: date, decision, alternatives considered, rationale, revisit-when.

---

## 2026-07 D-001 — Single transcriber + swappable reducer (not one extractor per hardware stage)
**Decision:** One faithful transcription stage; hardware stages (1/2/3-note, 2-bow) only change the reducer's `StageConfig`.
**Alternatives:** Separate extraction modules per hardware generation.
**Rationale:** Transcription is perception; playability is constrained optimization. Coupling them forces rewrites per hardware rev.
**Revisit when:** Never expected; reducer interface may grow.

## 2026-07 D-002 — Output stays hardware-agnostic acoustics
**Decision:** MelodyExtractor emits pitch/onset/duration/dynamics/harmonics only. String/finger/bow choices live in MotionPlanner.
**Alternatives:** Emit fingering directly.
**Rationale:** Keeps extraction reusable across hardware; SysID (acoustics→actuation) doesn't exist yet.
**Revisit when:** SysID data arrives and MotionPlanner interface solidifies.

## 2026-07 D-003 — Batch/offline processing, no real-time path
**Decision:** Buffer a phrase/track, process behind playback.
**Alternatives:** Streaming low-latency extraction.
**Rationale:** Mechanical bow/finger can't react instantly anyway; batch is far simpler and deterministic.
**Revisit when:** AudioFeedback needs online correction.

## 2026-07 D-004 — Reduction algorithm: input-output HMM + Viterbi (Hori 2013)
**Decision:** HMM/Viterbi over GA (Tuohy 2005) and graph+CNN (Matos 2025 AutoTab).
**Alternatives:** GA — stochastic, guitar-idiom fitness function, no violin precedent. Graph+CNN — discrete-fret assumption, mediocre chord accuracy (60–78 %), claimed-open-source code not findable.
**Rationale:** Only approach with a published violin adaptation (Maezawa 2012, HMM over continuous fingerboard positions); deterministic; O(T·N²); cheapest to reimplement from paper; interpretable per-cost tuning across hardware stages. None of the three papers' original code was accessible — reimplementation cost drove the choice.
**Revisit when:** Reducer quality plateaus; then consider learned costs.

## 2026-07 D-005 — Idea 1 (direct feature extraction) over Idea 2 (DDSP timbre-transfer first)
**Decision:** Extract f0/amplitude/harmonics directly from source audio (CREPE/basic-pitch + Essentia). Rejected: routing audio through a pretrained DDSP violin model before extraction.
**Alternatives:** DDSP/Tone Transfer pipeline (Idea 2).
**Rationale:** DDSP conditions only on f0+loudness — it discards the source's real harmonic distribution and re-hallucinates the training violin's (someone else's instrument, not our robot). Standard DDSP is monophonic-only, breaking the chord requirement. Idea 2's effort is a strict superset of Idea 1's (it embeds the same f0/loudness extraction). "Violin-shaped prior" benefit is fake until it matches our SysID manifold.
**Revisit when:** If a polyphonic DDSP trained on *our robot's own* SysID recordings exists — then reconsider as the mapper's forward model.
**Fallback agreed:** If contested, build both pipelines and score both by distance to the SysID-measured achievable manifold.

## 2026-07 D-006 — DDSP retained for SoundSim previews only
**Decision:** FluidSynth + solo-violin SoundFont is the baseline renderer; pretrained Magenta violin Tone Transfer is an optional higher-realism preview.
**Rationale:** Rendering is exactly the direction DDSP is good at (control signals → violin audio); no information-loss concern downstream of extraction.

## 2026-07 D-007 — Transcriber choices: CREPE (mono), basic-pitch (poly), Demucs (pre-separation), Melodia (salience fallback)
**Rationale:** CREPE: SOTA mono f0 + confidence, deterministic Viterbi smoothing. basic-pitch: lightweight, polyphonic, instrument-agnostic, pitch bends, amplitude-scaled velocity, maintained by Spotify. Demucs only when the melody is buried in a mix. All are pip-installable with pretrained weights — no training in scope.

## 2026-07 D-008 — Timbre features via Essentia classical DSP (not learned embeddings)
**Decision:** HarmonicPeaks, odd/even harmonic energy ratio, tristimulus, inharmonicity per note.
**Rationale:** Interpretable, deterministic, directly comparable against future SysID sweep outputs (bow speed/force ↔ harmonic distribution); no training data needed.

## 2026-07 D-009 — 3-note chords are "rolled", not simultaneous (single-bow stages)
**Rationale:** GhostPlay (Kamatani 2022) measured a narrow bow-inclination band for adjacent-string double stops and never implemented them; curved bridge makes sustained triple stops infeasible with one bow. Onset/duration semantics for rolled chords TBD with MotionPlanner.
**Revisit when:** Two-bow hardware ("Next Gen") lands — Mills Violano-Virtuoso precedent shows per-string bowing enables true simultaneity.

## 2026-07 D-010 — Input capture order: files first, capture hardware later
**Decision:** POC ingests files (USB-drive scenario) + MIDI. Bluetooth A2DP / AirPlay / AUX-ADC / phono chain deferred to a hardware I/O module; buy (USB audio interface, RIAA preamp), don't build, the analog front-end.
**Rationale:** All paths converge to PCM; file path exercises 100 % of the extraction pipeline with 0 % of the driver work.

## 2026-07-18 D-011 — Dependency-light deterministic fallbacks: YIN (mono f0), numpy timbre DSP, additive-synth renderer
**Decision:** Each heavy-model stage ships with an always-available, pure-numpy fallback backend: mono f0 via a deterministic YIN implementation (de Cheveigné & Kawahara 2002), timbre via numpy STFT harmonic analysis implementing the Essentia formulas, rendering via additive synthesis. CREPE / basic-pitch / Essentia / FluidSynth remain the quality path as optional extras (`[mono-dnn]`, `[poly]`, `[timbre-essentia]`, `[render-fluidsynth]`); backend="auto" prefers them when installed.
**Alternatives:** Hard-require the PRD models (breaks Windows dev — Essentia has no Windows wheels, FluidSynth needs a system DLL, CREPE/basic-pitch pull TensorFlow — and makes CI heavy); librosa pyin (adds numba/llvmlite, no clear win over own YIN for this purpose).
**Rationale:** Repo rules demand core importable without heavy deps and deterministic, testable stages; fallbacks let the full pipeline + test suite run end-to-end with core deps only on any machine. Extraction *quality* claims still go through the PRD backends per the algorithm-validation skill.
**Revisit when:** Accuracy gates require CREPE/basic-pitch by default (then make an extras-install the documented default instead).

## 2026-07-18 D-012 — NoteSequence schema v0.1.0 shipped
**Decision:** Initial versioned schema in `melody_extractor/schema.py`: top-level `{schema_version, notes[], features[], meta}`; notes carry pitch_hz (float), onset_s, duration_s, amp_db_envelope (real dBFS), optional velocity/confidence/voice/rolled/f0_contour/harmonics; features are per-frame f0/voicing/amp tracks; meta records source, backends+versions, and post-reduction StageConfig. Frozen dataclasses (reducer purity); deterministic JSON (sorted keys, canonical note order by onset/pitch/duration, NaN/Inf rejected); major-version gate on load; lossy MIDI export (nearest semitone + per-note pitch bend > 5 cents, one instrument per voice).
**Rationale:** Encodes the module contracts already fixed in MelodyExtractor/CLAUDE.md; schema changes henceforth require a version bump + a new entry here.
**Revisit when:** MotionPlanner integration surfaces missing fields (expected: rolled-chord timing semantics, per D-009).

## 2026-07-18 D-013 — WAV files are written via scipy.io.wavfile, not libsndfile/soundfile
**Decision:** All WAV *writing* (fixtures, SoundSim renders) uses `scipy.io.wavfile.write`. Reading may use soundfile.
**Rationale:** libsndfile embeds a PEAK chunk in float WAVs containing a Unix timestamp (verified: byte 60 differs between two writes ~1 s apart), which silently violates the byte-identical-output requirement. scipy's writer is verified byte-deterministic.
**Revisit when:** We need WAV subtypes scipy can't write (then strip/normalize the PEAK chunk explicitly).

## 2026-07-18 D-014 — Citation correction for D-009 + adopted Maezawa 2012 constants
**Decision:** D-009's *decision* (3-note chords are rolled on single-bow stages) stands, but its stated rationale over-attributed to GhostPlay. Paper verification (paper-reader pass over the project PDFs) found: Kamatani 2022 states multiple stops were "possible from a mechanical viewpoint" but unimplemented, publishes NO numeric bow-inclination band (qualitative Figure 12 only), and contains no rolled-chord timing semantics. Basis for D-009 is therefore: curved-bridge violin acoustics + our single-bow hardware roadmap, with GhostPlay as qualitative precedent only.
**Also adopted from Maezawa 2012 into the reducer defaults:** melody-continuity transition cost shaped as (Δsemitones/7)² after their horizontal–vertical model p(Si|Sj; v) ∝ exp(−v((Δp/7)² + Δs²)) (p. 63, v=30 tuned); cross-string finger-span default 5 semitones = their Pnat(index,little), hard max 6 = Pmax (p. 65 playability matrices). Note-level (not frame-level) Viterbi time steps match their design. Neither paper covers polyphonic reduction itself — our interval-grid subset-state HMM is the Hori 2013 shape, reimplemented (D-004).
**Revisit when:** Real hardware kinematics replace paper-derived span/cost constants (PRD open question).

## 2026-07-19 D-015 — Exact-ET open-string constants + cents-tolerance feasibility boundaries; first eval baseline
**Decision:** `reducer.OPEN_STRINGS_HZ` uses exact equal-temperament values (195.99771799087463 / 293.6647679174076 / 440.0 / 659.2551138257398), and all range/position feasibility comparisons moved to the semitone domain with `StageConfig.pitch_tolerance_semitones = 0.3` slack.
**Trigger:** The mir_eval harness's melody-retention metric caught the rounded literal `196.0` rejecting exact-ET G3 notes (195.9977 Hz sits below it by more than the old 1e-6 Hz epsilon), silently dropping them even at stage 1 (mono_arpeggio retention 6/7). A Hz-domain epsilon is the wrong unit for pitch; transcription jitter of a few cents must never disqualify a nominally playable note.
**Metrics after fix (fixture corpus, yin backend):** melody retention 1.0 for all 4 fixtures × stages 1–3; playability violations 0 everywhere; aggregate mono RPA 0.9898, onset/onset+pitch/onset+offset+pitch F1 = 1.0; extraction byte-determinism true for every fixture. First `eval_baseline.json` written (MelodyExtractor/out/) from this run — future baseline updates require a new entry here.
**Revisit when:** Hardware intonation data suggests a different tolerance (0.3 st is provisional).

## 2026-07-19 D-016 — Streamlit GUI (Pipeline Inspector + Eval Dashboard) with preset-JSON export; CLI gains `--config`
**Decision:** A Streamlit + plotly GUI ships as the optional `[gui]` extra (`melody_extractor/gui/`, entry point `melody-extractor-gui`). Two tabs: Pipeline Inspector (stage-by-stage visualization captioned with the module.function it shows, live parameter tweaking over a cached pipeline chain, A/B audio) and Eval Dashboard (runs the mir_eval harness with the current sidebar configs, compares against the baseline read-only). Good parameter sets are exported as named preset JSONs in `MelodyExtractor/presets/` (schema v1, byte-deterministic), consumed by a new `--config PATH` flag on `extract|reduce|render|eval` with precedence explicit flag > preset > code default. Supporting refactor: eval computation moved out of `cli.py` into `eval_harness.run_eval` (byte-identical output, regression-guarded) and preset I/O lives in `config_io.py` — both core-level, streamlit-free.
**Alternatives:** Gradio — weaker layout control for the multi-panel inspector, cache semantics poorer fit than `st.cache_data` over frozen configs. PySide6/Qt — heavyweight desktop dep, no notebook-style incremental dev, overkill for an internal inspection tool. Jupyter notebooks — no persistent widget→config discipline, encourages divergent per-user copies, not testable via anything like AppTest.
**Rationale:** Goals are understand/judge/adjust, not production UI; Streamlit's rerun+cache model maps directly onto the frozen-config pipeline (hashable cache keys), and `AppTest` gives headless CI coverage. Preset files keep code defaults immutable and make good settings reproducible through the same CLI the robot is debugged against.
**Explicitly excluded:** The GUI never writes `eval_baseline.json` or eval reports — baseline updates stay CLI-governed (`melody-extractor eval --config` + a new decisions.md entry, per D-015). GUI writes are limited to preset JSONs and user-initiated report downloads. `open_strings_hz` is not exposed as a GUI knob (constants are decision-governed per D-015).
**Revisit when:** The GUI needs multi-user or remote deployment (then revisit framework), or MotionPlanner wants its own inspection panels (new module, not scope creep into this one).

## 2026-07-19 D-017 — URL audio ingestion (yt-dlp) + first custom GUI CSS (mobile pass)
**Decision:** The GUI gains a URL input source: YouTube/SoundCloud (and anything yt-dlp's extractors cover) are downloaded directly; Spotify links are *never* downloaded (streams are DRM-protected) — the public oEmbed endpoint resolves the track title (metadata only, no auth) and yt-dlp's `ytsearch1:` fetches the best YouTube match, with the resolved video surfaced in the sidebar so the user can verify it (a cover/live version is possible; not fixable without Spotify API auth). Implementation: new streamlit-free core module `url_fetch.py` (CLI-reusable later, like `eval_harness`/`config_io`) behind a new `[url]` extra (`yt-dlp` floor-pinned only — extractors must stay updatable); GUI wraps it in `pipeline_cache.fetch_url_bytes` (`st.cache_data`, `max_entries=8`) and feeds the bytes through the existing `register_bytes` digest flow. Duration is probed *before* download and capped at 15 min; live streams refused. Audio is taken as native `bestaudio` (m4a preferred, never re-encoded); `AUDIO_EXTENSIONS` gains `.webm`/`.opus`, which route through the existing ffmpeg decode branch. Input precedence in the sidebar: upload > URL result > fixture. Second part: first custom CSS in the repo, confined by rule to `gui/style.py` (mobile media-query only: main-block padding, title size, hide plotly modebar) — `st.columns` stacking and `use_container_width` already handle the rest on phones.
**Alternatives:** spotdl for Spotify — heavier dep that does the same YouTube-matching under the hood with less transparency. Re-encoding downloads to wav/mp3 via yt-dlp postprocessor — lossy, second ffmpeg touchpoint, unnecessary since the decode path sniffs containers. Per-widget CSS or a component framework for mobile — blast radius too large for an internal tool; a single media query fixes the real problems.
**Boundaries kept:** Downloads live only in `%TEMP%` + session memory — never written under the (OneDrive-synced) repo, consistent with D-016's GUI-write restrictions. Determinism: URL fetching is input *acquisition*; the pipeline stays byte-deterministic per digest, but re-fetching the same URL later may yield different bytes (platforms re-encode) — anyone debugging against a URL-sourced input should export/keep the bytes. ToS note: downloading from these platforms is the user's responsibility; personal/research use.
**Tests:** `tests/test_url_fetch.py` (pure unit, all network monkeypatched; real download opt-in via `MELODY_EXTRACTOR_NETWORK_TESTS=1` + new `network` marker); AppTest coverage with `fetch_url_bytes` monkeypatched to fixture bytes.
**Revisit when:** yt-dlp breaks repeatedly (consider vendoring a pinned known-good version + scheduled bumps), Spotify matching quality becomes a real problem (then Spotify Web API auth), or the GUI needs a real responsive redesign (then revisit framework per D-016).

## 2026-07-19 D-018 - GUI/pipeline performance hardening for long songs + FluidSynth preview install
**Trigger:** User reports on multi-minute songs: extraction takes minutes, the browser frequently shows "Connection error - Is Streamlit still running?", and a browser refresh cannot recover the app. Root causes found by measurement, not vibes: (1) `_segment_notes` recomputed `np.median` over a growing Python list per frame - O(n^2) in note length, 131 s of a 153 s transcription of a 5-minute song; (2) `yin_track` stacked ALL frames into one matrix - >1 GB of transient float64 arrays on a 5-minute song, enough to page/kill the server process; (3) the Input spectrogram shipped the raw STFT as `.tolist()` JSON (~3M cells, tens of MB) over the websocket on EVERY Streamlit rerun, and frame-track/pianoroll traces were unbounded too; (4) all pipeline caches except `render` were unbounded, so full-song AudioBuffers/NoteSequences/WAV bytes accumulated in RAM for the life of the server. A dead server is unrecoverable from the browser, hence "refresh doesn't restart".
**Decision:**
- `transcriber._segment_notes`: running medians via incremental sorted lists (`bisect.insort` + `_sorted_median`, which reproduces np.median's IEEE arithmetic bit-exactly). Verified value-identical notes on scale/noisy/long-tone/noise signals; 131 s -> 0.3 s.
- `transcriber.yin_track`: frames processed in `_YIN_BLOCK_FRAMES=2048` blocks (~100 MB peak instead of >1 GB). Row-independent math; verified byte-identical f0/voicing/amp on 4 signals incl. a multi-block 45 s tone.
- `gui/figures.py` payload caps: spectrogram time-axis max-pooled to <=1000 columns (float32 z), frame-track and pianoroll f0 overlays stride-decimated to <=4000 points/trace (vrect runs still computed at full resolution), arrays passed to plotly as numpy so plotly>=6 serializes compact base64 instead of JSON number text. Display-only; the pipeline never reads figures.
- Sidebar "Display -> Show charts" toggle (default on): off skips building/sending every plotly chart; transcription, reduction, and A/B audio still run.
- All `pipeline_cache` caches bounded: load max_entries=8, transcribe/timbre/reduce 16, render 32->8.
- `run_gui.bat`: crash auto-restart loop (headless on restarts) so a browser refresh reconnects after a server death instead of dead-ending.
- FluidSynth preview path made real on this machine (PRD D-006 baseline renderer): pyfluidsynth 1.4.0 in the venv, FluidSynth 2.5.6 win64 DLLs copied next to the venv executables, GeneralUser-GS.sf2 (30.8 MB) in `%LOCALAPPDATA%\MelodyExtractor\soundfonts\`. `soundsim._render_fluidsynth` gains fallback SoundFont discovery (`soundfont_path` -> `MELODY_EXTRACTOR_SF2` env -> first .sf2 in that folder) so the GUI works by just switching backend to "fluidsynth". Heavy binaries live outside the OneDrive-synced repo (consistent with D-017's boundary).
**Explicitly NOT changed:** default `RenderConfig.backend` stays "additive" (always-available deterministic fallback per D-011); extraction outputs are bit-identical to before (yin + segmentation verified), so no new eval baseline is needed.
**Revisit when:** Streamlit adds first-class figure diffing (drop the toggle), or a curated solo-violin SoundFont replaces GeneralUser GS (then pin it by name in docs).

## 2026-07-19 D-019 - Numbered pipeline steps with live status/auto-scroll; default demo URL; FluidSynth DLL bootstrap
**Trigger:** User feedback after D-018: wanted (a) a spinner on each pipeline section while it computes, (b) numbered steps that start collapsed, auto-expand when finished, and smooth-scroll so the next still-running step stays visible; (c) the example YouTube link prefilled with no Enter keypress needed; (d) the GUI raised "pyfluidsynth ... neither is installed here" at the A/B step even though D-018 installed everything - and that full-page traceback at the A/B step also read as "extraction crashed".
**Decision:**
- Inspector sections are now numbered `st.status` boxes (1. Input, 2. FrameTrack, 3. Notes, 4. Timbre, 5. Reducer, 6. A/B audio) in `gui/inspector_view.py`; each stage's pipeline_cache call runs INSIDE its box so the header spinner is truthful. A stage whose body took > 0.75 s (`_FRESH_S`, i.e. a real cache miss) auto-expands on completion and smooth-scrolls so the NEXT step's collapsed running header sits ~150 px above the viewport bottom; instant cache hits never touch scroll or the user's open/close state. Anchors + scroll JS live ONLY in `gui/style.py` (`step_anchor`/`scroll_to_anchor`) - style.py remains the single sanctioned home for custom front-end injection (extends D-017's CSS rule to JS/HTML).
- **Default demo link: `DEFAULT_URL` constant near the top of `MelodyExtractor/melody_extractor/gui/app.py` - edit that one line to change the default song in the sidebar URL box.** Prefilling means the Fetch button is enabled immediately (Streamlit text inputs commit on blur, so clicking Fetch works without pressing Enter).
- Root cause of the GUI-only fluidsynth ImportError: pyfluidsynth locates libfluidsynth-3.dll via ctypes' PATH search, and the Streamlit server process does not have the venv Scripts dir (where D-018 put the DLLs) on PATH. `soundsim._import_fluidsynth` now prepends every known DLL location (venv Scripts via sys.executable, `%LOCALAPPDATA%/MelodyExtractor/fluidsynth/*/bin`) to PATH + `os.add_dll_directory` before importing, and the error message now distinguishes package-missing from DLL-not-loadable. Verified: full 44.1 kHz fluidsynth render succeeds in a process whose PATH is stripped to System32 alone.
- The A/B step wraps its two renders in try/except and shows `st.error` inline instead of letting a backend exception paint a full-page traceback (which the user read as a crash/connection loss).
**Revisit when:** Streamlit gives expanders/status boxes a native scroll-into-view API (drop the JS), or the demo link should move into a preset/config file instead of a code constant.

## 2026-07-19 D-020 - Repo published to GitHub (lololchen/viobot_claude); version-control policy
**Trigger:** User asked to put the project on GitHub. First git history for the repo, so the ignore/versioning policy is set here.
**Decision:**
- Remote: `https://github.com/lololchen/viobot_claude`, branch `main`. Root `README.md` added as the public entry point.
- Versioned: all source, tests + WAV/MIDI fixtures (ground truth the accuracy numbers are measured against), presets, docs, `.claude/` skills/agents, `.vscode/settings.json`, `run_gui.bat`, and `MelodyExtractor/out/eval_baseline.json` (baseline updates stay decision-governed per D-015/D-016).
- Ignored: everything else under `MelodyExtractor/out/` (generated listening pairs/eval reports), `.venv`, caches/egg-info, and `docs/*.pdf` - the reference papers are copyrighted, so they stay local-only in case the repo is or becomes public. Papers remain cited by name in CLAUDE.md/PRD; new checkouts must obtain them separately for the paper-reader tooling.
- `.gitattributes`: `* text=auto` with `*.bat` forced CRLF (cmd misparses LF, see run_gui.bat header) and WAV/MID/SF2/PDF marked binary so fixture bytes are never touched by newline normalization (byte-determinism per D-013).
**Revisit when:** The repo moves to a team org (transfer, don't re-create, to keep history), or a rights-cleared way to distribute the papers appears.
