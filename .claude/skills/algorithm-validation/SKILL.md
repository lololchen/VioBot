---
name: algorithm-validation
description: Validate any melody-extraction, transcription, timbre, or reduction algorithm change in this repo before declaring it done. Use this skill whenever code in MelodyExtractor's transcriber, reducer, timbre, or soundsim changes, whenever a new model/library version is pinned, whenever the user asks "does it work", "how accurate", "compare pipelines", or before any commit claiming improved extraction quality — even if the user doesn't explicitly ask for validation.
---

# Algorithm Validation

No extraction/reduction change is "done" on eyeballed output. Done = metrics + paired listening files + decision-log entry.

## 1. Quantitative — mir_eval against fixtures

Fixtures live in `MelodyExtractor/tests/fixtures/`: each case is `{name}.mid` (ground truth) + `{name}.wav` (rendered or recorded audio). Start with synthetic renders (scales, arpeggios, two-voice pieces at each stage-N), add real recordings later.

```bash
python -m melody_extractor.cli eval --fixtures MelodyExtractor/tests/fixtures --report out/eval.json
```

Report per fixture and aggregate:
- **Mono f0:** `mir_eval.melody` — RPA (raw pitch accuracy), RCA (chroma), overall accuracy, voicing recall/false-alarm. Regression gate: RPA must not drop >1 pt vs `out/eval_baseline.json`.
- **Note events:** `mir_eval.transcription` — onset F1 (50 ms tolerance), onset+pitch F1, onset+offset+pitch F1.
- **Reducer:** playability violations must be **0** (voices > StageConfig.N, non-adjacent double-stop strings); report melody-retention (fraction of ground-truth top-voice notes surviving reduction).
- **Timbre:** compare odd/even ratio + tristimulus per note against fixture-rendered values; report mean absolute deviation (sanity check, loose gate).

Baseline handling: first run writes `eval_baseline.json`; subsequent runs diff against it. Updating the baseline requires a decisions.md entry.

## 2. Qualitative — paired listening
For ≥2 fixtures and ≥1 real-world track:
```bash
python -m melody_extractor.cli render --paired out/listen/{name}/
```
Emit `original.wav` + `extracted_render.wav`, loudness-matched. Present the file paths to the user and ask them to listen — never claim perceptual quality yourself.

## 3. Determinism check
Run extraction twice on one fixture; outputs must be byte-identical (`sha256sum`). Any diff is a failing result.

## 4. Record
If behavior or metrics changed: append a `docs/decisions.md` entry (what changed, metric deltas, why accepted). Include the eval.json aggregate table in the PR/commit message.

## Failure triage order
1. Input adapter (sample rate, mono-mix, clipping) — check first, it causes most "model is bad" illusions.
2. Model confidence gating thresholds.
3. Note segmentation (min-duration, merge gaps).
4. Only then the model/algorithm itself.
