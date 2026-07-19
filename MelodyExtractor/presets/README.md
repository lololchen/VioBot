# Presets

A **preset** is a named, commented JSON snapshot of the four pipeline config
dataclasses (`MonoConfig`, `TimbreConfig`, `StageConfig`, `RenderConfig`).
Presets exist so you can tune parameters interactively (in the GUI, or by
hand) and hand the robot pipeline a fixed, reviewable, version-controllable
set of numbers — without ever touching the code defaults themselves. Code
defaults are the source of truth for "how the pipeline behaves out of the
box"; a preset only records the fields you deliberately chose to override.

This directory holds preset JSON files. None are required for the pipeline
to run — every CLI subcommand and every GUI control falls back to the plain
code defaults when no preset is given.

## Schema

```json
{
  "preset_schema_version": "1",
  "name": "example",
  "comment": "free-text note on why this preset exists",
  "configs": {
    "mono":   { "...": "MonoConfig fields" },
    "timbre": { "...": "TimbreConfig fields" },
    "stage":  { "...": "StageConfig fields" },
    "render": { "...": "RenderConfig fields" }
  }
}
```

Rules (enforced by `melody_extractor/config_io.py`):

- **Missing fields fall back to the dataclass default.** A preset only needs
  to list the fields it actually overrides; you never have to restate the
  whole config.
- **Unknown fields are dropped silently.** In particular, `StageConfig` also
  exposes a computed `config_dict()` (used to stamp `NoteSequence.meta.stage`
  after reduction) which adds a `"reducer_version"` entry that is not a real
  `StageConfig` field — you can feed that dict straight back into a preset's
  `"stage"` section and it will be ignored on load.
- **List values are coerced back to tuples** wherever the corresponding field
  default is itself a tuple (currently only `StageConfig.open_strings_hz`).
- **Only major version `1` is accepted.** A `preset_schema_version` of `"2.x"`
  (or any other major version) is rejected with a clear error — load a preset
  written for an incompatible schema and you get a loud failure, not silently
  wrong config values.
- **Files are byte-deterministic.** `save_preset` writes sorted keys, 2-space
  indent, UTF-8, LF line endings, and a trailing newline — saving the same
  `Preset` twice produces byte-identical files, and diffs stay clean.
- **No absolute paths, no timestamps.** Preset files never embed machine- or
  time-specific data (repo-wide determinism rule).

## Using a preset

Every CLI subcommand accepts `--config PATH`:

```sh
melody-extractor extract song.wav --config presets/bright_bow.json
melody-extractor reduce song.json --stage 2 --config presets/bright_bow.json
melody-extractor render song.stage2.json --config presets/bright_bow.json
melody-extractor eval --fixtures tests/fixtures --report out/eval.json --config presets/bright_bow.json
```

**Precedence is always: explicit CLI flag > preset value > code default.**
For example, `extract --backend yin --config presets/bright_bow.json` uses
the preset's `mono` config for everything *except* `backend`, which the
explicit `--backend yin` flag overrides.

On `reduce`, `--stage N` always overrides the preset's `StageConfig.max_voices`
to `N` — the preset supplies everything else (open strings, cost weights,
tolerances, ...).

On `eval`, a preset supplies the reducer weights/tolerances used at every
stage (`max_voices` is still swept 1/2/3 to match the hardware stage gates).
Because the eval baseline (`out/eval_baseline.json`) must stay comparable
across runs, `eval --config PATH` only ever *compares* against an existing
baseline — if no baseline exists yet and the preset differs from code
defaults, the report is written but no baseline is auto-created; run `eval`
without `--config` first to establish one.

The GUI (Tab 1's sidebar) can export the sliders' current values as a preset
at any time; it never writes the eval baseline itself (that stays CLI/
`docs/decisions.md`-governed, per D-016).
