---
name: model-routing
description: Route work to the right Claude model (Fable 5, Opus, Sonnet, Haiku) inside Claude Code for this repo. Consult whenever starting a significant task, spawning subagents, debating architecture, doing bulk file exploration, or when the user mentions cost, speed, model choice, /model, or "which model". Also defines which subagents auto-run on cheaper models.
---

# Model Routing for This Repo

Verify current model names/behavior at https://code.claude.com/docs/en/model-config if anything below seems stale.

## Tiers (fastest/cheapest → deepest)
- **haiku** — near-instant; lookups, grep-style exploration, renames, running existing test suites.
- **sonnet** — default tier; implementing well-specified functions, writing tests, refactors, CLI plumbing. **Repo default.**
- **opus** — deep multi-file reasoning, tricky debugging, large refactors.
- **fable (Fable 5)** — Anthropic's newest top tier, above Opus. Reserve for: HMM reducer cost-function design, cross-module interface decisions (NoteSequence schema changes), interpreting the DSP papers, anything touching decisions.md-level architecture.

## How switching works (all verified mechanisms)
- **Per session:** `claude --model <alias|full-id>`; mid-session via `/model`. Switching keeps conversation history.
- **Persistent default:** `"model"` in `.claude/settings.json`, or `ANTHROPIC_MODEL` env var. Full IDs if needed: `claude-fable-5`, `claude-opus-4-8`, `claude-sonnet-4-6`, `claude-haiku-4-5-20251001`.
- **Automatic — subagents:** `model:` field in `.claude/agents/*.md` frontmatter. This repo ships `explorer` (haiku) and `paper-reader` (sonnet); delegation to them switches models with no manual action. `CLAUDE_CODE_SUBAGENT_MODEL` sets the default for agents using `inherit`.
- **Automatic — skills/commands:** `model:` frontmatter in a skill or command pins the model whenever it runs.
- **`opusplan` alias:** plan in Opus, execute in Sonnet — good fit for "design the reducer change, then implement it".
- **`/fast`:** toggles the faster tier for grunt-work stretches.
- **Effort levels** combine with tier (low effort haiku ≈ instant; max effort fable = deepest); tune both, not just tier.

## Repo-specific routing table
| Task | Model |
|---|---|
| Find where X is defined; list fixture files; run pytest | haiku (delegate to `explorer`) |
| Summarize/quote the project PDFs (Hori, Maezawa, Kamatani…) | sonnet (delegate to `paper-reader`) |
| Implement input_adapter / schema / cli / soundsim | sonnet |
| Debug mir_eval regressions across transcriber+reducer | opus |
| Design/change reducer HMM costs, stage gates, NoteSequence schema | fable (plan) → sonnet (implement), or `opusplan` |
| Bulk mechanical edits, docstrings, renames | haiku or `/fast` |

## Session pattern that works here
1. Start `claude --model sonnet` (or `opusplan` for design days).
2. Main session = judgment + synthesis; push exploration/paper-lookup down to the cheaper subagents automatically.
3. Escalate the *session* to fable only for architecture debates; drop back after. `/model` mid-session is cheap — history survives.
4. Never run fable/opus on tasks whose output you'll diff mechanically (formatting, boilerplate) — the quality delta is ~zero there.
