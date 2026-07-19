"""Generate MotionPlanner's NoteSequence fixtures from MelodyExtractor's MIDI corpus.

Each fixture is the deterministic result of MelodyExtractor's MIDI path +
reducer at the stage the piece was designed for (the corpus is stage-mapped by
construction — see MelodyExtractor/tests/fixtures/generate_fixtures.py):

    mono_scale        -> stage 1      mono_arpeggio -> stage 1
    two_voice_thirds  -> stage 2      triple_rolled -> stage 3

Byte-pinned: regenerating always produces identical JSON (MIDI load and the
reducer are both deterministic), so the corpus doubles as a cross-package
determinism probe. Regenerate: python MotionPlanner/tests/fixtures/generate_fixtures.py
"""
from __future__ import annotations

from pathlib import Path

from melody_extractor.input_adapter import load_midi
from melody_extractor.reducer import StageConfig, reduce as reduce_seq

FIXTURES_DIR = Path(__file__).resolve().parent
MELODY_FIXTURES_DIR = FIXTURES_DIR.parents[2] / "MelodyExtractor" / "tests" / "fixtures"

STAGES = {
    "mono_scale": 1,
    "mono_arpeggio": 1,
    "two_voice_thirds": 2,
    "triple_rolled": 3,
}


def fixture_path(name: str) -> Path:
    return FIXTURES_DIR / f"{name}.stage{STAGES[name]}.json"


def generate(name: str) -> Path:
    seq = load_midi(MELODY_FIXTURES_DIR / f"{name}.mid")
    reduced = reduce_seq(seq, StageConfig.stage(STAGES[name]))
    out = fixture_path(name)
    reduced.to_json(out)
    return out


def generate_all() -> "list[Path]":
    return [generate(name) for name in STAGES]


if __name__ == "__main__":
    for path in generate_all():
        print(f"wrote {path}")
