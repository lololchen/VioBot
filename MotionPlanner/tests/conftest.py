from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "fixtures"))

import generate_fixtures  # noqa: E402  (fixtures/ helper, not a package)

from melody_extractor.schema import NoteSequence  # noqa: E402
from motion_planner.profile_io import load_profile  # noqa: E402

PROFILES_DIR = Path(__file__).resolve().parents[1] / "profiles"


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    """Ensure the reduced-NoteSequence fixture corpus exists (generate on demand)."""
    for name in generate_fixtures.STAGES:
        if not generate_fixtures.fixture_path(name).exists():
            generate_fixtures.generate(name)
    return generate_fixtures.FIXTURES_DIR


@pytest.fixture(scope="session")
def fixture_sequences(fixtures_dir) -> "dict[str, NoteSequence]":
    return {name: NoteSequence.from_json(generate_fixtures.fixture_path(name))
            for name in generate_fixtures.STAGES}


@pytest.fixture(scope="session")
def profiles() -> dict:
    return {name: load_profile(PROFILES_DIR / f"{name}.json")
            for name in ("concept_a_1finger", "concept_a_2finger", "concept_b_4finger")}
