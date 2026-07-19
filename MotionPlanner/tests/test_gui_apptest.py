"""Headless AppTest smoke run (slow marker, mirrors MelodyExtractor's)."""
from pathlib import Path

import pytest

streamlit = pytest.importorskip("streamlit")

APP_PATH = Path(__file__).resolve().parents[1] / "motion_planner" / "gui" / "app.py"


@pytest.mark.slow
def test_app_runs_headless(fixtures_dir):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP_PATH), default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    # Sidebar renders the profile selector + parameter groups.
    assert at.sidebar.selectbox
    # The plan tab ran on the default input: metrics appear.
    assert any("Feasibility" in str(m.label) for m in at.metric), \
        [str(m.label) for m in at.metric]


@pytest.mark.slow
def test_app_no_errors_on_default_state(fixtures_dir):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP_PATH), default_timeout=120)
    at.run()
    errors = [str(e.value) for e in at.error]
    assert not errors, errors
