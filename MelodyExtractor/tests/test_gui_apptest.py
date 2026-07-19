"""Headless Streamlit AppTest coverage for the Pipeline Inspector app.

Known AppTest limits (docs/plan_GUI_MelodyExtractor.md "Tests" section): it
cannot simulate `st.file_uploader`, plotly chart *selection* events, or
inspect `st.audio` output. That is exactly why the sidebar carries a fixture
selectbox (drivable here) alongside the uploader, and why every chart-driven
picker (the piano-roll note picker) has a selectbox fallback -- these tests
drive the fallback path, not the (untestable) chart click path.
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

APP_PATH = Path(__file__).resolve().parents[1] / "melody_extractor" / "gui" / "app.py"


def _run_with_fixture(fixture_name: str = "mono_scale.wav") -> AppTest:
    """Boot the app, then drive the sidebar fixture selectbox (the
    file_uploader alternative AppTest *can* simulate) and rerun."""
    at = AppTest.from_file(str(APP_PATH), default_timeout=120)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]

    select = at.sidebar.selectbox(key="input_fixture_select")
    select.set_value(fixture_name)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


@pytest.mark.slow
def test_app_boots_with_no_input_selected():
    """Before any fixture/upload is picked, the app must render both tabs
    and the Pipeline Inspector's placeholder info message without raising."""
    at = AppTest.from_file(str(APP_PATH), default_timeout=120)
    at.run()

    assert not at.exception, [str(e) for e in at.exception]
    assert len(at.tabs) == 2
    assert len(at.info) >= 1


@pytest.mark.slow
def test_selecting_a_fixture_renders_key_pipeline_elements():
    at = _run_with_fixture()

    # Every collapsible stage section renders at least one plotly chart
    # (Input, FrameTrack, Notes, Timbre, Reducer panels).
    assert len(at.get("plotly_chart")) >= 5
    # Timbre panel's st.metric row (odd/even ratio, inharmonicity).
    assert len(at.metric) >= 2
    # Notes panel's selectbox fallback for note picking.
    assert any(sb.key == "note_select_fallback" for sb in at.selectbox)
    # Reducer panel's stage radio.
    assert any(r.key == "reducer_stage_radio" for r in at.radio)


@pytest.mark.slow
def test_voicing_threshold_tweak_reruns_without_exception():
    at = _run_with_fixture()

    slider = at.sidebar.slider(key="cfgw_mono_voicing_threshold")
    slider.set_value(min(1.0, slider.value + 0.1))
    at.run()

    assert not at.exception, [str(e) for e in at.exception]
    assert len(at.get("plotly_chart")) >= 5


@pytest.mark.slow
def test_reducer_stage_radio_tweak_reruns_without_exception():
    at = _run_with_fixture()

    radio = at.radio(key="reducer_stage_radio")
    radio.set_value(2)
    at.run()

    assert not at.exception, [str(e) for e in at.exception]


@pytest.mark.slow
def test_url_fetch_path_feeds_pipeline(monkeypatch):
    """The URL input path (D-017) with the network layer stubbed out:
    `pipeline_cache.fetch_url_bytes` is monkeypatched to return fixture
    bytes -- feasible because AppTest runs app.py in-process against the
    already-imported module object. Covers text_input -> Fetch button ->
    register_bytes -> full pipeline."""
    from melody_extractor.gui import pipeline_cache

    fixture = Path(__file__).resolve().parent / "fixtures" / "mono_scale.wav"
    monkeypatch.setattr(
        pipeline_cache, "fetch_url_bytes",
        lambda url, max_duration_s: (
            fixture.read_bytes(), "mono_scale.wav", "Fake Title",
            "https://www.youtube.com/watch?v=fake",
        ),
    )

    at = AppTest.from_file(str(APP_PATH), default_timeout=120)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]

    at.sidebar.text_input(key="input_url").set_value("https://www.youtube.com/watch?v=fake")
    at.run()
    at.sidebar.button(key="input_url_fetch").click()
    at.run()

    assert not at.exception, [str(e) for e in at.exception]
    assert len(at.get("plotly_chart")) >= 5  # pipeline ran off the URL bytes


@pytest.mark.slow
def test_eval_dashboard_tab_renders_without_exception():
    at = _run_with_fixture()

    assert not at.exception, [str(e) for e in at.exception]
    # Eval Dashboard tab is the second of the two top-level tabs.
    assert at.tabs[1].label == "Eval Dashboard"
