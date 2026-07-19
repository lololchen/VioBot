"""Entry point: sidebar (input source + params + preset UI), two tabs.

Runnable two ways (plan_GUI_MelodyExtractor.md deliverable 5):
- `streamlit run .../gui/app.py` -- Streamlit executes this file as a bare
  script (no package context), so the sys.path bootstrap below is needed
  before the absolute `melody_extractor.gui.*` imports will resolve.
- Imported/executed by `streamlit.testing.v1.AppTest.from_file(...)`, which
  runs this same file through Streamlit's own ScriptRunner -- the identical
  code path, so no special-casing is needed for tests.
"""
from __future__ import annotations

import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent            # .../melody_extractor/gui
_REPO_ROOT = _THIS_DIR.parents[1]                       # .../MelodyExtractor (contains melody_extractor/)
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st  # noqa: E402

from melody_extractor import url_fetch  # noqa: E402
from melody_extractor.gui import eval_view, inspector_view, params, pipeline_cache, style  # noqa: E402
from melody_extractor.input_adapter import AUDIO_EXTENSIONS, MIDI_EXTENSIONS  # noqa: E402

FIXTURES_DIR = _REPO_ROOT / "tests" / "fixtures"

# Default demo link prefilled in the sidebar URL box (D-019). To change the
# default song, edit THIS constant -- documented in docs/decisions.md D-019.
DEFAULT_URL = "https://www.youtube.com/watch?v=wh-pBxeHE3U"


def main() -> None:
    st.set_page_config(page_title="MelodyExtractor — Pipeline Inspector", layout="wide")
    style.inject()
    st.title("MelodyExtractor — Pipeline Inspector")

    st.sidebar.header("Input")
    fixture_names = sorted(p.name for p in FIXTURES_DIR.glob("*.wav")) if FIXTURES_DIR.is_dir() else []
    fixture_choice = st.sidebar.selectbox(
        "Fixture (tests/fixtures/*.wav)", ["(none)"] + fixture_names, key="input_fixture_select",
    )

    upload_types = sorted(ext.lstrip(".") for ext in (AUDIO_EXTENSIONS | MIDI_EXTENSIONS))
    uploaded = st.sidebar.file_uploader(
        "Or upload an audio/MIDI file", type=upload_types, key="input_uploader",
    )

    # URL source (D-017): YouTube/SoundCloud direct via yt-dlp; Spotify links
    # resolve to a title (oEmbed) and fetch the YouTube match. The fetched
    # digest is stashed in session_state so parameter tweaks never re-download.
    # Prefilled with DEFAULT_URL so the Fetch button is enabled immediately --
    # no Enter keypress needed; clicking the button commits any edited text.
    url_value = st.sidebar.text_input(
        "Or paste a URL (YouTube / SoundCloud / Spotify)", value=DEFAULT_URL, key="input_url",
    )
    fetch_clicked = st.sidebar.button(
        "Fetch audio", key="input_url_fetch",
        disabled=not url_value.strip(), use_container_width=True,
    )
    if fetch_clicked:
        with st.sidebar.status("Fetching audio...") as status:
            try:
                data, filename, title, resolved_url = pipeline_cache.fetch_url_bytes(
                    url_value.strip(), url_fetch.DEFAULT_MAX_DURATION_S,
                )
                url_digest = pipeline_cache.register_bytes(data, filename)
                st.session_state["input_url_result"] = (url_digest, title, resolved_url)
                status.update(label=f"Loaded: {title}", state="complete")
            except url_fetch.UrlFetchError as e:
                status.update(label="Fetch failed", state="error")
                st.sidebar.error(str(e))
            except ImportError as e:  # yt-dlp missing: [url] extra not installed
                status.update(label="Fetch failed", state="error")
                st.sidebar.error(str(e))
    url_result = st.session_state.get("input_url_result")
    if url_result is not None:
        _, url_title, url_resolved = url_result
        st.sidebar.caption(f"URL audio: {url_title}")
        if url_resolved != url_value.strip():
            st.sidebar.caption(f"Fetched from: {url_resolved}")
        if st.sidebar.button("Clear URL audio", key="input_url_clear", use_container_width=True):
            st.session_state.pop("input_url_result", None)
            url_result = None

    # Input precedence: upload > URL result > fixture. An explicit local file
    # beats everything; a fetched URL is an explicit action and beats the
    # fixture selectbox, which may just hold a stale value.
    digest = None
    if uploaded is not None:
        digest = pipeline_cache.register_bytes(uploaded.getvalue(), uploaded.name)
    elif url_result is not None:
        digest = url_result[0]
    elif fixture_choice != "(none)":
        digest = pipeline_cache.register_bytes((FIXTURES_DIR / fixture_choice).read_bytes(), fixture_choice)

    # Display switch (D-018): charts are the GUI's single biggest per-rerun
    # cost on long songs (payload build + websocket send + browser render);
    # none of them are needed for the transcription itself.
    st.sidebar.header("Display")
    show_charts = st.sidebar.toggle(
        "Show charts", value=True, key="display_show_charts",
        help="Turn off to skip all charts (waveform, spectrogram, frame track, "
             "piano rolls) for much faster reruns on long songs. Transcription, "
             "reduction, and the A/B audio players still run.",
    )

    st.sidebar.header("Parameters")
    mono_cfg = params.build_mono_config()
    timbre_cfg = params.build_timbre_config()
    stage_cfg = params.build_stage_config()
    render_cfg = params.build_render_config()

    params.build_preset_controls(mono_cfg, timbre_cfg, stage_cfg, render_cfg)

    tab_inspector, tab_eval = st.tabs(["Pipeline Inspector", "Eval Dashboard"])
    with tab_inspector:
        inspector_view.render(digest, mono_cfg, timbre_cfg, stage_cfg, render_cfg,
                              show_charts=show_charts)
    with tab_eval:
        eval_view.render(mono_cfg, timbre_cfg, stage_cfg)


if st.runtime.exists():
    # Normal paths: `streamlit run` and AppTest both execute this file inside
    # a Streamlit runtime (verified: st.runtime.exists() is True for both).
    main()
else:
    # Bare `python app.py` (e.g. VS Code's Run button): there is no Streamlit
    # runtime, so main() would render into the void. Hand off to the same
    # launcher the `melody-extractor-gui` console script uses, which re-execs
    # this file under `streamlit run`.
    from melody_extractor.gui import launch

    launch.main()
