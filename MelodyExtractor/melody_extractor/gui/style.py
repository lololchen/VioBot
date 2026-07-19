"""Mobile-friendly CSS -- the ONLY place custom CSS is allowed in this repo
(D-017). Everything else must use plain Streamlit layout primitives.

Selectors use `data-testid` attributes, verified against streamlit>=1.37
(the [gui] extra's floor); they can drift across Streamlit majors, so
re-verify here on any major version bump. Streamlit already stacks
`st.columns` vertically on narrow viewports and every chart passes
`use_container_width=True`, so this block only fixes what Streamlit does
not: desktop-sized paddings/title, and plotly's touch-hostile modebar.
"""
from __future__ import annotations

import streamlit as st

_CSS = """
<style>
@media (max-width: 640px) {
    /* wide-layout default padding wastes most of a phone's width */
    [data-testid="stMainBlockContainer"] {
        padding: 1rem 0.75rem;
    }
    /* app title wraps to ~3 lines at the desktop font size */
    h1 {
        font-size: 1.4rem;
    }
    /* plotly's modebar is useless on touch and overlaps narrow charts */
    .modebar {
        display: none !important;
    }
}
</style>
"""


def inject() -> None:
    """Call once per script run, right after st.set_page_config."""
    st.markdown(_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Step anchors + smooth auto-scroll (D-019). Kept here with the CSS because
# this is the repo's single sanctioned home for custom front-end injection;
# views must call these helpers instead of writing their own HTML/JS.
# ---------------------------------------------------------------------------

def step_anchor(anchor_id: str) -> None:
    """Drop an invisible, scroll-addressable anchor at the current layout
    position (used just above each pipeline-step box)."""
    st.markdown(f'<span id="{anchor_id}"></span>', unsafe_allow_html=True)


def scroll_to_anchor(anchor_id: str, bottom_margin_px: int = 150) -> None:
    """Smooth-scroll the page so `anchor_id` sits `bottom_margin_px` above the
    viewport bottom -- i.e. the step *above* the anchor fills the screen while
    the next step's header (collapsed, spinner still running) stays visible.

    Runs from a zero-height components iframe; the anchor may not exist yet
    when this executes (Streamlit streams elements as the script progresses),
    so it retries briefly instead of assuming the DOM is complete.
    """
    import streamlit.components.v1 as components

    components.html(
        f"""<script>
        (function() {{
            let tries = 0;
            function go() {{
                const win = window.parent;
                const el = win.document.getElementById({anchor_id!r});
                if (!el) {{ if (++tries < 20) setTimeout(go, 200); return; }}
                const y = el.getBoundingClientRect().top + win.pageYOffset
                          - win.innerHeight + {int(bottom_margin_px)};
                win.scrollTo({{top: Math.max(y, 0), behavior: "smooth"}});
            }}
            setTimeout(go, 300);
        }})();
        </script>""",
        height=0,
    )
