"""Mobile-friendly CSS — the ONLY place custom CSS is allowed in this package
(mirrors melody_extractor/gui/style.py, D-017 rule)."""
from __future__ import annotations

import streamlit as st

_CSS = """
<style>
/* Help (`?`) hover explanations: a gray a step brighter than the default dark
   sidebar (#262730) so the tooltip box stays legible over either the sidebar or
   the main panel. Targets the tooltip content in current Streamlit and the
   underlying BaseWeb node as a fallback across minor versions. */
[data-testid="stTooltipContent"],
[data-testid="stTooltipContent"] div,
div[data-baseweb="tooltip"] {
    background-color: #3a3b47 !important;
    color: #fafafa !important;
}
@media (max-width: 640px) {
    [data-testid="stMainBlockContainer"] {
        padding: 1rem 0.75rem;
    }
    h1 {
        font-size: 1.4rem;
    }
    .modebar {
        display: none !important;
    }
}
</style>
"""


def inject() -> None:
    """Call once per script run, right after st.set_page_config."""
    st.markdown(_CSS, unsafe_allow_html=True)
