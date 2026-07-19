"""Streamlit GUI for Sound2Motion ([gui] extra, D-030).

Mirrors melody_extractor/gui's architecture: app.py entry + view modules +
params (sidebar widgets with formula tooltips) + pipeline_cache (st.cache_data)
+ figures (plotly) + style (the ONLY custom-CSS home in this package).
gui/ is the only place allowed to import streamlit/plotly; the GUI writes only
profiles/presets/workspace exports, never baselines.
"""
