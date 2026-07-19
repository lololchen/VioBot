"""AudioFeedback tab — placeholder page (concept only, D-003/D-030).

The module design lives in docs/CONCEPT_AudioFeedback.md; building v0 is
gated on first hardware recordings.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st  # noqa: E402

from gui_hub import workspace  # noqa: E402


def main() -> None:
    st.set_page_config(page_title="AudioFeedback — coming soon", layout="wide")
    st.title("AudioFeedback — coming soon")
    st.caption("Closed-loop listening during play. Concept: docs/CONCEPT_AudioFeedback.md")

    st.markdown(
        "**v0 (first hardware take):** record the robot → run MelodyExtractor on the recording "
        "→ offline DTW alignment → mir_eval diff vs the target NoteSequence → a CorrectionSet "
        "JSON (per-string/position intonation offsets, bow force/speed trims) that Sound2Motion "
        "applies as a calibration overlay on the next run.\n\n"
        "**v1:** online score follower with slow servo trims (requires a decisions.md entry "
        "revisiting D-003 batch-only). **v2:** sub-100 ms reflex loops, likely firmware-side.\n\n"
        "**SysID synergy:** every take carries its MotionScore hash, so (commanded controls × "
        "measured acoustics) pairs land in the SweepDataset for free — the learned bow-sound "
        "model trains while the robot practices.")

    st.divider()
    st.subheader("What this page will consume")
    for stage in ("note_sequence", "motion_score"):
        st.caption(f"workspace {stage}: {workspace.describe(stage)}")


if st.runtime.exists():
    main()
else:
    from gui_hub.registry import open_or_launch

    open_or_launch("audiofeedback")
