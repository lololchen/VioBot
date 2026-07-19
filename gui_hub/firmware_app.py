"""Firmware tab — placeholder page until the multi-axis firmware exists (D-030).

Already useful today: shows the latest MotionScore from the workspace and can
generate the Native-Protocol-v2 dry-run byte log for one track channel via
motion_planner.firmware_bridge (no hardware needed).
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
for extra in (_REPO_ROOT, _REPO_ROOT / "MotionPlanner"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import streamlit as st  # noqa: E402

from gui_hub import workspace  # noqa: E402


def main() -> None:
    st.set_page_config(page_title="Firmware — coming soon", layout="wide")
    st.title("Firmware — coming soon")
    st.caption("Motor control lives in the external repo "
               "[AachenBQ/Motor_Architecture](https://github.com/AachenBQ/Motor_Architecture) "
               "(TC375 + SimpleFOC, Native Protocol v2 over UART).")

    st.markdown(
        "**Planned here once hardware lands:** live telemetry plots (position/current @100 Hz), "
        "single-motor bench control (`motion-planner bench`), motor-command timeline view, and "
        "the SysID sweep executor (docs/SysID_Protocol.md).\n\n"
        "**Gaps being negotiated with the firmware team** (docs/PRD_MotionPlanner.md): "
        "multi-device addressing (M2–M8), trajectory streaming, synced start, force estimation "
        "without current sense, SI-scaling ownership, SysID telemetry.")

    st.divider()
    st.subheader("Dry-run: MotionScore → Native Protocol v2 bytes")
    st.caption(f"workspace motion_score: {workspace.describe('motion_score')}")
    score_path = workspace.latest("motion_score")
    if score_path is None:
        st.info("Plan something in the Sound2Motion tab first (it registers its "
                "MotionScore into the workspace), then come back here.")
        return
    try:
        from motion_planner.firmware_bridge.streamer import stream_dry_run
        from motion_planner.schema import MotionScore

        score = MotionScore.from_json(score_path)
        channels = sorted(score.tracks.channels) if score.tracks else []
        channel = st.selectbox("Track channel → device 0x01 (firmware MVP is single-motor)",
                               channels, index=0 if channels else None)
        if channel and st.button("Generate byte log"):
            raw = stream_dry_run(score, channel)
            st.success(f"{len(raw)} bytes ({len(raw) // 16} frames-ish) at 100 Hz + heartbeats")
            st.download_button("Download byte log", data=raw,
                               file_name=f"dryrun_{channel.replace('.', '_')}.bin")
    except ImportError as e:
        st.error(f"motion-planner not installed in this environment: {e}")


if st.runtime.exists():
    main()
else:
    from gui_hub.registry import open_or_launch

    open_or_launch("firmware")
