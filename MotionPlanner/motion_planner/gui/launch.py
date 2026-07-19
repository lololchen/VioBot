"""Console-script wrapper for `motion-planner-gui` (pyproject.toml).

Registry-aware (D-030): uses the gui_hub port so a bare launch coexists with
the hub; if the port already serves, just opens a browser tab instead of a
second server. Falls back to plain `streamlit run` when gui_hub is absent
(package installed standalone outside the mono-repo)."""
from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    hub = repo_root / "gui_hub"
    if (hub / "registry.py").exists():
        sys.path.insert(0, str(repo_root))
        from gui_hub.registry import open_or_launch

        open_or_launch("sound2motion")
        return

    app_path = str(Path(__file__).resolve().parent / "app.py")
    from streamlit.web import cli as stcli

    sys.argv = ["streamlit", "run", app_path,
                "--server.fileWatcherType", "none", *sys.argv[1:]]
    stcli.main()


if __name__ == "__main__":
    main()
