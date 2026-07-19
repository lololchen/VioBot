"""Console-script wrapper for the `melody-extractor-gui` entry point (pyproject.toml).

Launches Streamlit's own CLI *programmatically* against `app.py`'s absolute
path, built from `__file__` and passed as part of an argument LIST assigned
to `sys.argv` -- never a shell string, since this repo's on-disk path
contains spaces (plan_GUI_MelodyExtractor.md deliverable 6). This mirrors
exactly what the `streamlit` console script itself does (its own
`console_scripts` entry point is `streamlit.web.cli:main`); we just prepend
`run <app.py>` to argv first.
"""
from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    app_path = str(Path(__file__).resolve().parent / "app.py")

    # gui_hub integration (D-030): when the mono-repo's hub is present, defer
    # to it — fixed registry port, and "just open a tab" if the hub already
    # serves this app (VS Code Run / repeated launches must not double-serve).
    repo_root = Path(__file__).resolve().parents[3]
    if (repo_root / "gui_hub" / "registry.py").exists():
        sys.path.insert(0, str(repo_root))
        from gui_hub.registry import open_or_launch

        open_or_launch("melody_extractor")
        return

    from streamlit.web import cli as stcli

    # File watching is disabled on purpose (D-021 addendum): Streamlit's
    # hot-reload only invalidates modules under the main script's folder
    # (gui/), so an edit touching BOTH gui/ and core modules (e.g. params.py
    # + soundsim.py) reloads half the change and crashes with mixed-version
    # classes (AttributeError on new fields). On this OneDrive-synced tree
    # the watcher is additionally unreliable. Restarting the server (close +
    # run_gui.bat) is the one reliable way to pick up code changes.
    sys.argv = [
        "streamlit", "run", app_path,
        "--server.fileWatcherType", "none",
        *sys.argv[1:],  # user-supplied flags come last so they can override
    ]
    stcli.main()  # click Command: reads sys.argv, exits the process itself


if __name__ == "__main__":
    main()
