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

    from streamlit.web import cli as stcli

    sys.argv = ["streamlit", "run", app_path, *sys.argv[1:]]
    stcli.main()  # click Command: reads sys.argv, exits the process itself


if __name__ == "__main__":
    main()
