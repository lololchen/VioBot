"""Port/app registry helpers + the shared self-launch logic (D-030).

registry.json is the single source of truth for module → port. Every app.py's
bare-python path ("Run code" in VS Code) funnels through open_or_launch():
if the registered port already serves (hub running), just open a browser tab
in the last-used window; otherwise start this one app on its registered port,
non-headless, so Streamlit opens the tab itself.
"""
from __future__ import annotations

import json
import socket
import sys
import webbrowser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = Path(__file__).resolve().parent / "registry.json"


def load_registry() -> dict:
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))["apps"]


def app_entry(key: str) -> dict:
    apps = load_registry()
    if key not in apps:
        raise KeyError(f"unknown gui_hub app {key!r} (have {sorted(apps)})")
    return apps[key]


def app_abspath(key: str) -> Path:
    return REPO_ROOT / app_entry(key)["app"]


def url_for(key: str) -> str:
    return f"http://localhost:{app_entry(key)['port']}"


def is_serving(port: int, timeout_s: float = 0.3) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=timeout_s):
            return True
    except OSError:
        return False


def open_or_launch(key: str) -> None:
    """Bare-python entry for one app: reuse the running server or start it."""
    entry = app_entry(key)
    if is_serving(entry["port"]):
        # Hub (or an earlier run) already serves this app: a plain open lands
        # as a tab in the last-used browser window — exactly the requirement.
        webbrowser.open(url_for(key))
        return
    from streamlit.web import cli as stcli

    sys.argv = [
        "streamlit", "run", str(app_abspath(key)),
        "--server.port", str(entry["port"]),
        # Hot-reload off, same rationale as D-021 addendum (partial reloads
        # crash with mixed-version classes; OneDrive watcher unreliable).
        "--server.fileWatcherType", "none",
    ]
    stcli.main()
