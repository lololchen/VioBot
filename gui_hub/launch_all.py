"""launch_all — start every registered GUI server and open ONE browser window
with all module tabs (D-030). Invoked by run_gui.bat; also runnable directly:

    python gui_hub/launch_all.py

Behavior:
- spawns each app from registry.json headless on its fixed port (skipping any
  port already serving — idempotent restarts),
- waits for readiness, then opens one NEW browser window containing all tabs
  (Chrome/Edge accept multiple URLs after --new-window; fallback: sequential
  webbrowser.open → tabs land in the last-used window),
- then babysits: a crashed server is restarted (the browser tab reconnects on
  refresh), mirroring run_gui.bat's old crash-restart loop. Ctrl+C stops all.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gui_hub.registry import REPO_ROOT, is_serving, load_registry  # noqa: E402

_READY_TIMEOUT_S = 90.0
_POLL_S = 0.5
_BROWSERS = (
    "chrome", "msedge",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
)


def _spawn(entry: dict) -> subprocess.Popen:
    cmd = [sys.executable, "-m", "streamlit", "run", str(REPO_ROOT / entry["app"]),
           "--server.port", str(entry["port"]),
           "--server.headless", "true",
           "--server.fileWatcherType", "none"]
    return subprocess.Popen(cmd, cwd=str(REPO_ROOT))


def _open_window(urls: "list[str]") -> None:
    for candidate in _BROWSERS:
        exe = shutil.which(candidate) if not candidate.endswith(".exe") else (
            candidate if Path(candidate).exists() else None)
        if exe:
            subprocess.Popen([exe, "--new-window", *urls])
            return
    # Unknown default browser: sequential opens (tabs in the last-used window).
    for url in urls:
        webbrowser.open(url)
        time.sleep(0.4)


def main() -> int:
    apps = load_registry()
    procs: "dict[str, subprocess.Popen | None]" = {}
    for key, entry in apps.items():
        if is_serving(entry["port"]):
            print(f"[hub] {key} already serving on :{entry['port']} - reusing")
            procs[key] = None
        else:
            print(f"[hub] starting {key} on :{entry['port']}")
            procs[key] = _spawn(entry)

    deadline = time.monotonic() + _READY_TIMEOUT_S
    pending = set(apps)
    while pending and time.monotonic() < deadline:
        for key in sorted(pending):
            if is_serving(apps[key]["port"]):
                pending.discard(key)
                break
        else:
            time.sleep(_POLL_S)
    if pending:
        print(f"[hub][warn] not ready in time: {sorted(pending)} - opening the rest anyway")

    urls = [f"http://localhost:{apps[k]['port']}" for k in apps]
    _open_window(urls)
    print("[hub] one browser window with all module tabs should be open.")
    print("[hub] close this console (or Ctrl+C) to stop all GUI servers.")

    try:
        while True:
            time.sleep(3.0)
            for key, proc in list(procs.items()):
                if proc is not None and proc.poll() is not None:
                    print(f"[hub][warn] {key} exited (code {proc.returncode}) - restarting; "
                          f"refresh its browser tab")
                    procs[key] = _spawn(apps[key])
    except KeyboardInterrupt:
        pass
    finally:
        for proc in procs.values():
            if proc is not None and proc.poll() is None:
                proc.terminate()
    return 0


if __name__ == "__main__":
    sys.exit(main())
