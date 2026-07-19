"""gui_hub — the unified GUI shell (D-030).

One browser window with all module tabs via run_gui.bat → launch_all.py;
fixed port registry (registry.json is the single source of truth); file-based
workspace manifest so the GUIs chain (extract → motion → motor command) or run
standalone. Each module's app.py stays runnable on its own (VS Code Run →
individual tab in the last-used window) via registry.open_or_launch.
"""
