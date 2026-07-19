import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]          # MotionPlanner/
REPO_ROOT = ROOT.parent


def test_core_modules_never_import_streamlit():
    """gui/ is the only place allowed to import streamlit/plotly (house rule)."""
    code = (
        "import importlib, sys\n"
        "for m in ('schema','hardware','profile_io','config_io','fingering','bowing',"
        "'vibrato','trajectory','planner','bow_sound_model','simulate','roundtrip',"
        "'compare','sysid','cli','firmware_bridge.protocol','firmware_bridge.streamer',"
        "'firmware_bridge.transport'):\n"
        "    importlib.import_module('motion_planner.' + m)\n"
        "assert 'streamlit' not in sys.modules, 'core imported streamlit!'\n"
        "assert 'plotly' not in sys.modules, 'core imported plotly!'\n"
        "print('clean')\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True,
                            text=True, cwd=str(ROOT))
    assert result.returncode == 0, result.stderr
    assert "clean" in result.stdout


def test_registry_ports_unique_and_apps_exist():
    registry = json.loads((REPO_ROOT / "gui_hub" / "registry.json").read_text())["apps"]
    ports = [entry["port"] for entry in registry.values()]
    assert len(set(ports)) == len(ports)
    assert set(registry) == {"melody_extractor", "sound2motion", "firmware", "audiofeedback"}
    for entry in registry.values():
        assert (REPO_ROOT / entry["app"]).exists(), entry["app"]


def test_workspace_register_and_latest(tmp_path, monkeypatch):
    sys.path.insert(0, str(REPO_ROOT))
    from gui_hub import workspace

    monkeypatch.setattr(workspace, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(workspace, "WORKSPACE_DIR", tmp_path / "workspace")
    monkeypatch.setattr(workspace, "MANIFEST_PATH", tmp_path / "workspace" / "manifest.json")

    assert workspace.latest("note_sequence") is None
    path = workspace.register_text("note_sequence", "x.json", '{"a": 1}\n', producer="test")
    assert workspace.latest("note_sequence") == path
    assert "x.json" in workspace.describe("note_sequence")
    with pytest.raises(ValueError, match="unknown workspace stage"):
        workspace.register_text("nonsense", "y.json", "{}", producer="test")


def test_figures_build_without_streamlit(fixture_sequences, profiles):
    plotly = pytest.importorskip("plotly")  # [gui] extra
    from motion_planner.config_io import PlannerConfig
    from motion_planner.gui import figures
    from motion_planner.planner import plan

    score, report = plan(fixture_sequences["triple_rolled"], profiles["concept_b_4finger"],
                         PlannerConfig())
    fig1 = figures.fingerboard_timeline(score, profiles["concept_b_4finger"])
    assert len(fig1.data) >= len(score.note_plan)
    fig2 = figures.bow_tracks_figure(score)
    assert len(fig2.data) == 4
    fig3 = figures.mechanism_animation(score, profiles["concept_b_4finger"])
    assert fig3.frames
    rows = [{"profile": "a", "piece": "p", "tempo_headroom": 1.2,
             "feasibility_pct": 100.0, "motor_count": 6}]
    fig4 = figures.compare_figure(rows)
    assert fig4.data
