"""Plotly figures — house style mirrors melody_extractor/gui/figures.py:
default template, axis titles, `use_container_width=True` at every call site,
hard caps on payload sizes (D-018 lesson: charts dominate rerun cost).
String palette: G/D/A/E in the repo's categorical order."""
from __future__ import annotations

import math

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from motion_planner.hardware import HardwareProfile
from motion_planner.schema import MotionScore

STRING_PALETTE = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728")   # G D A E
STRING_NAMES = ("G", "D", "A", "E")

_MAX_TRACK_POINTS = 4000
_MAX_ANIM_FRAMES = 120


def fingerboard_timeline(score: MotionScore, profile: HardwareProfile) -> go.Figure:
    """Time × fingerboard-mm: one bar per note at its stopped position,
    string-colored, dashed for open strings; finger id in the hover."""
    fig = go.Figure()
    seen = set()
    for p in score.note_plan:
        color = STRING_PALETTE[p.string]
        name = f"{STRING_NAMES[p.string]} string"
        t0 = p.realized_onset_s
        t1 = t0 + p.realized_duration_s
        label = "open" if p.finger is None else f"finger {p.finger}"
        fig.add_trace(go.Scatter(
            x=[t0, t1], y=[p.position_mm, p.position_mm], mode="lines",
            line=dict(width=8, color=color, dash="dash" if p.finger is None else "solid"),
            name=name, legendgroup=name, showlegend=name not in seen,
            hovertemplate=(f"note {p.note_index} · {label}<br>pos %{{y:.1f}} mm "
                           f"({p.position_st:.2f} st)<br>t=%{{x:.3f}} s<extra></extra>")))
        seen.add(name)
        if p.violations:
            fig.add_trace(go.Scatter(
                x=[t0], y=[p.position_mm], mode="markers",
                marker=dict(symbol="x", size=10, color="red"),
                name="violation", legendgroup="violation",
                showlegend="violation" not in seen,
                hovertemplate=(f"note {p.note_index}: "
                               f"{', '.join(v.kind for v in p.violations)}<extra></extra>")))
            seen.add("violation")
    fig.update_layout(xaxis_title="Time (s)", yaxis_title="Position from nut (mm)",
                      title="Fingerboard timeline")
    return fig


def _downsample(values, stride: int):
    return values[::stride]


def bow_tracks_figure(score: MotionScore) -> go.Figure:
    tracks = score.tracks
    rows = [("bow.speed_mps", "v_b (m/s)"), ("bow.force_n", "F (N)"),
            ("bow.inclination_rad", "incl (rad)"), ("bow.y_m", "y (m)")]
    fig = make_subplots(rows=len(rows), cols=1, shared_xaxes=True,
                        subplot_titles=[label for _, label in rows])
    if tracks is not None and tracks.n_samples():
        stride = max(1, math.ceil(tracks.n_samples() / _MAX_TRACK_POINTS))
        times = _downsample(tracks.times_s(), stride)
        for r, (channel, label) in enumerate(rows, start=1):
            values = _downsample(tracks.channels.get(channel, ()), stride)
            fig.add_trace(go.Scatter(x=times, y=values, mode="lines",
                                     name=label, showlegend=False), row=r, col=1)
    fig.update_layout(height=560, title="Bow control tracks")
    fig.update_xaxes(title_text="Time (s)", row=len(rows), col=1)
    return fig


def mechanism_animation(score: MotionScore, profile: HardwareProfile) -> go.Figure:
    """Top-view schematic (x = mm along fingerboard, y = string lane) animated
    over time: finger dots (size = press force) + a bow marker on its band."""
    tracks = score.tracks
    fig = go.Figure()
    if tracks is None or not tracks.n_samples():
        return fig
    n = tracks.n_samples()
    stride = max(1, math.ceil(n / _MAX_ANIM_FRAMES))
    ch = tracks.channels
    spacing_m = profile.strings.spacing_bridge_mm / 1000.0
    n_fingers = len(profile.fingers)
    angles = profile.strings.band_angles_rad

    def frame_data(k: int):
        xs, ys, sizes, texts = [], [], [], []
        for fi in range(n_fingers):
            xs.append(ch[f"f{fi}.z_m"][k] * 1000.0)
            ys.append(ch[f"f{fi}.x_m"][k] / spacing_m if spacing_m else 0.0)
            pressed = ch[f"f{fi}.press_n"][k]
            sizes.append(10 + 6 * min(pressed, 4.0))
            texts.append(f"f{fi} ({'pressed' if pressed > 0.5 else 'up'})")
        incl = ch["bow.inclination_rad"][k]
        band = min(range(len(angles)), key=lambda b: abs(angles[b] - incl))
        bow_y = band / 2.0
        in_contact = ch["bow.y_m"][k] < 5e-4
        return (go.Scatter(x=xs, y=ys, mode="markers+text", text=texts,
                           textposition="top center",
                           marker=dict(size=sizes, color="#9467bd"), name="fingers"),
                go.Scatter(x=[-25.0], y=[bow_y], mode="markers",
                           marker=dict(symbol="diamond", size=16,
                                       color="#8c564b" if in_contact else "lightgrey"),
                           name="bow (band)"))

    first = frame_data(0)
    for trace in first:
        fig.add_trace(trace)
    frames, steps = [], []
    for k in range(0, n, stride):
        t = tracks.start_s + k * tracks.hop_s
        frames.append(go.Frame(data=list(frame_data(k)), name=f"{t:.2f}"))
        steps.append(dict(method="animate", label=f"{t:.1f}",
                          args=[[f"{t:.2f}"], {"mode": "immediate",
                                               "frame": {"duration": 0, "redraw": True}}]))
    fig.frames = frames
    fig.update_layout(
        title="Mechanism motion (top view)",
        xaxis=dict(title="Along fingerboard (mm from nut)", range=[-40, 340]),
        yaxis=dict(title="String lane", tickvals=[0, 1, 2, 3],
                   ticktext=list(STRING_NAMES), range=[-0.7, 3.7]),
        updatemenus=[dict(type="buttons", showactive=False, y=1.15, x=0,
                          buttons=[dict(label="Play", method="animate",
                                        args=[None, {"frame": {"duration": 60, "redraw": True},
                                                     "fromcurrent": True}])])],
        sliders=[dict(steps=steps, currentvalue={"prefix": "t = ", "suffix": " s"})],
        height=430)
    for s, name in enumerate(STRING_NAMES):
        fig.add_hline(y=s, line_width=1, line_color=STRING_PALETTE[s], opacity=0.35)
    return fig


def compare_figure(rows: "list[dict]") -> go.Figure:
    """Grouped bars: tempo headroom per piece, one color per profile."""
    fig = go.Figure()
    profiles = sorted({r["profile"] for r in rows})
    palette = ("#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b")
    for i, prof in enumerate(profiles):
        sub = [r for r in rows if r["profile"] == prof]
        fig.add_trace(go.Bar(
            x=[r["piece"] for r in sub], y=[r["tempo_headroom"] for r in sub],
            name=prof, marker_color=palette[i % len(palette)],
            customdata=[[r["feasibility_pct"], r["motor_count"]] for r in sub],
            hovertemplate=("%{x}<br>headroom %{y:.2f}×<br>feasibility "
                           "%{customdata[0]:.1f}%<br>%{customdata[1]} motors<extra></extra>")))
    fig.add_hline(y=1.0, line_dash="dash", line_color="black",
                  annotation_text="as written")
    fig.update_layout(barmode="group", xaxis_title="Piece",
                      yaxis_title="Tempo headroom (×)", title="Topology comparison")
    return fig
