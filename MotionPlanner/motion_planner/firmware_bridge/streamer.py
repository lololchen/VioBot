"""Streamer — MotionScore tracks → timed CMD_SET_TARGET frames + heartbeats.

Firmware MVP reality (D-029): ONE motor at device 0x01, point-to-point targets,
no trajectory streaming, no multi-axis sync. The streamer therefore maps ONE
selected track channel onto device 0x01 at the command rate (default 100 Hz)
and interleaves the mandatory 250 ms heartbeat. Multi-axis streaming becomes a
loop over profile device_ids once the firmware grows M2..M8 addressing — the
gaps list lives in the PRD.

Unit scaling: SI task-space values are streamed as float32 as-is; the
task-space→joint conversion (lead-screw pitch, four-bar, differential mixing)
is firmware-gap #5 and deliberately NOT invented here. CONFIRM-WITH-FIRMWARE.

Dry-run: `stream_dry_run` produces the exact (t_s, frame) schedule and the
concatenated byte log without any transport or clock — golden-byte tests and
the GUI's "motor command" view consume it.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..schema import MotionScore
from .protocol import (
    CMD_HEARTBEAT,
    CMD_SET_ENABLE,
    CMD_SET_MODE,
    CMD_SET_TARGET,
    HEARTBEAT_INTERVAL_S,
    MODE_POSITION,
    Frame,
    encode,
    set_mode_payload,
    set_target_payload,
)

STREAMER_VERSION = "streamer-0.1.0"

DEFAULT_RATE_HZ = 100.0
MVP_DEVICE = 0x01


@dataclass(frozen=True)
class ScheduledFrame:
    t_s: float
    frame: Frame


def stream_schedule(score: MotionScore, channel: str, device: int = MVP_DEVICE,
                    rate_hz: float = DEFAULT_RATE_HZ, mode: int = MODE_POSITION,
                    ) -> "list[ScheduledFrame]":
    """Deterministic (t, frame) schedule for one track channel: enable + mode,
    then targets at rate_hz (sampled from the track grid, zero-order hold),
    heartbeats every 250 ms interleaved."""
    tracks = score.tracks
    if tracks is None or channel not in tracks.channels:
        raise ValueError(f"score has no track channel {channel!r}")
    values = tracks.channels[channel]
    duration = tracks.start_s + (len(values) - 1) * tracks.hop_s
    seq_counter = 0

    def next_seq() -> int:
        nonlocal seq_counter
        s = seq_counter & 0xFF
        seq_counter += 1
        return s

    out: "list[ScheduledFrame]" = [
        ScheduledFrame(0.0, Frame(device=device, cmd=CMD_SET_MODE, seq=next_seq(),
                                  payload=set_mode_payload(mode))),
        ScheduledFrame(0.0, Frame(device=device, cmd=CMD_SET_ENABLE, seq=next_seq(),
                                  payload=b"\x01")),
    ]
    dt = 1.0 / rate_hz
    n_cmds = int(duration / dt) + 1
    next_heartbeat = 0.0
    for k in range(n_cmds):
        t = k * dt
        while next_heartbeat <= t:
            out.append(ScheduledFrame(next_heartbeat,
                                      Frame(device=device, cmd=CMD_HEARTBEAT, seq=next_seq())))
            next_heartbeat += HEARTBEAT_INTERVAL_S
        idx = min(int(round((t - tracks.start_s) / tracks.hop_s)), len(values) - 1)
        out.append(ScheduledFrame(t, Frame(device=device, cmd=CMD_SET_TARGET, seq=next_seq(),
                                           payload=set_target_payload(values[max(idx, 0)]))))
    out.append(ScheduledFrame(duration, Frame(device=device, cmd=CMD_SET_ENABLE,
                                              seq=next_seq(), payload=b"\x00")))
    return out


def stream_dry_run(score: MotionScore, channel: str, **kwargs) -> bytes:
    """Concatenated byte log of the full schedule (golden-byte tests)."""
    return b"".join(encode(item.frame) for item in stream_schedule(score, channel, **kwargs))


def stream_to_transport(score: MotionScore, channel: str, transport,
                        realtime: bool = False, **kwargs) -> int:
    """Send the schedule through a transport. realtime=True paces with the wall
    clock (bench use); False sends back-to-back (mock/golden tests). Returns
    the number of frames sent."""
    schedule = stream_schedule(score, channel, **kwargs)
    if realtime:  # pragma: no cover - wall-clock pacing, bench only
        import time
        t0 = time.monotonic()
        for item in schedule:
            delay = item.t_s - (time.monotonic() - t0)
            if delay > 0:
                time.sleep(delay)
            transport.send(item.frame)
    else:
        for item in schedule:
            transport.send(item.frame)
    return len(schedule)
