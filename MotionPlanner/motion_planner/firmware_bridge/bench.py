"""bench — single-motor shakedown for the TC375 MVP (device 0x01) and, later,
the SysID sweep executor.

    motion-planner bench --sine 0.5 1.0 --duration 2            (mock, dry run)
    motion-planner bench --port COM5 --mode position --step 1.0 (real UART)

Without --port everything goes through MockTransport and the byte stream can
be written via --dry-run-log for inspection/golden tests.
"""
from __future__ import annotations

import math
from pathlib import Path

from .protocol import (
    CMD_HEARTBEAT,
    CMD_PING,
    CMD_SET_ENABLE,
    CMD_SET_MODE,
    CMD_SET_TARGET,
    HEARTBEAT_INTERVAL_S,
    MODE_POSITION,
    MODE_SPEED,
    MODE_TORQUE,
    Frame,
    set_mode_payload,
    set_target_payload,
)
from .transport import MockTransport

_MODES = {"position": MODE_POSITION, "speed": MODE_SPEED, "torque": MODE_TORQUE}
_RATE_HZ = 100.0


def _targets(args) -> "list[float]":
    n = int(args.duration * _RATE_HZ) + 1
    if args.sine is not None:
        ampl, freq = args.sine
        return [ampl * math.sin(2.0 * math.pi * freq * k / _RATE_HZ) for k in range(n)]
    step = args.step if args.step is not None else 0.0
    return [step] * n


def run_bench(args) -> int:
    if args.port:  # pragma: no cover - needs hardware
        from .transport import SerialTransport
        transport = SerialTransport(args.port)
        realtime = True
    else:
        transport = MockTransport()
        realtime = False

    device = args.device
    seq = 0

    def send(cmd: int, payload: bytes = b"") -> None:
        nonlocal seq
        transport.send(Frame(device=device, cmd=cmd, seq=seq & 0xFF, payload=payload))
        seq += 1

    send(CMD_PING)
    send(CMD_SET_MODE, set_mode_payload(_MODES[args.mode]))
    send(CMD_SET_ENABLE, b"\x01")
    next_heartbeat = 0.0
    if realtime:  # pragma: no cover - wall-clock pacing
        import time
        t0 = time.monotonic()
    for k, target in enumerate(_targets(args)):
        t = k / _RATE_HZ
        while next_heartbeat <= t:
            send(CMD_HEARTBEAT)
            next_heartbeat += HEARTBEAT_INTERVAL_S
        if realtime:  # pragma: no cover
            delay = t - (time.monotonic() - t0)
            if delay > 0:
                time.sleep(delay)
        send(CMD_SET_TARGET, set_target_payload(target))
    send(CMD_SET_ENABLE, b"\x00")
    transport.close()

    if isinstance(transport, MockTransport):
        n = len(transport.frames_sent)
        print(f"mock bench: {n} frames ({args.mode}, {args.duration}s)")
        if args.dry_run_log:
            path = Path(args.dry_run_log)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(bytes(transport.byte_log))
            print(f"wrote {path} ({len(transport.byte_log)} bytes)")
    return 0
