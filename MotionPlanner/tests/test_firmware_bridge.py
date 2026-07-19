import struct
from types import SimpleNamespace

import pytest

from motion_planner.config_io import PlannerConfig
from motion_planner.firmware_bridge import protocol
from motion_planner.firmware_bridge.bench import run_bench
from motion_planner.firmware_bridge.streamer import (
    stream_dry_run,
    stream_schedule,
    stream_to_transport,
)
from motion_planner.firmware_bridge.transport import MockTransport
from motion_planner.planner import plan


def test_crc16_modbus_reference_vector():
    assert protocol.crc16_modbus(b"123456789") == 0x4B37


def test_frame_encode_decode_roundtrip():
    frame = protocol.Frame(device=1, cmd=protocol.CMD_SET_TARGET, seq=42,
                           payload=protocol.pack_f32(1.5, -0.25))
    raw = protocol.encode(frame)
    assert raw[:2] == b"\xaa\x55" and raw[2] == 0x02
    decoded, rest = protocol.decode(raw + b"tail")
    assert decoded == frame and rest == b"tail"
    assert protocol.unpack_f32(decoded.payload) == (1.5, -0.25)


def test_decode_rejects_corruption():
    raw = bytearray(protocol.encode(protocol.Frame(device=1, cmd=protocol.CMD_PING, seq=0)))
    raw[5] ^= 0xFF  # flip the cmd byte -> CRC must fail
    with pytest.raises(protocol.FrameError, match="CRC"):
        protocol.decode(bytes(raw))
    with pytest.raises(protocol.FrameError, match="sync"):
        protocol.decode(b"\x00\x00" + bytes(raw)[2:])


def test_encode_decode_fuzz_roundtrip():
    # Deterministic pseudo-fuzz over payload lengths and byte patterns.
    for n in (0, 1, 3, 4, 17, 255, 1024, 2048):
        payload = bytes((7 * i + n) % 256 for i in range(n))
        f = protocol.Frame(device=n % 8, cmd=(n % 32) + 1, seq=n % 256, payload=payload)
        decoded, rest = protocol.decode(protocol.encode(f))
        assert decoded == f and rest == b""
    with pytest.raises(protocol.FrameError):
        protocol.encode(protocol.Frame(device=1, cmd=1, seq=0, payload=b"x" * 2049))


def test_streamer_schedule_and_heartbeats(fixture_sequences, profiles):
    score, _ = plan(fixture_sequences["mono_scale"], profiles["concept_a_1finger"],
                    PlannerConfig())
    schedule = stream_schedule(score, "f0.z_m")
    cmds = [s.frame.cmd for s in schedule]
    assert cmds[0] == protocol.CMD_SET_MODE and cmds[1] == protocol.CMD_SET_ENABLE
    assert cmds[-1] == protocol.CMD_SET_ENABLE  # disable at the end
    duration = schedule[-1].t_s
    heartbeats = [s for s in schedule if s.frame.cmd == protocol.CMD_HEARTBEAT]
    assert len(heartbeats) >= int(duration / protocol.HEARTBEAT_INTERVAL_S)
    gaps = [b.t_s - a.t_s for a, b in zip(heartbeats, heartbeats[1:])]
    assert all(abs(g - protocol.HEARTBEAT_INTERVAL_S) < 1e-9 for g in gaps)
    targets = [s for s in schedule if s.frame.cmd == protocol.CMD_SET_TARGET]
    assert len(targets) == int(duration / 0.01) + 1  # 100 Hz


def test_streamer_golden_bytes_deterministic(fixture_sequences, profiles):
    score, _ = plan(fixture_sequences["mono_scale"], profiles["concept_a_1finger"],
                    PlannerConfig())
    log1 = stream_dry_run(score, "f0.z_m")
    log2 = stream_dry_run(score, "f0.z_m")
    assert log1 == log2 and len(log1) > 1000
    # The byte log decodes back into the schedule's frames.
    transport = MockTransport()
    n = stream_to_transport(score, "f0.z_m", transport)
    assert bytes(transport.byte_log) == log1
    assert len(transport.decoded_log()) == n


def test_mock_transport_acks_and_scripted_telemetry():
    telem = protocol.Frame(device=1, cmd=protocol.CMD_TELEMETRY, seq=0,
                           payload=protocol.pack_f32(100.0, 0.5))
    transport = MockTransport(telemetry_script=[telem])
    received = []
    transport.on_telemetry(received.append)
    reply = transport.request(protocol.Frame(device=1, cmd=protocol.CMD_PING, seq=7))
    assert reply.cmd == protocol.CMD_ACK and reply.seq == 7
    assert received == [telem]


def test_bench_dry_run(tmp_path):
    args = SimpleNamespace(port=None, device=1, mode="position", sine=(0.5, 2.0),
                           step=None, duration=0.5, dry_run_log=str(tmp_path / "bench.bin"))
    assert run_bench(args) == 0
    raw = (tmp_path / "bench.bin").read_bytes()
    frames = []
    rest = raw
    while rest:
        f, rest = protocol.decode(rest)
        frames.append(f)
    assert frames[0].cmd == protocol.CMD_PING
    assert frames[-1].cmd == protocol.CMD_SET_ENABLE and frames[-1].payload == b"\x00"
    targets = [f for f in frames if f.cmd == protocol.CMD_SET_TARGET]
    assert len(targets) == int(0.5 * 100) + 1
    values = [protocol.unpack_f32(f.payload)[0] for f in targets]
    assert max(values) == pytest.approx(0.5, abs=0.01)  # sine amplitude reached
