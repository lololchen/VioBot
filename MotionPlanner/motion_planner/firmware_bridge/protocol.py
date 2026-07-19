"""Native Protocol v2 framing (AachenBQ/Motor_Architecture, lower_computer/protocol/
NATIVE_PROTOCOL_V2.md as of 2026-07-17).

Frame:  AA 55 | VER(0x02) | FLAGS | DEVICE | CMD | SEQ | LEN:u16 LE | PAYLOAD(≤2048) | CRC16:u16 LE
CRC16/MODBUS (init 0xFFFF, reflected poly 0xA001) over VER..PAYLOAD, appended
little-endian. Multi-byte payload fields are little-endian; physical values are
IEEE-754 float32 in SI units (torque N·m, speed rad/s, position rad).

Every constant below mirrors the firmware header include/native_protocol.h at
the commit above. The firmware repo is days old — treat all of them as
provisional and re-verify before first hardware contact (D-029).
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

PROTOCOL_VERSION = 0x02          # CONFIRM-WITH-FIRMWARE
SYNC = b"\xaa\x55"               # CONFIRM-WITH-FIRMWARE
MAX_PAYLOAD = 2048               # CONFIRM-WITH-FIRMWARE
BROADCAST_DEVICE = 0xFF          # CONFIRM-WITH-FIRMWARE (emergency/disable only)

# Command ids — CONFIRM-WITH-FIRMWARE (native_protocol.h, 2026-07-17)
CMD_PING = 0x01
CMD_GET_DEVICE_INFO = 0x02
CMD_GET_CAPABILITIES = 0x03
CMD_HEARTBEAT = 0x04
CMD_SET_ENABLE = 0x10
CMD_SET_MODE = 0x11
CMD_SET_TARGET = 0x12
CMD_SET_PID = 0x13
CMD_CALIBRATE = 0x15
CMD_CLEAR_FAULT = 0x16
CMD_SET_LIMITS = 0x17
CMD_GET_LIMITS = 0x18
CMD_SAVE_CONFIG = 0x19
CMD_RESTORE_DEFAULTS = 0x1A
CMD_CONTROLLED_STOP = 0x1B
CMD_QUICK_STOP = 0x1C
CMD_EMERGENCY_STOP = 0x1F
CMD_GET_PID = 0x20
CMD_GET_DIAGNOSTICS = 0x22
CMD_SET_TELEMETRY_PROFILE = 0x23
CMD_GET_BACKEND_INFO = 0x24
CMD_GET_TELEMETRY_PROFILE = 0x25
CMD_TELEMETRY = 0x80             # MCU-initiated push
CMD_FAULT_EVENT = 0x81           # MCU-initiated push
CMD_ACK = 0xF0
CMD_ERROR = 0xF1

# Control modes — CONFIRM-WITH-FIRMWARE
MODE_TORQUE = 0
MODE_SPEED = 1
MODE_POSITION = 2

HEARTBEAT_INTERVAL_S = 0.25      # CONFIRM-WITH-FIRMWARE (250 ms host heartbeat)
DEFAULT_LEASE_S = 0.75           # CONFIRM-WITH-FIRMWARE (300–5000 ms window)


def crc16_modbus(data: bytes) -> int:
    """CRC16/MODBUS: init 0xFFFF, reflected poly 0xA001. b'123456789' -> 0x4B37."""
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    return crc


@dataclass(frozen=True)
class Frame:
    device: int
    cmd: int
    seq: int
    payload: bytes = b""
    flags: int = 0
    version: int = PROTOCOL_VERSION


class FrameError(ValueError):
    pass


def encode(frame: Frame) -> bytes:
    if not 0 <= len(frame.payload) <= MAX_PAYLOAD:
        raise FrameError(f"payload length {len(frame.payload)} out of range")
    body = struct.pack("<BBBBBH", frame.version, frame.flags, frame.device & 0xFF,
                       frame.cmd & 0xFF, frame.seq & 0xFF, len(frame.payload)) + frame.payload
    return SYNC + body + struct.pack("<H", crc16_modbus(body))


def decode(data: bytes) -> "tuple[Frame, bytes]":
    """Decode one frame from the head of `data`; returns (frame, remainder).
    Raises FrameError on malformed input (bad sync/CRC/length)."""
    if len(data) < 11:
        raise FrameError("frame too short")
    if data[:2] != SYNC:
        raise FrameError("bad sync bytes")
    version, flags, device, cmd, seq = data[2], data[3], data[4], data[5], data[6]
    (length,) = struct.unpack("<H", data[7:9])
    end = 9 + length
    if length > MAX_PAYLOAD or len(data) < end + 2:
        raise FrameError("truncated frame")
    payload = data[9:end]
    (crc,) = struct.unpack("<H", data[end:end + 2])
    body = data[2:end]
    if crc != crc16_modbus(body):
        raise FrameError("CRC mismatch")
    return Frame(device=device, cmd=cmd, seq=seq, payload=payload,
                 flags=flags, version=version), data[end + 2:]


# -- payload helpers (float32 LE SI values) --

def pack_f32(*values: float) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


def unpack_f32(payload: bytes) -> "tuple[float, ...]":
    n = len(payload) // 4
    return struct.unpack(f"<{n}f", payload[:4 * n])


def set_mode_payload(mode: int) -> bytes:
    return struct.pack("<B", mode)   # CONFIRM-WITH-FIRMWARE (payload layout)


def set_target_payload(value: float) -> bytes:
    return pack_f32(value)           # CONFIRM-WITH-FIRMWARE (payload layout)
