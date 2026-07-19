"""Transports: MockTransport (deterministic, always available) and
SerialTransport ([serial] extra, real UART).

MockTransport ACKs every request, optionally replays scripted telemetry
frames, and records the full outgoing byte log — golden-byte regression tests
and dry-runs read that log. It is the reason firmware drift can never break
this repo's test suite (D-029).
"""
from __future__ import annotations

from typing import Callable, Optional, Protocol

from .protocol import CMD_ACK, Frame, decode, encode


class Transport(Protocol):
    def request(self, frame: Frame) -> Frame: ...
    def send(self, frame: Frame) -> None: ...
    def close(self) -> None: ...


class MockTransport:
    """Deterministic loopback. `telemetry_script`: frames handed to the
    callback in order, one per request, once a callback is registered."""

    def __init__(self, telemetry_script: "list[Frame] | None" = None):
        self.byte_log = bytearray()
        self.frames_sent: "list[Frame]" = []
        self._telemetry = list(telemetry_script or [])
        self._callback: "Optional[Callable[[Frame], None]]" = None

    def on_telemetry(self, callback: "Callable[[Frame], None]") -> None:
        self._callback = callback

    def send(self, frame: Frame) -> None:
        raw = encode(frame)
        self.byte_log += raw
        self.frames_sent.append(frame)

    def request(self, frame: Frame) -> Frame:
        self.send(frame)
        if self._callback is not None and self._telemetry:
            self._callback(self._telemetry.pop(0))
        return Frame(device=frame.device, cmd=CMD_ACK, seq=frame.seq)

    def close(self) -> None:
        pass

    def decoded_log(self) -> "list[Frame]":
        frames = []
        rest = bytes(self.byte_log)
        while rest:
            frame, rest = decode(rest)
            frames.append(frame)
        return frames


class SerialTransport:
    """Real UART via pyserial ([serial] extra). Request/response with a read
    timeout; telemetry frames arriving between responses go to the callback."""

    def __init__(self, port: str, baudrate: int = 115200, timeout_s: float = 0.5):
        try:
            import serial  # type: ignore
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise ImportError(
                "pyserial is required for SerialTransport - install the [serial] extra"
            ) from exc
        self._serial = serial.Serial(port, baudrate=baudrate, timeout=timeout_s)
        self._buffer = b""
        self._callback: "Optional[Callable[[Frame], None]]" = None

    def on_telemetry(self, callback: "Callable[[Frame], None]") -> None:
        self._callback = callback

    def send(self, frame: Frame) -> None:
        self._serial.write(encode(frame))

    def request(self, frame: Frame) -> Frame:  # pragma: no cover - needs hardware
        self.send(frame)
        while True:
            chunk = self._serial.read(256)
            if not chunk:
                raise TimeoutError(f"no response to cmd 0x{frame.cmd:02X}")
            self._buffer += chunk
            try:
                decoded, self._buffer = decode(self._buffer)
            except ValueError:
                continue
            if decoded.cmd in (0x80, 0x81) and self._callback is not None:
                self._callback(decoded)
                continue
            return decoded

    def close(self) -> None:  # pragma: no cover - needs hardware
        self._serial.close()
