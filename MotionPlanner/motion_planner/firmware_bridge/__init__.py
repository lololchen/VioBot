"""firmware_bridge — Native Protocol v2 host side for AachenBQ/Motor_Architecture.

The firmware repo is a days-old skeleton: every command constant here carries a
CONFIRM-WITH-FIRMWARE marker and MockTransport keeps the whole test suite green
regardless of firmware drift (D-029). Core imports stay stdlib-only; pyserial
lives behind the [serial] extra.
"""
