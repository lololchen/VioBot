"""motion_planner — Sound2Motion: NoteSequence -> MotionScore + sim + comparison.

Public surface: schema (MotionScore), hardware (HardwareProfile + geometry),
planner.plan() as the pure entry point, simulate/roundtrip/compare for
validation, firmware_bridge for the wire. See MotionPlanner/CLAUDE.md.
"""

__version__ = "0.1.0"
