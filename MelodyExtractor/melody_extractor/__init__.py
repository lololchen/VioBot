"""MelodyExtractor — perception front-end of the violin robot.

Pipeline: InputAdapter -> Transcriber -> (Timbre) -> Reducer -> Exporter,
with SoundSim attachable to any stage output. See ../CLAUDE.md and
../../docs/PRD_MelodyExtractor.md.

Core stays importable without heavy optional deps (tensorflow/crepe,
basic-pitch, essentia, fluidsynth) — modules guard those imports.
"""
__version__ = "0.1.0"

from .schema import (  # noqa: F401
    SCHEMA_VERSION,
    AmpEnvelope,
    F0Contour,
    FrameTrack,
    Harmonics,
    Meta,
    Note,
    NoteSequence,
    hz_to_midi,
    midi_to_hz,
)
