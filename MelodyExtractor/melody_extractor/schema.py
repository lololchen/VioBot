"""NoteSequence — the versioned, hardware-agnostic interface of MelodyExtractor.

This module is the contract consumed by MotionPlanner and AudioFeedback
(see ../CLAUDE.md "Contracts"): any change to the JSON layout requires a
schema-version bump and a docs/decisions.md entry.

Content model (PRD: NoteSequence{notes[], features[], meta}):
- notes[]    — note events. Every note carries pitch_hz (float — violin is
               continuous-pitch, never just a MIDI int), onset_s, duration_s,
               amp_db_envelope (real dB levels, not only 0-127 velocity),
               optional f0_contour (bends/vibrato) and harmonics block.
- features[] — per-frame tracks (f0, voicing confidence, amplitude) from the
               transcriber, kept alongside the segmented notes.
- meta       — provenance: source file, backends + versions, stage config
               after reduction. Needed for determinism audits.

Serialization is deterministic: the same NoteSequence always produces
byte-identical JSON (sorted keys, canonical note order, no NaN/Inf).
All dataclasses are frozen; list-like fields are coerced to tuples so a
NoteSequence can be shared safely (the reducer is pure).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace  # noqa: F401  (replace re-exported for callers)
from pathlib import Path
from typing import Any, Optional

SCHEMA_VERSION = "0.1.0"

_A4_HZ = 440.0
_A4_MIDI = 69.0


def hz_to_midi(hz: float) -> float:
    """Continuous MIDI number (float) for a frequency in Hz. hz must be > 0."""
    return _A4_MIDI + 12.0 * math.log2(hz / _A4_HZ)


def midi_to_hz(midi: float) -> float:
    """Frequency in Hz for a continuous MIDI number (float)."""
    return _A4_HZ * 2.0 ** ((midi - _A4_MIDI) / 12.0)


def _tuple_f(values) -> tuple[float, ...]:
    return tuple(float(v) for v in values)


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def _all_finite(values, what: str) -> None:
    _require(all(math.isfinite(v) for v in values), f"{what} contains NaN/Inf")


@dataclass(frozen=True)
class AmpEnvelope:
    """Sampled amplitude envelope in dBFS, times relative to note onset."""

    times_s: tuple[float, ...]
    amp_db: tuple[float, ...]

    def __post_init__(self):
        object.__setattr__(self, "times_s", _tuple_f(self.times_s))
        object.__setattr__(self, "amp_db", _tuple_f(self.amp_db))

    @classmethod
    def constant(cls, amp_db: float, duration_s: float) -> "AmpEnvelope":
        """Flat envelope — used by the MIDI input path where only velocity exists."""
        return cls(times_s=(0.0, duration_s), amp_db=(amp_db, amp_db))

    def validate(self) -> None:
        _require(len(self.times_s) == len(self.amp_db) >= 1, "envelope: times/amp length mismatch or empty")
        _all_finite(self.times_s, "envelope times_s")
        _all_finite(self.amp_db, "envelope amp_db")
        _require(all(b >= a for a, b in zip(self.times_s, self.times_s[1:])), "envelope times_s not sorted")
        _require(self.times_s[0] >= 0.0, "envelope times_s must start >= 0 (relative to onset)")

    def peak_db(self) -> float:
        return max(self.amp_db)


@dataclass(frozen=True)
class F0Contour:
    """Fine pitch trajectory (bends/vibrato), times relative to note onset."""

    times_s: tuple[float, ...]
    f0_hz: tuple[float, ...]

    def __post_init__(self):
        object.__setattr__(self, "times_s", _tuple_f(self.times_s))
        object.__setattr__(self, "f0_hz", _tuple_f(self.f0_hz))

    def validate(self) -> None:
        _require(len(self.times_s) == len(self.f0_hz) >= 1, "f0_contour: times/f0 length mismatch or empty")
        _all_finite(self.times_s, "f0_contour times_s")
        _all_finite(self.f0_hz, "f0_contour f0_hz")
        _require(all(b >= a for a, b in zip(self.times_s, self.times_s[1:])), "f0_contour times_s not sorted")
        _require(all(f > 0.0 for f in self.f0_hz), "f0_contour f0_hz must be > 0")


@dataclass(frozen=True)
class Harmonics:
    """Note-level harmonic-distribution descriptors (PRD F3).

    harmonic_amps_db[k] is the note-mean amplitude of harmonic k+1 (fundamental
    first) in dBFS. Ratios follow the Essentia definitions:
    - odd_even_ratio: energy of odd harmonics (3,5,7,...) over even (2,4,6,...),
      fundamental excluded.
    - tristimulus: (T1, T2, T3) = energy shares of h1, h2-h4, h5+ (sums to ~1).
    - inharmonicity: energy-weighted deviation of measured partials from k*f0,
      in [0, 1].
    """

    harmonic_amps_db: tuple[float, ...]
    odd_even_ratio: float
    tristimulus: tuple[float, float, float]
    inharmonicity: float

    def __post_init__(self):
        object.__setattr__(self, "harmonic_amps_db", _tuple_f(self.harmonic_amps_db))
        object.__setattr__(self, "tristimulus", _tuple_f(self.tristimulus))

    def validate(self) -> None:
        _require(len(self.harmonic_amps_db) >= 1, "harmonics: empty harmonic_amps_db")
        _all_finite(self.harmonic_amps_db, "harmonic_amps_db")
        _require(len(self.tristimulus) == 3, "tristimulus must have 3 components")
        _all_finite(self.tristimulus, "tristimulus")
        _require(math.isfinite(self.odd_even_ratio) and self.odd_even_ratio >= 0, "odd_even_ratio invalid")
        _require(math.isfinite(self.inharmonicity) and self.inharmonicity >= 0, "inharmonicity invalid")


@dataclass(frozen=True)
class Note:
    """One note event. pitch_hz is the representative (median) f0 of the note."""

    pitch_hz: float
    onset_s: float
    duration_s: float
    amp_db_envelope: AmpEnvelope
    velocity: Optional[int] = None      # 0-127, amplitude-scaled (basic-pitch); raw dB lives in the envelope
    confidence: Optional[float] = None  # 0-1 transcriber confidence
    voice: Optional[int] = None         # assigned by the reducer; 0 = principal voice
    rolled: bool = False                # chord member must be arpeggiated (stage-3 semantics, D-009)
    f0_contour: Optional[F0Contour] = None
    harmonics: Optional[Harmonics] = None

    def validate(self) -> None:
        _require(math.isfinite(self.pitch_hz) and self.pitch_hz > 0, f"pitch_hz invalid: {self.pitch_hz}")
        _require(math.isfinite(self.onset_s) and self.onset_s >= 0, f"onset_s invalid: {self.onset_s}")
        _require(math.isfinite(self.duration_s) and self.duration_s > 0, f"duration_s invalid: {self.duration_s}")
        self.amp_db_envelope.validate()
        if self.velocity is not None:
            _require(0 <= int(self.velocity) <= 127, f"velocity out of range: {self.velocity}")
        if self.confidence is not None:
            _require(0.0 <= self.confidence <= 1.0, f"confidence out of range: {self.confidence}")
        if self.f0_contour is not None:
            self.f0_contour.validate()
        if self.harmonics is not None:
            self.harmonics.validate()

    @property
    def offset_s(self) -> float:
        return self.onset_s + self.duration_s


@dataclass(frozen=True)
class FrameTrack:
    """Per-frame features (PRD F2). f0_hz == 0.0 marks unvoiced frames."""

    hop_s: float
    f0_hz: tuple[float, ...]
    voicing: tuple[float, ...]
    amp_db: tuple[float, ...]
    start_s: float = 0.0
    name: str = "f0"

    def __post_init__(self):
        object.__setattr__(self, "f0_hz", _tuple_f(self.f0_hz))
        object.__setattr__(self, "voicing", _tuple_f(self.voicing))
        object.__setattr__(self, "amp_db", _tuple_f(self.amp_db))

    def validate(self) -> None:
        _require(self.hop_s > 0, "frame track hop_s must be > 0")
        n = len(self.f0_hz)
        _require(len(self.voicing) == n and len(self.amp_db) == n, "frame track arrays must have equal length")
        _all_finite(self.f0_hz, "frame f0_hz")
        _all_finite(self.voicing, "frame voicing")
        _all_finite(self.amp_db, "frame amp_db")
        _require(all(f >= 0.0 for f in self.f0_hz), "frame f0_hz must be >= 0 (0 = unvoiced)")

    def times_s(self) -> tuple[float, ...]:
        return tuple(self.start_s + i * self.hop_s for i in range(len(self.f0_hz)))


@dataclass(frozen=True)
class Meta:
    """Provenance. backends maps stage -> 'name-version' (e.g. transcriber: 'yin-0.1.0')."""

    source: str = ""
    source_kind: str = ""            # "audio" | "midi" | "synthetic"
    sample_rate: Optional[int] = None
    backends: dict = field(default_factory=dict)
    stage: Optional[dict] = None     # StageConfig dump, present after reduction
    extra: dict = field(default_factory=dict)


@dataclass(frozen=True)
class NoteSequence:
    notes: tuple[Note, ...] = ()
    features: tuple[FrameTrack, ...] = ()
    meta: Meta = field(default_factory=Meta)

    def __post_init__(self):
        object.__setattr__(self, "notes", tuple(self.notes))
        object.__setattr__(self, "features", tuple(self.features))

    # ---------- validation ----------

    def validate(self) -> "NoteSequence":
        for n in self.notes:
            n.validate()
        for t in self.features:
            t.validate()
        return self

    def sorted(self) -> "NoteSequence":
        """Canonical note order: (onset_s, pitch_hz, duration_s). Serialization uses this."""
        return replace(self, notes=tuple(sorted(self.notes, key=lambda n: (n.onset_s, n.pitch_hz, n.duration_s))))

    # ---------- JSON ----------

    def to_json_dict(self) -> dict:
        seq = self.sorted()
        return {
            "schema_version": SCHEMA_VERSION,
            "meta": {
                "source": seq.meta.source,
                "source_kind": seq.meta.source_kind,
                "sample_rate": seq.meta.sample_rate,
                "backends": dict(seq.meta.backends),
                "stage": seq.meta.stage,
                "extra": dict(seq.meta.extra),
            },
            "notes": [_note_to_dict(n) for n in seq.notes],
            "features": [
                {
                    "name": t.name,
                    "hop_s": t.hop_s,
                    "start_s": t.start_s,
                    "f0_hz": list(t.f0_hz),
                    "voicing": list(t.voicing),
                    "amp_db": list(t.amp_db),
                }
                for t in seq.features
            ],
        }

    def to_json(self, path: "str | Path | None" = None) -> str:
        """Deterministic JSON: sorted keys, canonical note order, LF newline at EOF."""
        text = json.dumps(self.to_json_dict(), sort_keys=True, indent=1, allow_nan=False) + "\n"
        if path is not None:
            Path(path).write_text(text, encoding="utf-8", newline="\n")
        return text

    @classmethod
    def from_json_dict(cls, d: dict) -> "NoteSequence":
        version = d.get("schema_version")
        _require(isinstance(version, str) and version, "missing schema_version")
        major = version.split(".")[0]
        _require(major == SCHEMA_VERSION.split(".")[0],
                 f"incompatible schema_version {version!r} (this build reads {SCHEMA_VERSION})")
        m = d.get("meta", {})
        meta = Meta(
            source=m.get("source", ""),
            source_kind=m.get("source_kind", ""),
            sample_rate=m.get("sample_rate"),
            backends=dict(m.get("backends", {})),
            stage=m.get("stage"),
            extra=dict(m.get("extra", {})),
        )
        notes = tuple(_note_from_dict(nd) for nd in d.get("notes", []))
        features = tuple(
            FrameTrack(
                hop_s=td["hop_s"],
                f0_hz=td["f0_hz"],
                voicing=td["voicing"],
                amp_db=td["amp_db"],
                start_s=td.get("start_s", 0.0),
                name=td.get("name", "f0"),
            )
            for td in d.get("features", [])
        )
        return cls(notes=notes, features=features, meta=meta).validate()

    @classmethod
    def from_json(cls, source: "str | Path") -> "NoteSequence":
        """Load from a JSON string or a file path."""
        if isinstance(source, Path) or (isinstance(source, str) and "\n" not in source and source.strip().endswith(".json")):
            text = Path(source).read_text(encoding="utf-8")
        else:
            text = source
        return cls.from_json_dict(json.loads(text))

    # ---------- MIDI export (PRD F6) ----------

    def to_midi(self, path: "str | Path") -> None:
        """Export via pretty_midi. One instrument per voice (channel-safe pitch
        bends); nearest-semitone note + constant per-note bend when the pitch
        deviates > 5 cents from equal temperament. Velocity from note.velocity,
        else scaled from the envelope peak. Lossy: contours/harmonics stay in JSON.
        """
        import pretty_midi

        seq = self.sorted()
        pm = pretty_midi.PrettyMIDI(resolution=960)
        voices = sorted({n.voice if n.voice is not None else 0 for n in seq.notes})
        instruments = {}
        for v in voices:
            inst = pretty_midi.Instrument(program=40, name=f"voice{v}")  # 40 = violin
            instruments[v] = inst
            pm.instruments.append(inst)
        for n in seq.notes:
            v = n.voice if n.voice is not None else 0
            midi_f = hz_to_midi(n.pitch_hz)
            midi_i = int(round(midi_f))
            midi_i = min(127, max(0, midi_i))
            dev_semitones = midi_f - midi_i
            velocity = n.velocity if n.velocity is not None else _db_to_velocity(n.amp_db_envelope.peak_db())
            instruments[v].notes.append(
                pretty_midi.Note(velocity=velocity, pitch=midi_i, start=n.onset_s, end=n.offset_s)
            )
            if abs(dev_semitones) * 100.0 > 5.0:  # > 5 cents
                bend = int(round(dev_semitones / 2.0 * 8192))  # assumes default ±2 semitone bend range
                bend = min(8191, max(-8192, bend))
                instruments[v].pitch_bends.append(pretty_midi.PitchBend(pitch=bend, time=n.onset_s))
                instruments[v].pitch_bends.append(pretty_midi.PitchBend(pitch=0, time=n.offset_s))
        pm.write(str(path))


def _db_to_velocity(peak_db: float) -> int:
    """Map dBFS peak to MIDI velocity: -60 dB -> 1, 0 dB -> 127, linear in dB."""
    v = int(round((max(-60.0, min(0.0, peak_db)) + 60.0) / 60.0 * 126.0)) + 1
    return min(127, max(1, v))


def _note_to_dict(n: Note) -> dict:
    d: dict[str, Any] = {
        "pitch_hz": n.pitch_hz,
        "onset_s": n.onset_s,
        "duration_s": n.duration_s,
        "amp_db_envelope": {"times_s": list(n.amp_db_envelope.times_s), "amp_db": list(n.amp_db_envelope.amp_db)},
        "rolled": n.rolled,
    }
    if n.velocity is not None:
        d["velocity"] = int(n.velocity)
    if n.confidence is not None:
        d["confidence"] = n.confidence
    if n.voice is not None:
        d["voice"] = int(n.voice)
    if n.f0_contour is not None:
        d["f0_contour"] = {"times_s": list(n.f0_contour.times_s), "f0_hz": list(n.f0_contour.f0_hz)}
    if n.harmonics is not None:
        d["harmonics"] = {
            "harmonic_amps_db": list(n.harmonics.harmonic_amps_db),
            "odd_even_ratio": n.harmonics.odd_even_ratio,
            "tristimulus": list(n.harmonics.tristimulus),
            "inharmonicity": n.harmonics.inharmonicity,
        }
    return d


def _note_from_dict(d: dict) -> Note:
    env = d["amp_db_envelope"]
    contour = d.get("f0_contour")
    harm = d.get("harmonics")
    return Note(
        pitch_hz=d["pitch_hz"],
        onset_s=d["onset_s"],
        duration_s=d["duration_s"],
        amp_db_envelope=AmpEnvelope(times_s=env["times_s"], amp_db=env["amp_db"]),
        velocity=d.get("velocity"),
        confidence=d.get("confidence"),
        voice=d.get("voice"),
        rolled=d.get("rolled", False),
        f0_contour=F0Contour(times_s=contour["times_s"], f0_hz=contour["f0_hz"]) if contour else None,
        harmonics=Harmonics(
            harmonic_amps_db=harm["harmonic_amps_db"],
            odd_even_ratio=harm["odd_even_ratio"],
            tristimulus=tuple(harm["tristimulus"]),
            inharmonicity=harm["inharmonicity"],
        ) if harm else None,
    )
