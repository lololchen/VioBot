"""Deterministic fixture corpus: {name}.wav (16 kHz additive render) + {name}.mid
(ground truth). Regenerating always produces identical bytes — the corpus is
also a determinism probe (algorithm-validation skill).

Stage coverage: mono scale + arpeggio (stage 1), adjacent-string-feasible
thirds (stage 2), a feasible rolled triple (stage 3). All pitches are exact
equal temperament so ground-truth MIDI rounding is lossless.
"""
from __future__ import annotations

import sys
from pathlib import Path

if str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import synth_util  # noqa: E402

# Exact equal-tempered frequencies (A4 = 440).
G3 = 195.99771799087463
B3 = 246.94165062806206
C4 = 261.6255653005986
D4 = 293.6647679174076
E4 = 329.6275569128699
F4 = 349.2282314330039
G4 = 391.99543598174927
A4 = 440.0
B4 = 493.8833012561241
C5 = 523.2511306011972
D5 = 587.3295358348151
E5 = 659.2551138257398
FS5 = 739.9888454232688
G5 = 783.9908719634985
B5 = 987.7666025122483


def _mono_scale():
    pitches = [C4, D4, E4, F4, G4, A4, B4, C5]
    return [(p, i * 0.5, 0.48) for i, p in enumerate(pitches)]


def _mono_arpeggio():
    pitches = [G3, B3, D4, G4, B4, D5, G5]
    return [(p, i * 0.5, 0.4) for i, p in enumerate(pitches)]


def _two_voice_thirds():
    chords = [(C5, E5), (D5, FS5), (E5, G5)]
    notes = []
    for i, (lo, hi) in enumerate(chords):
        onset = i * 0.65
        notes.append((lo, onset, 0.6))
        notes.append((hi, onset, 0.6))
    return notes


def _triple_rolled():
    return [
        (A4, 0.0, 0.5),
        (G4, 0.6, 1.0),
        (D5, 0.6, 1.0),
        (B5, 0.6, 1.0),
        (E5, 1.7, 0.5),
    ]


FIXTURES = {
    "mono_scale": _mono_scale,
    "mono_arpeggio": _mono_arpeggio,
    "two_voice_thirds": _two_voice_thirds,
    "triple_rolled": _triple_rolled,
}


def generate_all(out_dir: "str | Path") -> "list[str]":
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    names = []
    for name, make in sorted(FIXTURES.items()):
        notes = make()
        pcm = synth_util.sequence_audio(notes)
        synth_util.write_wav(out / f"{name}.wav", pcm)
        synth_util.write_ground_truth_midi(out / f"{name}.mid", notes)
        names.append(name)
    return names


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent
    print("generated:", ", ".join(generate_all(target)))
