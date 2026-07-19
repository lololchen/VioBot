"""Input adapter tests: ffmpeg discovery, audio decode + deterministic
resample (direct-soundfile and ffmpeg-decode paths), MIDI parsing
(pitch/onset/duration/velocity/bends), and extension-based dispatch."""
from __future__ import annotations

import subprocess

import numpy as np
import pretty_midi
import pytest
import soundfile as sf

import synth_util

from melody_extractor.input_adapter import (
    AUDIO_EXTENSIONS,
    TARGET_SR,
    AudioBuffer,
    _velocity_to_db,
    find_ffmpeg,
    load,
    load_audio,
    load_midi,
)
from melody_extractor.schema import NoteSequence, midi_to_hz


def _dominant_freq(pcm: np.ndarray, sample_rate: int) -> float:
    """FFT bin with peak magnitude, as a frequency in Hz."""
    spec = np.fft.rfft(pcm.astype(np.float64))
    freqs = np.fft.rfftfreq(len(pcm), d=1.0 / sample_rate)
    return float(freqs[int(np.argmax(np.abs(spec)))])


def _write_midi(path, pitches_midi, onsets, durations, velocities, program=40):
    pm = pretty_midi.PrettyMIDI(resolution=960)
    inst = pretty_midi.Instrument(program=program, name="violin")
    for pitch, onset, dur, vel in zip(pitches_midi, onsets, durations, velocities):
        inst.notes.append(pretty_midi.Note(velocity=vel, pitch=pitch, start=onset, end=onset + dur))
    pm.instruments.append(inst)
    pm.write(str(path))
    return path


# ---------- audio decode: direct soundfile path ----------


def test_wav_stereo_44100_to_16000_mono_preserves_length_ratio_and_frequency(tmp_path):
    freq = 440.0
    duration = 1.0
    src_sr = 44100
    mono = synth_util.harmonic_tone(freq, duration, sample_rate=src_sr)
    # Distinct per-channel gain exercises the mean-mixdown (not just a copy).
    stereo = np.stack([mono, mono * 0.5], axis=1)
    wav_path = tmp_path / "stereo.wav"
    sf.write(str(wav_path), stereo.astype(np.float32), src_sr, subtype="FLOAT")

    buf = load_audio(wav_path, target_sr=16000)

    assert isinstance(buf, AudioBuffer)
    assert buf.sample_rate == 16000
    assert buf.pcm.dtype == np.float32
    assert buf.pcm.ndim == 1
    assert buf.source == wav_path.name

    expected_len = round(len(mono) * 16000 / src_sr)
    assert abs(len(buf.pcm) - expected_len) <= max(1, round(0.01 * expected_len))

    dom = _dominant_freq(buf.pcm, buf.sample_rate)
    assert dom == pytest.approx(freq, rel=0.01)


def test_load_audio_flac(tmp_path):
    freq = 300.0
    duration = 0.5
    sr = 16000
    mono = synth_util.harmonic_tone(freq, duration, sample_rate=sr)
    flac_path = tmp_path / "tone.flac"
    sf.write(str(flac_path), mono.astype(np.float32), sr, subtype="PCM_16")

    buf = load_audio(flac_path)

    assert buf.sample_rate == TARGET_SR
    assert buf.pcm.dtype == np.float32
    assert len(buf.pcm) == round(duration * TARGET_SR)
    dom = _dominant_freq(buf.pcm, buf.sample_rate)
    assert dom == pytest.approx(freq, rel=0.02)


def test_load_audio_aiff(tmp_path):
    mono = synth_util.harmonic_tone(220.0, 0.3, sample_rate=TARGET_SR)
    path = tmp_path / "tone.aiff"
    sf.write(str(path), mono.astype(np.float32), TARGET_SR, format="AIFF", subtype="PCM_16")

    buf = load_audio(path)

    assert buf.sample_rate == TARGET_SR
    assert buf.pcm.dtype == np.float32
    assert len(buf.pcm) == round(0.3 * TARGET_SR)


def test_load_audio_float_pcm_range(tmp_path):
    mono = synth_util.harmonic_tone(220.0, 0.3, sample_rate=TARGET_SR, peak=0.9)
    path = tmp_path / "tone.wav"
    synth_util.write_wav(path, mono, sample_rate=TARGET_SR)

    buf = load_audio(path)

    assert buf.pcm.dtype == np.float32
    assert np.max(np.abs(buf.pcm)) <= 1.0
    assert np.max(np.abs(buf.pcm)) > 0.5  # not accidentally zeroed or over-attenuated


def test_load_audio_mono_source_unchanged_by_mixdown(tmp_path):
    """A single-channel source must pass through the mean-mixdown unaltered."""
    mono = synth_util.harmonic_tone(261.63, 0.4, sample_rate=TARGET_SR)
    path = tmp_path / "mono.wav"
    synth_util.write_wav(path, mono, sample_rate=TARGET_SR)

    buf = load_audio(path)

    assert len(buf.pcm) == len(mono)
    np.testing.assert_allclose(buf.pcm, mono.astype(np.float32), atol=1e-6)


def test_load_audio_no_resample_when_rate_already_matches(tmp_path):
    mono = synth_util.harmonic_tone(440.0, 0.25, sample_rate=TARGET_SR)
    path = tmp_path / "already16k.wav"
    synth_util.write_wav(path, mono, sample_rate=TARGET_SR)

    buf = load_audio(path, target_sr=TARGET_SR)

    assert len(buf.pcm) == len(mono)


# ---------- ffmpeg discovery + decode path ----------


def test_find_ffmpeg_returns_usable_path():
    exe = find_ffmpeg()
    assert exe, "imageio-ffmpeg is installed; find_ffmpeg() must resolve a binary"
    proc = subprocess.run([exe, "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.returncode == 0
    assert b"ffmpeg version" in proc.stdout.lower()


def test_mp3_roundtrip_via_ffmpeg(tmp_path):
    exe = find_ffmpeg()
    assert exe

    probe = subprocess.run([exe, "-hide_banner", "-encoders"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if b"libmp3lame" not in probe.stdout:
        pytest.skip("no mp3 encoder available in this ffmpeg build")

    freq = 440.0
    sr = 44100
    mono = synth_util.harmonic_tone(freq, 1.0, sample_rate=sr)
    wav_path = tmp_path / "src.wav"
    sf.write(str(wav_path), mono.astype(np.float32), sr, subtype="FLOAT")
    mp3_path = tmp_path / "src.mp3"
    enc = subprocess.run(
        [exe, "-v", "error", "-y", "-i", str(wav_path), "-codec:a", "libmp3lame", "-q:a", "2", str(mp3_path)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if enc.returncode != 0 or not mp3_path.exists():
        pytest.skip(f"mp3 encode failed in this environment: {enc.stderr.decode(errors='replace')}")

    buf = load_audio(mp3_path, target_sr=16000)

    assert buf.sample_rate == 16000
    assert buf.pcm.dtype == np.float32
    dom = _dominant_freq(buf.pcm, buf.sample_rate)
    assert dom == pytest.approx(freq, rel=0.02)


def test_load_audio_missing_ffmpeg_raises_clear_runtime_error(tmp_path, monkeypatch):
    import melody_extractor.input_adapter as ia

    monkeypatch.setattr(ia, "find_ffmpeg", lambda: None)
    fake_mp3 = tmp_path / "fake.mp3"
    fake_mp3.write_bytes(b"not really mp3 data")
    with pytest.raises(RuntimeError, match="ffmpeg"):
        ia.load_audio(fake_mp3)


# ---------- MIDI ----------


def test_load_midi_pitch_onset_duration_velocity(tmp_path):
    pitches_midi = [60, 64, 67, 72]  # C4 E4 G4 C5, equal tempered
    onsets = [0.0, 0.5, 1.0, 1.75]
    durations = [0.5, 0.5, 0.75, 0.25]
    velocities = [1, 64, 90, 127]
    path = _write_midi(tmp_path / "notes.mid", pitches_midi, onsets, durations, velocities)

    seq = load_midi(path)

    assert isinstance(seq, NoteSequence)
    seq.validate()
    assert len(seq.notes) == 4
    assert seq.meta.source == path.name
    assert seq.meta.source_kind == "midi"
    assert seq.meta.backends == {"input_adapter": "midi-0.1.0"}

    for note, midi_pitch, onset, dur, vel in zip(seq.notes, pitches_midi, onsets, durations, velocities):
        assert note.pitch_hz == pytest.approx(midi_to_hz(float(midi_pitch)), rel=1e-12)
        assert note.onset_s == pytest.approx(onset, abs=2e-3)
        assert note.duration_s == pytest.approx(dur, abs=2e-3)
        assert note.velocity == vel
        assert note.confidence == 1.0
        expected_db = (vel - 1) / 126.0 * 60.0 - 60.0
        assert note.amp_db_envelope.amp_db[0] == pytest.approx(expected_db)
        assert note.amp_db_envelope.amp_db[-1] == pytest.approx(expected_db)
        assert note.amp_db_envelope.times_s[0] == pytest.approx(0.0)
        assert note.amp_db_envelope.times_s[-1] == pytest.approx(dur, abs=2e-3)
        assert note.f0_contour is None


def test_velocity_to_db_matches_inverse_of_schema_mapping():
    # schema._db_to_velocity: v = round((clamp(db,-60,0)+60)/60*126) + 1
    # so the inverse at the endpoints must be exact.
    assert _velocity_to_db(1) == pytest.approx(-60.0)
    assert _velocity_to_db(127) == pytest.approx(0.0)
    assert _velocity_to_db(64) == pytest.approx((64 - 1) / 126.0 * 60.0 - 60.0)


def test_load_midi_pitch_bends_attach_f0_contour(tmp_path):
    pm = pretty_midi.PrettyMIDI(resolution=960)
    inst = pretty_midi.Instrument(program=40)
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=69, start=0.0, end=1.0))  # A4
    inst.pitch_bends.append(pretty_midi.PitchBend(pitch=4096, time=0.25))   # +1 semitone
    inst.pitch_bends.append(pretty_midi.PitchBend(pitch=-4096, time=0.75))  # -1 semitone
    pm.instruments.append(inst)
    path = tmp_path / "bend.mid"
    pm.write(str(path))

    seq = load_midi(path)
    note = seq.notes[0]

    assert note.f0_contour is not None
    note.f0_contour.validate()
    assert note.f0_contour.times_s == pytest.approx((0.25, 0.75), abs=2e-3)
    a4 = midi_to_hz(69.0)
    expected_up = a4 * (2.0 ** (1.0 / 12.0))
    expected_down = a4 * (2.0 ** (-1.0 / 12.0))
    assert note.f0_contour.f0_hz == pytest.approx((expected_up, expected_down), rel=1e-9)


def test_load_midi_note_without_bends_has_no_contour(tmp_path):
    path = _write_midi(tmp_path / "plain.mid", [60], [0.0], [1.0], [90])
    seq = load_midi(path)
    assert seq.notes[0].f0_contour is None


def test_load_midi_does_not_touch_audio_pipeline(tmp_path, monkeypatch):
    """MIDI input must not round-trip through audio (repo contract)."""
    import melody_extractor.input_adapter as ia

    def _boom(*args, **kwargs):
        raise AssertionError("load_midi must not call load_audio")

    monkeypatch.setattr(ia, "load_audio", _boom)
    path = _write_midi(tmp_path / "plain.mid", [60], [0.0], [1.0], [90])
    seq = ia.load_midi(path)
    assert len(seq.notes) == 1


def test_load_midi_multi_instrument_canonical_sort(tmp_path):
    pm = pretty_midi.PrettyMIDI(resolution=960)
    inst_a = pretty_midi.Instrument(program=40, name="a")
    inst_a.notes.append(pretty_midi.Note(velocity=90, pitch=72, start=0.5, end=1.0))
    inst_b = pretty_midi.Instrument(program=41, name="b")
    inst_b.notes.append(pretty_midi.Note(velocity=90, pitch=60, start=0.0, end=0.5))
    pm.instruments.append(inst_a)
    pm.instruments.append(inst_b)
    path = tmp_path / "multi.mid"
    pm.write(str(path))

    seq = load_midi(path)

    assert [n.onset_s for n in seq.notes] == pytest.approx([0.0, 0.5], abs=2e-3)


def test_load_midi_is_deterministic(tmp_path):
    path = _write_midi(tmp_path / "det.mid", [60, 64], [0.0, 0.3], [0.3, 0.3], [70, 100])
    a = load_midi(path).to_json()
    b = load_midi(path).to_json()
    assert a == b


# ---------- dispatch ----------


@pytest.mark.parametrize("ext,fmt,subtype", [
    (".wav", None, "FLOAT"),
    (".flac", None, "PCM_16"),
    (".aiff", "AIFF", "PCM_16"),
])
def test_load_dispatches_directly_readable_audio(tmp_path, ext, fmt, subtype):
    mono = synth_util.harmonic_tone(440.0, 0.2, sample_rate=TARGET_SR)
    path = tmp_path / f"tone{ext}"
    if fmt:
        sf.write(str(path), mono.astype(np.float32), TARGET_SR, format=fmt, subtype=subtype)
    else:
        sf.write(str(path), mono.astype(np.float32), TARGET_SR, subtype=subtype)

    result = load(path)

    assert isinstance(result, AudioBuffer)


def test_load_dispatches_midi(tmp_path):
    path = _write_midi(tmp_path / "d.mid", [69], [0.0], [1.0], [90])
    result = load(path)
    assert isinstance(result, NoteSequence)


def test_load_unknown_extension_raises_value_error(tmp_path):
    path = tmp_path / "song.xyz"
    path.write_bytes(b"garbage")
    with pytest.raises(ValueError, match="unsupported"):
        load(path)


def test_load_audio_unknown_extension_raises_value_error(tmp_path):
    path = tmp_path / "song.xyz"
    path.write_bytes(b"garbage")
    with pytest.raises(ValueError, match="unsupported"):
        load_audio(path)


def test_audio_extensions_and_midi_extensions_disjoint():
    from melody_extractor.input_adapter import MIDI_EXTENSIONS

    assert AUDIO_EXTENSIONS.isdisjoint(MIDI_EXTENSIONS)


def test_ytdlp_native_formats_route_through_ffmpeg():
    """.webm/.opus (yt-dlp bestaudio containers, D-017) are accepted audio
    extensions but must NOT be read by soundfile directly -- they decode via
    the ffmpeg branch."""
    from melody_extractor.input_adapter import _DIRECT_SOUNDFILE_EXTENSIONS

    assert {".webm", ".opus"} <= AUDIO_EXTENSIONS
    assert _DIRECT_SOUNDFILE_EXTENSIONS.isdisjoint({".webm", ".opus"})
