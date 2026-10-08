import numpy as np

from amt.midi import build_midi, frames_to_notes, hz_to_midi

FD = 0.01


def test_hz_to_midi_reference_points():
    assert hz_to_midi(np.array([440.0, 261.6256]))[:2].round(2).tolist() == [69.0, 60.0]
    assert np.isnan(hz_to_midi(np.array([0.0]))[0])


def test_constant_pitch_is_one_note():
    notes = frames_to_notes(np.full(100, 440.0))
    assert [n[0] for n in notes] == [69]
    assert abs(notes[0][1]) < 1e-9 and abs(notes[0][2] - 1.0) < 1e-9


def test_vibrato_does_not_fragment_a_note():
    t = np.arange(150) * FD
    f0 = 440.0 * 2 ** (0.3 * np.sin(2 * np.pi * 6 * t) / 12)  # +-30 cents at 6 Hz
    assert len(frames_to_notes(f0)) == 1


def test_gap_is_bridged_only_when_short_and_pitch_matches():
    f0 = np.r_[np.full(30, 440.0), np.zeros(2), np.full(30, 440.0)]
    assert len(frames_to_notes(f0)) == 1
    f0 = np.r_[np.full(30, 440.0), np.zeros(10), np.full(30, 440.0)]
    assert len(frames_to_notes(f0)) == 2


def test_very_short_blips_are_dropped():
    f0 = np.r_[np.full(30, 440.0), np.zeros(10), np.full(2, 880.0), np.zeros(10)]
    assert [n[0] for n in frames_to_notes(f0)] == [69]


def test_unvoiced_only_gives_no_notes():
    assert frames_to_notes(np.zeros(50)) == []


def test_multitrack_midi_assigns_programs():
    pm = build_midi({"vocals": [(60, 0.0, 0.5)], "bass": [(40, 0.0, 0.5)], "other": []})
    assert [(i.name, i.program) for i in pm.instruments] == [("Vocals", 52), ("Bass", 33), ("Other", 0)]
