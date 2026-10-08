import numpy as np
import pretty_midi
from scipy.ndimage import median_filter

from .config import HOP_LENGTH, SAMPLE_RATE, STEM_INSTRUMENT_MAP

FRAME_DUR = HOP_LENGTH / SAMPLE_RATE


def hz_to_midi(f0: np.ndarray) -> np.ndarray:
    """MIDI = 69 + 12 log2(f0 / 440); unvoiced (f0 <= 0) frames become NaN."""
    f0 = np.asarray(f0, dtype=float)
    out = np.full(f0.shape, np.nan)
    v = f0 > 0
    out[v] = 69.0 + 12.0 * np.log2(f0[v] / 440.0)
    return out


def frames_to_notes(
    gated_f0: np.ndarray,
    min_note_dur: float = 0.05,
    max_gap: float = 0.03,
    smooth_frames: int = 5,
    frame_dur: float = FRAME_DUR,
) -> list[tuple[int, float, float]]:
    """Gated f0 contour -> [(midi_pitch, start_s, end_s)].

    Rounding a jittery contour frame by frame produces many tiny notes, so the
    MIDI pitch is median-smoothed over voiced runs first, gaps shorter than
    `max_gap` are bridged when the pitch on both sides agrees, and notes shorter
    than `min_note_dur` are dropped.
    """
    midi = hz_to_midi(gated_f0)
    voiced = ~np.isnan(midi)
    if not voiced.any():
        return []
    smoothed = np.full_like(midi, np.nan)
    edges = np.flatnonzero(np.diff(np.r_[0, voiced.astype(int), 0]))
    for s, e in zip(edges[::2], edges[1::2]):
        run = midi[s:e]
        smoothed[s:e] = median_filter(run, size=min(smooth_frames, len(run) | 1), mode="nearest")
    q = np.rint(smoothed)

    notes = []
    i, n = 0, len(q)
    while i < n:
        if np.isnan(q[i]):
            i += 1
            continue
        j = i
        while j < n and q[j] == q[i]:
            j += 1
        notes.append([int(q[i]), i * frame_dur, j * frame_dur])
        i = j
    merged: list[list] = []
    for p, s, e in notes:
        if merged and merged[-1][0] == p and s - merged[-1][2] <= max_gap:
            merged[-1][2] = e
        else:
            merged.append([p, s, e])
    return [
        (int(np.clip(p, 0, 127)), s, e) for p, s, e in merged if e - s >= min_note_dur
    ]


def build_midi(stem_notes: dict[str, list[tuple[int, float, float]]]) -> pretty_midi.PrettyMIDI:
    pm = pretty_midi.PrettyMIDI()
    for name, notes in stem_notes.items():
        inst = pretty_midi.Instrument(program=STEM_INSTRUMENT_MAP.get(name, 0), name=name.capitalize())
        for pitch, start, end in notes:
            inst.notes.append(pretty_midi.Note(velocity=100, pitch=pitch, start=start, end=end))
        pm.instruments.append(inst)
    return pm
