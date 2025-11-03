# transcribe_crepe.py
import numpy as np
import crepe
import pretty_midi
from scipy.ndimage import median_filter


def freq_to_midi(freq):
    """Convert frequency array (Hz) to MIDI note numbers (float). NaN preserved."""
    midi = 69 + 12 * np.log2(freq / 440.0)
    return midi


def smooth_and_quantize(midi_array, conf, min_conf=0.5, med_len=3):
    """
    Apply confidence masking, median smoothing, and quantize to nearest integer MIDI pitch.
    """
    midi = midi_array.copy()
    midi[conf < min_conf] = np.nan
    # median filter ignoring NaNs: replace NaN with a sentinel then revert small windows
    sentinel = -999
    arr = np.where(np.isnan(midi), sentinel, midi)
    arr = median_filter(arr, size=med_len)
    arr = np.where(arr == sentinel, np.nan, arr)
    # quantize to nearest MIDI int (but keep NaNs)
    q = np.where(np.isnan(arr), np.nan, np.rint(arr).astype(np.int32))
    return q


def midi_array_to_notes(q_midi, times, min_note_dur=0.06):
    """
    Convert quantized MIDI array (ints or NaN) + times -> list of pretty_midi.Note.
    Merges contiguous frames with same pitch.
    """
    notes = []
    if len(q_midi) == 0:
        return notes
    cur_pitch = q_midi[0]
    start_time = times[0]
    for i in range(1, len(q_midi)):
        if np.isnan(cur_pitch) and np.isnan(q_midi[i]):
            # still silence
            start_time = times[i]
            continue
        if (
            (np.isnan(cur_pitch) and not np.isnan(q_midi[i]))
            or (not np.isnan(cur_pitch) and np.isnan(q_midi[i]))
            or (
                not np.isnan(cur_pitch)
                and not np.isnan(q_midi[i])
                and q_midi[i] != cur_pitch
            )
        ):
            # pitch changed or silence boundaries
            end_time = times[i]
            if not np.isnan(cur_pitch):
                if end_time - start_time >= min_note_dur:
                    notes.append((int(cur_pitch), float(start_time), float(end_time)))
            start_time = times[i]
            cur_pitch = q_midi[i]
    # finalize last note
    end_time = times[-1] + (times[1] - times[0] if len(times) > 1 else 0.01)
    if not np.isnan(cur_pitch) and end_time - start_time >= min_note_dur:
        notes.append((int(cur_pitch), float(start_time), float(end_time)))
    return notes


def transcribe_audio(y, sr, step_size=10, min_confidence=0.5):
    """
    Transcribe monophonic audio waveform y (numpy array) sampled at sr.
    Returns pretty_midi.PrettyMIDI object.
    - step_size (ms) default 10
    - min_confidence default 0.5
    """
    # CREPE returns times (s), frequency (Hz), confidence, activation
    time, frequency, confidence, _ = crepe.predict(
        y, sr, viterbi=True, step_size=step_size
    )
    # convert Hz -> MIDI (float), keep nan where frequency is 0 or invalid
    frequency = np.where(frequency <= 0, np.nan, frequency)
    midi_float = freq_to_midi(frequency)
    # smoothing + quantize
    q_midi = smooth_and_quantize(
        midi_float, confidence, min_conf=min_confidence, med_len=5
    )
    # build notes
    notes = midi_array_to_notes(q_midi, time, min_note_dur=0.05)

    # build PrettyMIDI
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    for pitch, start, end in notes:
        # clip pitch to valid range
        p = max(0, min(127, pitch))
        note = pretty_midi.Note(velocity=100, pitch=p, start=start, end=end)
        inst.notes.append(note)
    pm.instruments.append(inst)
    return pm
