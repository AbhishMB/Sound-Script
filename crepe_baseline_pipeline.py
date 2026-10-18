import numpy as np
import librosa
import torch
import torchcrepe


class CrepeBaseline:
    """
    Minimal CREPE-only pipeline:
    - Load audio
    - CREPE pitch estimation (no denoise, no mask, no gating)
    - Optional MIDI export
    """

    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.hop_length = 160  # 10 ms at 16 kHz

    def predict(self, audio_array, sr=16000):
        audio_tensor = torch.tensor(audio_array).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            f0, periodicity = torchcrepe.predict(
                audio_tensor,
                sr,
                hop_length=self.hop_length,
                fmin=50,
                fmax=2000,
                model="full",
                batch_size=2048,
                decoder=torchcrepe.decode.weighted_argmax,
                return_periodicity=True,
            )
        f0 = f0.squeeze().cpu().numpy()
        conf = periodicity.squeeze().cpu().numpy()
        time_axis = np.arange(len(f0)) * (self.hop_length / sr)
        return time_axis, f0, conf

    def run(self, audio_path, sr=16000, output_midi=None, conf_threshold=0.0):
        """
        Run CREPE baseline on a single audio file.
        If output_midi is provided, export MIDI (only frames above conf_threshold).
        """
        audio, _ = librosa.load(audio_path, sr=sr)
        t, f0, conf = self.predict(audio, sr)

        if output_midi:
            f0_for_midi = f0.copy()
            f0_for_midi[conf < conf_threshold] = 0.0
            generate_midi(t, f0_for_midi, output_midi)

        return {"time": t, "freq": f0, "conf": conf}


def hz_to_midi(freq):
    """
    Convert frequency in Hz to MIDI note number.
    MIDI = 69 + 12 * log2(freq / 440.0)
    """
    freq = np.asarray(freq, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        midi = 69 + 12 * np.log2(freq / 440.0)
    midi[freq <= 0] = 0  # Unvoiced frames
    return np.round(midi).astype(int)


def generate_midi(est_time, est_freq, output_path="output.mid", velocity=80):
    """
    Standalone MIDI writer (no dependency on experiment_pipeline.py).
    """
    try:
        import pretty_midi
    except ImportError:
        print("Error: pretty_midi not installed. Install with: pip install pretty-midi")
        return None

    midi_notes = hz_to_midi(est_freq)

    # If there are no pitched frames, still write an empty MIDI for consistency
    if np.sum(midi_notes > 0) == 0:
        midi_data = pretty_midi.PrettyMIDI()
        midi_data.write(output_path)
        print(f"  No voiced frames; wrote empty MIDI to {output_path}")
        return output_path

    midi_data = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano

    current_note = None
    note_start_time = None

    for t, note in zip(est_time, midi_notes):
        if note <= 0:
            if current_note is not None:
                note_obj = pretty_midi.Note(
                    velocity=velocity, pitch=current_note, start=note_start_time, end=t
                )
                instrument.notes.append(note_obj)
                current_note = None
            continue

        note = int(max(0, min(127, note)))

        if current_note != note:
            if current_note is not None:
                note_obj = pretty_midi.Note(
                    velocity=velocity, pitch=current_note, start=note_start_time, end=t
                )
                instrument.notes.append(note_obj)
            current_note = note
            note_start_time = t

    if current_note is not None and len(est_time) > 0:
        note_obj = pretty_midi.Note(
            velocity=velocity,
            pitch=current_note,
            start=note_start_time,
            end=est_time[-1],
        )
        instrument.notes.append(note_obj)

    midi_data.instruments.append(instrument)
    midi_data.write(output_path)
    print(f"✓ MIDI saved to {output_path}")
    return output_path


if __name__ == "__main__":
    # Example usage:
    # python crepe_baseline_pipeline.py input_song.wav [output.mid]
    import sys

    if len(sys.argv) < 2:
        print("Usage: python crepe_baseline_pipeline.py <audio_file> [output_midi]")
        sys.exit(1)

    audio_file = sys.argv[1]
    output_midi = sys.argv[2] if len(sys.argv) > 2 else None

    baseline = CrepeBaseline()
    results = baseline.run(audio_file, output_midi=output_midi, conf_threshold=0.15)

    voiced = np.sum(results["freq"] > 0)
    print(f"✓ Baseline CREPE done. Voiced frames: {voiced}/{len(results['freq'])}")
