"""
Hybrid CREPE Pipeline for Polyphonic Music Transcription
=========================================================
Complete implementation of the research paper methodology:
  Decomposition (Demucs v4) -> Dual-Path Processing -> Fusion -> MIDI

Usage:
    python hybrid_pipeline.py <audio_file> [--output_dir OUTPUT] [--stems vocals,bass]
"""

import os
import sys
import argparse
import subprocess
import numpy as np
import librosa
import torch
import torchcrepe
import soundfile as sf
from scipy.signal import butter, lfilter
from scipy.ndimage import median_filter

# ---------------------------------------------------------------------------
# Constants (consistent across all steps as per paper)
# ---------------------------------------------------------------------------
SR = 16000           # Sample rate for pitch estimation
HOP_LENGTH = 160     # 10 ms at 16 kHz  (consistent across CREPE, Mel, Chroma)
N_MELS = 256         # High-resolution Mel filter bank
CONF_THRESH = 0.15   # Periodicity confidence threshold for fusion gating
FMIN = 50            # Minimum frequency (Hz) – vocal range lower bound
FMAX = 2000          # Maximum frequency (Hz) – vocal range upper bound
ALL_STEMS = ["vocals", "bass", "drums", "other"]

# MIDI program numbers for instrument assignment
STEM_PROGRAMS = {
    "vocals": 52,   # Choir Aahs
    "bass":   33,   # Electric Bass (finger)
    "drums":  0,    # Acoustic Grand Piano (placeholder; drums use channel 10)
    "other":  0,    # Acoustic Grand Piano
}


# ============================================================
# A) Source Separator  –  Demucs v4 (htdemucs)
# ============================================================

class SourceSeparator:
    """
    Demucs v4 (htdemucs) pre-trained source separation.

    Splits a polyphonic audio mixture into four stems:
      vocals, drums, bass, other

    The htdemucs model uses a dual-branch architecture:
      - Temporal branch  (1-D convolutions on raw waveform)
      - Spectral branch  (2-D convolutions on spectrogram)
    combined via a Transformer cross-attention mechanism.
    """

    def __init__(self, model_name: str = "htdemucs", device: str = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def separate(self, input_file: str, output_root: str = "separated_stems") -> dict:
        """
        Run Demucs separation on *input_file*.

        Returns
        -------
        dict  : stem_name -> absolute path to the separated WAV file
                e.g. {"vocals": ".../vocals.wav", "bass": ".../bass.wav", ...}
        """
        filename = os.path.splitext(os.path.basename(input_file))[0]
        out_dir = os.path.join(output_root, self.model_name, filename)

        # Skip if stems already exist
        if self._stems_exist(out_dir):
            print(f"[SourceSeparator] Stems already exist at {out_dir}, skipping.")
            return self._build_stem_dict(out_dir)

        print(f"[SourceSeparator] Running {self.model_name} on {input_file} ...")

        # Try the demucs Python API first; fall back to CLI
        try:
            self._run_via_api(input_file, output_root)
        except Exception as api_err:
            print(f"[SourceSeparator] API failed ({api_err}), trying CLI ...")
            self._run_via_cli(input_file, output_root)

        if not self._stems_exist(out_dir):
            raise RuntimeError(
                f"Separation failed – expected stems not found in {out_dir}"
            )

        print(f"[SourceSeparator] Stems saved to {out_dir}")
        return self._build_stem_dict(out_dir)

    # -- internal helpers --------------------------------------------------

    def _run_via_api(self, input_file, output_root):
        """Use demucs Python API (demucs >= 4.0)."""
        from demucs.api import Separator

        separator = Separator(model=self.model_name, device=self.device)
        _, outputs = separator.separate_audio_file(input_file)

        filename = os.path.splitext(os.path.basename(input_file))[0]
        out_dir = os.path.join(output_root, self.model_name, filename)
        os.makedirs(out_dir, exist_ok=True)

        for stem_name, stem_tensor in outputs.items():
            stem_np = stem_tensor.cpu().numpy()
            # demucs outputs shape: (channels, samples)
            if stem_np.ndim == 2:
                stem_np = stem_np.T  # -> (samples, channels)
            out_path = os.path.join(out_dir, f"{stem_name}.wav")
            sf.write(out_path, stem_np, separator.samplerate)

    def _run_via_cli(self, input_file, output_root):
        """Fallback: call demucs via command line."""
        cmd = [
            sys.executable, "-m", "demucs",
            "--name", self.model_name,
            "--out", output_root,
            input_file,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"demucs CLI failed:\n{result.stderr}")

    @staticmethod
    def _stems_exist(out_dir: str) -> bool:
        if not os.path.isdir(out_dir):
            return False
        for stem in ALL_STEMS:
            if not os.path.isfile(os.path.join(out_dir, f"{stem}.wav")):
                return False
        return True

    @staticmethod
    def _build_stem_dict(out_dir: str) -> dict:
        return {
            stem: os.path.join(out_dir, f"{stem}.wav")
            for stem in ALL_STEMS
        }


# ============================================================
# B) Pitch Estimator  –  Path A (pre-trained CREPE "full")
# ============================================================

class PitchEstimator:
    """
    CREPE: a 6-layer convolutional neural network for pitch estimation.

    Used in its pre-trained "full" capacity on raw audio waveforms.
    Produces a high-resolution f0 contour spanning six octaves (C1–B7)
    with 20-cent resolution, plus a periodicity confidence score.
    """

    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.hop_length = HOP_LENGTH

    def predict(self, audio_array: np.ndarray, sr: int = SR):
        """
        Returns
        -------
        est_time : np.ndarray – timestamps in seconds
        est_freq : np.ndarray – fundamental frequency (Hz), 0 = unvoiced
        est_conf : np.ndarray – periodicity confidence [0, 1]
        """
        audio_tensor = (
            torch.tensor(audio_array).unsqueeze(0).float().to(self.device)
        )
        with torch.no_grad():
            f0, periodicity = torchcrepe.predict(
                audio_tensor,
                sr,
                hop_length=self.hop_length,
                fmin=FMIN,
                fmax=FMAX,
                model="full",
                batch_size=2048,
                decoder=torchcrepe.decode.weighted_argmax,
                return_periodicity=True,
            )
        f0 = f0.squeeze().cpu().numpy()
        conf = periodicity.squeeze().cpu().numpy()
        time_axis = np.arange(len(f0)) * (self.hop_length / sr)
        return time_axis, f0, conf


# ============================================================
# C) Voicing Detector  –  Path B (Mel spectrogram + Chroma)
# ============================================================

class VoicingDetector:
    """
    Voice-activity / voicing-mask detector built from Mel-spectrogram
    and Chroma features.

    Process:
      1. Compute Mel spectrogram (256 triangular filters)
      2. Derive Chroma features (12 pitch classes) from Mel
      3. Average chroma energy per frame
      4. Threshold at 10th percentile (empirical)
      5. Smooth with median filter (size 3)

    Result: binary voicing mask (True = musical pitch, False = noise/silence).
    """

    def __init__(self, n_mels: int = N_MELS, hop_length: int = HOP_LENGTH):
        self.n_mels = n_mels
        self.hop_length = hop_length

    def get_voicing_mask(self, audio: np.ndarray, sr: int = SR) -> np.ndarray:
        """Return a boolean voicing mask aligned to hop_length frames."""
        # Step 1 – Mel spectrogram
        S = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_mels=self.n_mels, hop_length=self.hop_length
        )
        # Step 2 – Chroma from Mel spectrogram
        chroma = librosa.feature.chroma_stft(S=S, sr=sr)

        # Step 3 – Average chroma energy per frame
        # E_chroma(t) = (1/12) * sum_{p=0}^{11} Chroma(p, t)
        energy = np.mean(chroma, axis=0)

        # Step 4 – Dynamic threshold at 10th percentile
        thresh = np.percentile(energy, 10)
        mask = energy > thresh

        # Step 5 – Median-filter smoothing (remove flickering)
        mask = median_filter(mask.astype(float), size=3) > 0.5

        return mask


# ============================================================
# D) Spectral Subtraction Denoiser
# ============================================================

def denoise_audio(audio: np.ndarray, n_fft: int = 1024,
                  hop_length: int = HOP_LENGTH) -> np.ndarray:
    """
    Simple spectral-subtraction denoiser.

    1. STFT → magnitude + phase
    2. Estimate noise floor from first 10 frames
    3. Subtract 1.5× noise floor from magnitude
    4. ISTFT back to time domain (preserves original length)
    """
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    mag, phase = librosa.magphase(stft)
    noise_floor = np.mean(mag[:, :10], axis=1, keepdims=True)
    mag_clean = np.maximum(mag - 1.5 * noise_floor, 0.0)
    return librosa.istft(mag_clean * phase, hop_length=hop_length,
                         length=len(audio))


# ============================================================
# E) Fusion Logic  –  Hybrid-Guided Pitch Estimation
# ============================================================

class FusionLogic:
    """
    Gating logic that combines Path A (CREPE) and Path B (Voicing Mask).

    For each frame t:
        if Mask(t) == 1  AND  C(t) >= 0.15:
            keep f0(t)
        else:
            f0(t) = 0   (unvoiced)

    This eliminates "ghost notes" — false pitch detections caused by
    percussive artifacts or broadband noise.
    """

    def __init__(self, conf_threshold: float = CONF_THRESH):
        self.conf_threshold = conf_threshold

    def apply(self, est_time: np.ndarray, est_freq: np.ndarray,
              est_conf: np.ndarray, voicing_mask: np.ndarray,
              hop_length: int = HOP_LENGTH, sr: int = SR) -> np.ndarray:
        """
        Returns
        -------
        est_freq_filtered : np.ndarray – gated f0 contour
        """
        # Align mask to CREPE's time axis via interpolation
        mask_time = np.arange(len(voicing_mask)) * (hop_length / sr)
        mask_interp = np.interp(
            est_time, mask_time, voicing_mask.astype(float)
        ) > 0.5

        est_freq_filtered = est_freq.copy()
        for i in range(len(est_freq_filtered)):
            if (not mask_interp[i]) or est_conf[i] < self.conf_threshold:
                est_freq_filtered[i] = 0.0

        return est_freq_filtered


# ============================================================
# F) MIDI Generator
# ============================================================

class MIDIGenerator:
    """
    Converts gated pitch contours to MIDI note events.

    MIDI = 69 + 12 * log2(f0 / 440)

    Continuous pitch values are rounded to the nearest integer MIDI note.
    Note boundaries are detected by shifts between voiced (f0 > 0) and
    unvoiced (f0 = 0) regions.
    """

    @staticmethod
    def hz_to_midi(freq: np.ndarray) -> np.ndarray:
        freq = np.asarray(freq, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            midi = 69.0 + 12.0 * np.log2(freq / 440.0)
        midi[freq <= 0] = 0
        return np.round(midi).astype(int)

    @staticmethod
    def generate_midi(est_time: np.ndarray, est_freq: np.ndarray,
                      output_path: str, program: int = 0,
                      velocity: int = 80, instrument_name: str = ""):
        """Write a single-track MIDI file from a pitch contour."""
        import pretty_midi

        midi_notes = MIDIGenerator.hz_to_midi(est_freq)
        midi_data = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(
            program=program, name=instrument_name
        )

        current_note = None
        note_start = None

        for t, note in zip(est_time, midi_notes):
            if note <= 0:
                if current_note is not None:
                    instrument.notes.append(pretty_midi.Note(
                        velocity=velocity, pitch=current_note,
                        start=note_start, end=t
                    ))
                    current_note = None
                continue

            note = int(max(0, min(127, note)))
            if current_note != note:
                if current_note is not None:
                    instrument.notes.append(pretty_midi.Note(
                        velocity=velocity, pitch=current_note,
                        start=note_start, end=t
                    ))
                current_note = note
                note_start = t

        # Close last note
        if current_note is not None and len(est_time) > 0:
            instrument.notes.append(pretty_midi.Note(
                velocity=velocity, pitch=current_note,
                start=note_start, end=est_time[-1]
            ))

        midi_data.instruments.append(instrument)
        midi_data.write(output_path)
        n_notes = len(instrument.notes)
        print(f"  ✓ MIDI saved: {output_path}  ({n_notes} notes)")
        return output_path

    @staticmethod
    def combine_midi_files(stem_midi_paths: dict, output_path: str):
        """
        Merge per-stem MIDI files into a single multi-track MIDI.

        Parameters
        ----------
        stem_midi_paths : dict  – {stem_name: midi_file_path}
        output_path     : str   – path for the combined MIDI
        """
        import pretty_midi

        combined = pretty_midi.PrettyMIDI()

        for stem_name, midi_path in stem_midi_paths.items():
            if not os.path.isfile(midi_path):
                continue
            src = pretty_midi.PrettyMIDI(midi_path)
            for inst in src.instruments:
                inst.name = stem_name.capitalize()
                inst.program = STEM_PROGRAMS.get(stem_name, 0)
                combined.instruments.append(inst)

        combined.write(output_path)
        print(f"  ✓ Combined MIDI saved: {output_path}")
        return output_path


# ============================================================
# G) Process Single Stem  (iterative step)
# ============================================================

def process_single_stem(stem_audio_path: str, stem_name: str,
                        estimator: PitchEstimator,
                        detector: VoicingDetector,
                        fusion: FusionLogic,
                        sr: int = SR,
                        verbose: bool = True) -> dict:
    """
    Run the full dual-path pipeline on one stem.

    Steps:
      1. Load & resample to 16 kHz
      2. Denoise (spectral subtraction)
      3. Path A – CREPE pitch estimation
      4. Path B – Voicing mask (Mel + Chroma)
      5. Fusion – Gating logic
    """
    if verbose:
        print(f"\n  ── Processing stem: {stem_name.upper()} ──")

    # Step 1: Load audio
    audio, _ = librosa.load(stem_audio_path, sr=sr)
    if verbose:
        print(f"     Duration : {len(audio)/sr:.2f} s")

    # Step 2: Denoise
    denoised = denoise_audio(audio, hop_length=HOP_LENGTH)

    # Step 3: Path A – CREPE pitch estimation
    est_time, est_freq, est_conf = estimator.predict(denoised, sr)
    if verbose:
        voiced_raw = np.sum(est_freq > 0)
        print(f"     CREPE    : {voiced_raw}/{len(est_freq)} voiced frames (raw)")

    # Step 4: Path B – Voicing mask
    voicing_mask = detector.get_voicing_mask(denoised, sr)

    # Step 5: Fusion
    est_freq_filtered = fusion.apply(
        est_time, est_freq, est_conf, voicing_mask,
        hop_length=HOP_LENGTH, sr=sr
    )
    if verbose:
        voiced_fused = np.sum(est_freq_filtered > 0)
        print(f"     Fused    : {voiced_fused}/{len(est_freq_filtered)} voiced frames")

    return {
        "stem_name": stem_name,
        "audio": audio,
        "denoised": denoised,
        "voicing_mask": voicing_mask,
        "est_time": est_time,
        "est_freq_raw": est_freq,
        "est_freq_filtered": est_freq_filtered,
        "est_conf": est_conf,
    }


# ============================================================
# H) Run Full Pipeline  –  main entry point
# ============================================================

def run_full_pipeline(input_file: str,
                      output_dir: str = "output",
                      stems_to_process: list = None,
                      run_separation: bool = True,
                      combine_midi: bool = True,
                      verbose: bool = True) -> dict:
    """
    End-to-end pipeline:
      1. Demucs source separation  (4 stems)
      2. For each stem: denoise → CREPE → voicing mask → fusion → MIDI
      3. Combine all stems into a multi-track MIDI

    Parameters
    ----------
    input_file        : path to polyphonic audio WAV
    output_dir        : directory to save MIDI outputs
    stems_to_process  : list of stem names, or None for all 4
    run_separation    : set False if stems already exist
    combine_midi      : create a combined multi-track MIDI
    verbose           : print progress

    Returns
    -------
    dict with per-stem results and paths to generated MIDI files
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.splitext(os.path.basename(input_file))[0]

    if stems_to_process is None:
        stems_to_process = list(ALL_STEMS)

    # ---- Step 1: Source Separation (Demucs v4 htdemucs) ----
    print("=" * 60)
    print("STEP 1: Source Separation (Demucs v4 – htdemucs)")
    print("=" * 60)

    separator = SourceSeparator()
    if run_separation:
        stem_paths = separator.separate(input_file)
    else:
        stem_dir = os.path.join("separated_stems", "htdemucs", filename)
        if not SourceSeparator._stems_exist(stem_dir):
            print("  Stems not found – running separation anyway ...")
            stem_paths = separator.separate(input_file)
        else:
            stem_paths = SourceSeparator._build_stem_dict(stem_dir)
            print(f"  Using existing stems in {stem_dir}")

    # ---- Initialise shared modules ----
    estimator = PitchEstimator()
    detector = VoicingDetector()
    fusion = FusionLogic()
    midi_gen = MIDIGenerator()

    # ---- Step 2-6: Iterative stem-wise processing ----
    print("\n" + "=" * 60)
    print("STEPS 2-6: Iterative Stem Processing")
    print("=" * 60)

    all_results = {}
    stem_midi_paths = {}

    for stem_name in stems_to_process:
        stem_path = stem_paths.get(stem_name)
        if stem_path is None or not os.path.isfile(stem_path):
            print(f"\n  ⚠ Stem '{stem_name}' not found, skipping.")
            continue

        # Process stem through dual-path pipeline
        result = process_single_stem(
            stem_audio_path=stem_path,
            stem_name=stem_name,
            estimator=estimator,
            detector=detector,
            fusion=fusion,
            sr=SR,
            verbose=verbose,
        )

        # Generate per-stem MIDI
        midi_path = os.path.join(output_dir, f"{filename}_{stem_name}.mid")
        midi_gen.generate_midi(
            est_time=result["est_time"],
            est_freq=result["est_freq_filtered"],
            output_path=midi_path,
            program=STEM_PROGRAMS.get(stem_name, 0),
            instrument_name=stem_name.capitalize(),
        )
        stem_midi_paths[stem_name] = midi_path
        result["midi_path"] = midi_path
        all_results[stem_name] = result

    # ---- Step 7: Combine into multi-track MIDI ----
    combined_path = None
    if combine_midi and len(stem_midi_paths) > 0:
        print("\n" + "=" * 60)
        print("STEP 7: Combining into Multi-Track MIDI")
        print("=" * 60)
        combined_path = os.path.join(output_dir, f"{filename}_combined.mid")
        midi_gen.combine_midi_files(stem_midi_paths, combined_path)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Output directory : {os.path.abspath(output_dir)}")
    for sn, mp in stem_midi_paths.items():
        print(f"    {sn:>8} MIDI : {mp}")
    if combined_path:
        print(f"  {'combined':>10} MIDI : {combined_path}")

    return {
        "stem_results": all_results,
        "stem_midi_paths": stem_midi_paths,
        "combined_midi_path": combined_path,
    }


# ============================================================
# CLI entry point
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Hybrid CREPE Pipeline – Polyphonic Music Transcription"
    )
    parser.add_argument("audio_file", help="Path to input audio WAV file")
    parser.add_argument("--output_dir", default="output",
                        help="Directory for MIDI outputs (default: output)")
    parser.add_argument("--stems", default=None,
                        help="Comma-separated stems to process "
                             "(default: all = vocals,bass,drums,other)")
    parser.add_argument("--no_separation", action="store_true",
                        help="Skip Demucs if stems already exist")
    parser.add_argument("--no_combine", action="store_true",
                        help="Don't create combined multi-track MIDI")

    args = parser.parse_args()

    stems = None
    if args.stems:
        stems = [s.strip() for s in args.stems.split(",")]

    run_full_pipeline(
        input_file=args.audio_file,
        output_dir=args.output_dir,
        stems_to_process=stems,
        run_separation=not args.no_separation,
        combine_midi=not args.no_combine,
    )


if __name__ == "__main__":
    main()
