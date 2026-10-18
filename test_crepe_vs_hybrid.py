import os
import numpy as np
import librosa
import torch
import torchcrepe
from scipy.signal import butter, lfilter
from scipy.ndimage import median_filter


def pick_stem_path(input_file, stem="vocals"):
    """
    Return path to a separated stem if present, otherwise fall back to input file.
    """
    filename = os.path.splitext(os.path.basename(input_file))[0]
    model_out_dir = os.path.join("separated_stems", "htdemucs", filename)
    candidate = os.path.join(model_out_dir, f"{stem}.wav")
    if os.path.exists(candidate):
        return candidate
    print(
        f"Warning: separated {stem} not found at {candidate}. Using input file directly."
    )
    return input_file


class PitchEstimator:
    """
    Standalone CREPE-based pitch estimator (same settings as main pipeline).
    """

    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.hop_length = 160

    def predict(self, audio_array, sr):
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


def denoise_audio(audio, n_fft=1024, hop_length=160):
    """
    Simple spectral subtraction denoiser (same logic as experiment_pipeline, but local).
    """
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    mag, phase = librosa.magphase(stft)
    noise_floor = np.mean(mag[:, :10], axis=1, keepdims=True)
    mag_clean = np.maximum(mag - 1.5 * noise_floor, 0.0)
    return librosa.istft(mag_clean * phase, hop_length=hop_length, length=len(audio))


def get_hybrid_mask(audio, sr, hop_length=160):
    """
    Local copy of the Mel/Chroma-based activity mask used in the hybrid pipeline.
    """
    S = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=256, hop_length=hop_length
    )
    chroma = librosa.feature.chroma_stft(S=S, sr=sr)
    energy = np.mean(chroma, axis=0)
    thresh = np.percentile(energy, 10)
    mask = energy > thresh
    mask = median_filter(mask.astype(float), size=3) > 0.5
    return mask


def process_single_stem(
    stem_audio_path, stem_name, estimator, sr=16000, hop_length=160, verbose=True
):
    """
    Standalone copy of the single-stem hybrid pipeline (denoise + mask + gating).
    """
    if verbose:
        print(f"\n  Processing {stem_name.upper()} stem (hybrid)...")

    clean_audio, _ = librosa.load(stem_audio_path, sr=sr)
    cleaner_audio = denoise_audio(clean_audio, hop_length=hop_length)
    vocal_mask = get_hybrid_mask(cleaner_audio, sr, hop_length=hop_length)

    est_time, est_freq, est_conf = estimator.predict(cleaner_audio, sr)

    mask_time = np.arange(len(vocal_mask)) * (hop_length / sr)
    vocal_mask_interp = np.interp(est_time, mask_time, vocal_mask.astype(float)) > 0.5

    conf_thresh = 0.15
    est_freq_filtered = est_freq.copy()
    for i in range(len(est_freq_filtered)):
        if (not vocal_mask_interp[i]) or est_conf[i] < conf_thresh:
            est_freq_filtered[i] = 0.0

    return {
        "stem_name": stem_name,
        "audio": clean_audio,
        "denoised_audio": cleaner_audio,
        "vocal_mask": vocal_mask,
        "est_time": est_time,
        "est_freq": est_freq,
        "est_freq_filtered": est_freq_filtered,
        "est_conf": est_conf,
    }


class CrepeBaseline:
    """
    Local CREPE-only baseline (no denoise, no mask, no gating).
    """

    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.hop_length = 160

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

    def run(self, audio_path, sr=16000):
        audio, _ = librosa.load(audio_path, sr=sr)
        return_dict = {}
        t, f0, conf = self.predict(audio, sr)
        return {"time": t, "freq": f0, "conf": conf}


def compare_crepe_vs_hybrid(input_file, stem="vocals", sr=16000, conf_thresh=0.15):
    """
    Run CREPE baseline (no mask/gating) vs. Hybrid CREPE (denoise + mask + gating).
    Returns summary statistics for quick inspection.
    """
    # Resolve stem path
    stem_path = pick_stem_path(input_file, stem=stem)

    # Baseline CREPE (no denoise, no mask)
    baseline = CrepeBaseline()
    base_res = baseline.run(stem_path, sr=sr, output_midi=None, conf_threshold=0.0)

    # Hybrid CREPE (full pipeline on this stem only)
    estimator = PitchEstimator()
    hybrid_res = process_single_stem(
        stem_audio_path=stem_path,
        stem_name=stem,
        estimator=estimator,
        sr=sr,
        hop_length=160,
        verbose=False,
    )

    # Align to common time grid (use hybrid time axis)
    t_h = hybrid_res["est_time"]
    f_base_interp = np.interp(t_h, base_res["time"], base_res["freq"])

    # Masks
    voiced_base = f_base_interp > 0
    voiced_hybrid = hybrid_res["est_freq_filtered"] > 0
    voiced_both = voiced_base & voiced_hybrid

    # Stats
    stats = {
        "stem": stem,
        "voiced_frames_baseline": int(np.sum(voiced_base)),
        "voiced_frames_hybrid": int(np.sum(voiced_hybrid)),
        "total_frames": len(t_h),
        "median_conf_baseline": float(np.median(base_res["conf"])),
        "median_conf_hybrid": float(np.median(hybrid_res["est_conf"])),
        "mean_abs_diff_voiced": float(
            np.nan
            if np.sum(voiced_both) == 0
            else np.mean(
                np.abs(
                    f_base_interp[voiced_both]
                    - hybrid_res["est_freq_filtered"][voiced_both]
                )
            )
        ),
    }

    return stats


if __name__ == "__main__":
    # Example usage:
    # python test_crepe_vs_hybrid.py input_song.wav [stem]
    import sys

    stem = "vocals"

    if len(sys.argv) < 2:
        print("Usage: python test_crepe_vs_hybrid.py <audio_file> [stem]")
        sys.exit(1)
    if len(sys.argv) >= 3:
        stem = sys.argv[2]

    audio_file = sys.argv[1]
    summary = compare_crepe_vs_hybrid(audio_file, stem=stem)

    print("\n=== CREPE Baseline vs Hybrid (Harmonic-Gated) ===")
    print(f"Stem: {summary['stem']}")
    print(f"Total frames: {summary['total_frames']}")
    print(f"Voiced frames (baseline): {summary['voiced_frames_baseline']}")
    print(f"Voiced frames (hybrid)  : {summary['voiced_frames_hybrid']}")
    print(f"Median confidence (baseline): {summary['median_conf_baseline']:.3f}")
    print(f"Median confidence (hybrid)  : {summary['median_conf_hybrid']:.3f}")
    mad = summary["mean_abs_diff_voiced"]
    if np.isnan(mad):
        print("Mean abs diff (voiced overlap): n/a (no overlapping voiced frames)")
    else:
        print(f"Mean abs diff (voiced overlap): {mad:.2f} Hz")
