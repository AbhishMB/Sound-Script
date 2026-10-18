
# ===========================
# 0. SETUP (run first in Colab)
# ===========================
# (Colab) !pip install torchcrepe librosa mir_eval soundfile matplotlib pandas pretty_midi

import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import torch
import torchcrepe
import mir_eval
from scipy.signal import lfilter, butter
from scipy.ndimage import median_filter
import pandas as pd

# ============================================================
# 1. PATHS – EDIT THESE STRINGS FOR YOUR DRIVE
# ============================================================

# Example (after mounting drive):
# VOCALSET_TECH_ROOT = "/content/drive/MyDrive/data/VocalSet/data_by_technique"
# ANNOTATED_ROOT      = "/content/drive/MyDrive/data/Annotated VocalSet"

VOCALSET_TECH_ROOT = "/content/drive/MyDrive/data_by_technique"
ANNOTATED_ROOT     = "/content/drive/MyDrive/annotated_vocalset/Annotated VocalSet"

RAW_DIRS = [
    f"{ANNOTATED_ROOT}/raw 1/csv",
    f"{ANNOTATED_ROOT}/raw 2/csv",
    f"{ANNOTATED_ROOT}/raw 3/csv",
    f"{ANNOTATED_ROOT}/raw 4/csv",
]

SR = 16000
HOP_LENGTH = 160
SNR_LEVELS = [40, 30, 20, 10, 5, 0]


# ============================================================
# 2. CORE MODULES: PITCH, MASK, NOISE, METRICS
# ============================================================

class PitchEstimator:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.hop_length = HOP_LENGTH

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


class NoiseAugmenter:
    @staticmethod
    def add_noise(audio, noise_type, snr_db):
        N = len(audio)
        if noise_type == "white":
            noise = np.random.randn(N)
        elif noise_type == "pink":
            b, a = butter(1, 0.04, btype="low")
            noise = lfilter(b, a, np.random.randn(N))
        else:
            noise = np.cumsum(np.random.randn(N))

        sig_power = np.mean(audio ** 2) + 1e-12
        sig_db = 10 * np.log10(sig_power)
        noise_db = sig_db - snr_db
        noise_power = 10 ** (noise_db / 10)

        current_noise_power = np.mean(noise ** 2) + 1e-12
        noise = noise * np.sqrt(noise_power / current_noise_power)
        return audio + noise


def denoise_audio(audio, n_fft=1024, hop_length=HOP_LENGTH):
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    mag, phase = librosa.magphase(stft)
    noise_floor = np.mean(mag[:, :10], axis=1, keepdims=True)
    mag_clean = np.maximum(mag - 1.5 * noise_floor, 0.0)
    return librosa.istft(mag_clean * phase, hop_length=hop_length, length=len(audio))


def get_hybrid_mask(audio, sr, hop_length=HOP_LENGTH):
    S = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=256, hop_length=hop_length
    )
    chroma = librosa.feature.chroma_stft(S=S, sr=sr)
    energy = np.mean(chroma, axis=0)
    thresh = np.percentile(energy, 10)
    mask = energy > thresh
    mask = median_filter(mask.astype(float), size=3) > 0.5
    return mask


def calculate_rpa(ref_freq, est_freq):
    ref_voicing = (ref_freq > 0).astype(bool)
    est_voicing = (est_freq > 0).astype(bool)
    rpa = mir_eval.melody.raw_pitch_accuracy(
        ref_voicing, ref_freq, est_voicing, est_freq
    )
    return rpa * 100.0


def compute_voicing_stats(ref_freq, est_freq):
    ref_voiced = ref_freq > 0
    est_voiced = est_freq > 0

    tp = np.sum(ref_voiced & est_voiced)
    fp = np.sum(~ref_voiced & est_voiced)
    fn = np.sum(ref_voiced & ~est_voiced)
    tn = np.sum(~ref_voiced & ~est_voiced)

    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    fp_rate_unvoiced = fp / (fp + tn + 1e-8)
    return prec, rec, f1, fp_rate_unvoiced


def resample_f0_to_hop(times, freqs, audio_len_samples, sr=SR, hop_length=HOP_LENGTH):
    n_frames = int(np.ceil(audio_len_samples / hop_length))
    t_grid = np.arange(n_frames) * (hop_length / sr)
    freqs_clean = np.nan_to_num(freqs, nan=0.0)
    f_interp = np.interp(t_grid, times, freqs_clean, left=0.0, right=0.0)
    return t_grid, f_interp


def run_crepe_baseline(audio, sr, estimator):
    return estimator.predict(audio, sr)


def run_crepe_hybrid(audio, sr, estimator, conf_thresh=0.15):
    cleaner_audio = denoise_audio(audio, hop_length=HOP_LENGTH)
    mask = get_hybrid_mask(cleaner_audio, sr, hop_length=HOP_LENGTH)
    est_time, est_freq, est_conf = estimator.predict(cleaner_audio, sr)
    mask_time = np.arange(len(mask)) * (HOP_LENGTH / sr)
    mask_interp = np.interp(est_time, mask_time, mask.astype(float)) > 0.5
    est_freq_filtered = est_freq.copy()
    for i in range(len(est_freq_filtered)):
        if (not mask_interp[i]) or est_conf[i] < conf_thresh:
            est_freq_filtered[i] = 0.0
    return est_time, est_freq_filtered, est_conf


# ============================================================
# 3. BUILD INDEX: track_id -> raw CSV (Annotated-VocalSet)
# ============================================================

def build_annotated_index():
    import os
    index = {}
    print("Building annotation index...")
    print(f"Checking {len(RAW_DIRS)} directories:")
    for raw_dir in RAW_DIRS:
        print(f"  - {raw_dir}")
        if not os.path.exists(raw_dir):
            print(f"    ❌ Directory does not exist!")
            continue
        csv_files = glob.glob(raw_dir + "/*.csv")
        print(f"    Found {len(csv_files)} CSV files")
        if len(csv_files) == 0:
            # Try to list what's actually in the directory
            try:
                all_files = os.listdir(raw_dir)
                print(f"    Directory contains: {all_files[:10]}..." if len(all_files) > 10 else f"    Directory contains: {all_files}")
            except Exception as e:
                print(f"    Error listing directory: {e}")
        for path in csv_files:
            base = path.split("/")[-1].rsplit(".", 1)[0]
            index[base] = path
    print(f"Total annotations indexed: {len(index)}")
    if len(index) > 0:
        print(f"Sample track IDs: {list(index.keys())[:5]}")
    return index

ANNOTATED_INDEX = build_annotated_index()

def load_raw_csv_f0(path):
    data = np.loadtxt(path, delimiter=",")
    times = data[:, 0]
    freqs = data[:, 1]
    return times, freqs


def iter_vocalset_by_technique_with_annotations():
    """
    VOCALSET_TECH_ROOT/data_by_technique/<technique>/*.wav
    Only yield files that exist in Annotated-VocalSet raw CSVs.
    """
    for tech_name in sorted([d for d in glob.glob(VOCALSET_TECH_ROOT + "/*") if "." not in d]):
        technique = tech_name.split("/")[-1]
        for wav_path in glob.glob(tech_name + "/*.wav"):
            track_id = wav_path.split("/")[-1].rsplit(".", 1)[0]
            csv_path = ANNOTATED_INDEX.get(track_id, None)
            if csv_path is None:
                # uncomment for debugging
                # print(f"[WARN] no annotation for {track_id}")
                continue
            times, freqs = load_raw_csv_f0(csv_path)
            yield track_id, technique, wav_path, times, freqs


# ============================================================
# 4. EVALUATION (NOISE-ROBUSTNESS COMPARISON)
# ============================================================

def evaluate_noise_robustness(
    snr_levels=SNR_LEVELS, noise_type="white", max_tracks=None
):
    est = PitchEstimator()
    results = []
    count = 0
    
    # Check if we can find any tracks
    import os
    print(f"\nChecking for tracks...")
    print(f"VOCALSET_TECH_ROOT: {VOCALSET_TECH_ROOT}")
    print(f"  Exists: {os.path.exists(VOCALSET_TECH_ROOT)}")
    if os.path.exists(VOCALSET_TECH_ROOT):
        tech_dirs = [d for d in glob.glob(VOCALSET_TECH_ROOT + "/*") if os.path.isdir(d)]
        print(f"  Found {len(tech_dirs)} technique directories")
        if len(tech_dirs) > 0:
            print(f"  Sample: {[os.path.basename(d) for d in tech_dirs[:5]]}")
            # Count WAV files
            total_wavs = sum(len(glob.glob(d + "/*.wav")) for d in tech_dirs)
            print(f"  Total WAV files: {total_wavs}")
    
    print(f"ANNOTATED_ROOT: {ANNOTATED_ROOT}")
    print(f"  Exists: {os.path.exists(ANNOTATED_ROOT)}")
    print(f"Number of indexed annotations: {len(ANNOTATED_INDEX)}")
    
    track_iterator = iter_vocalset_by_technique_with_annotations()
    first_track = next(track_iterator, None)
    if first_track is None:
        print("\nWARNING: No tracks found with matching annotations!")
        print("\nTroubleshooting:")
        print("1. Check that VOCALSET_TECH_ROOT contains technique subdirectories with .wav files")
        print("2. Check that ANNOTATED_ROOT contains 'raw 1/csv', 'raw 2/csv', etc. with .csv files")
        print("3. Ensure track IDs (filenames without extension) match between .wav and .csv files")
        return pd.DataFrame(results)
    
    # Process first track
    track_id, technique, audio_path, f0_times, f0_freqs = first_track
    count += 1
    print(f"\n=== {track_id} | technique: {technique} ===")
    print("Audio:", audio_path)

    audio, _ = librosa.load(audio_path, sr=SR)
    n_samples = len(audio)
    _, ref_f = resample_f0_to_hop(f0_times, f0_freqs, n_samples, sr=SR, hop_length=HOP_LENGTH)

    # clean baseline
    _, b_freq, _ = run_crepe_baseline(audio, SR, est)
    L = min(len(ref_f), len(b_freq))
    ref_clean = ref_f[:L]
    b_freq = b_freq[:L]
    rpa_b = calculate_rpa(ref_clean, b_freq)
    prec_b, rec_b, f1_b, fp_b = compute_voicing_stats(ref_clean, b_freq)

    results.append({
        "track_id": track_id,
        "technique": technique,
        "snr_db": None,
        "system": "baseline",
        "rpa": rpa_b,
        "voicing_precision": prec_b,
        "voicing_recall": rec_b,
        "voicing_f1": f1_b,
        "fp_rate_unvoiced": fp_b,
    })

    # clean hybrid
    _, h_freq, _ = run_crepe_hybrid(audio, SR, est)
    L = min(len(ref_f), len(h_freq))
    ref_clean_h = ref_f[:L]
    h_freq = h_freq[:L]
    rpa_h = calculate_rpa(ref_clean_h, h_freq)
    prec_h, rec_h, f1_h, fp_h = compute_voicing_stats(ref_clean_h, h_freq)

    results.append({
        "track_id": track_id,
        "technique": technique,
        "snr_db": None,
        "system": "hybrid",
        "rpa": rpa_h,
        "voicing_precision": prec_h,
        "voicing_recall": rec_h,
        "voicing_f1": f1_h,
        "fp_rate_unvoiced": fp_h,
    })

    print(f"  Clean | RPA baseline {rpa_b:.2f} | hybrid {rpa_h:.2f}")

    # noise robustness
    for snr in snr_levels:
        noisy = NoiseAugmenter.add_noise(audio, noise_type, snr_db=snr)

        # baseline noisy
        _, b_freq_n, _ = run_crepe_baseline(noisy, SR, est)
        L = min(len(ref_f), len(b_freq_n))
        ref_n = ref_f[:L]
        b_freq_n = b_freq_n[:L]
        rpa_bn = calculate_rpa(ref_n, b_freq_n)
        prec_bn, rec_bn, f1_bn, fp_bn = compute_voicing_stats(ref_n, b_freq_n)

        results.append({
            "track_id": track_id,
            "technique": technique,
            "snr_db": snr,
            "system": "baseline",
            "rpa": rpa_bn,
            "voicing_precision": prec_bn,
            "voicing_recall": rec_bn,
            "voicing_f1": f1_bn,
            "fp_rate_unvoiced": fp_bn,
        })

        # hybrid noisy
        _, h_freq_n, _ = run_crepe_hybrid(noisy, SR, est)
        L = min(len(ref_f), len(h_freq_n))
        ref_hn = ref_f[:L]
        h_freq_n = h_freq_n[:L]
        rpa_hn = calculate_rpa(ref_hn, h_freq_n)
        prec_hn, rec_hn, f1_hn, fp_hn = compute_voicing_stats(ref_hn, h_freq_n)

        results.append({
            "track_id": track_id,
            "technique": technique,
            "snr_db": snr,
            "system": "hybrid",
            "rpa": rpa_hn,
            "voicing_precision": prec_hn,
            "voicing_recall": rec_hn,
            "voicing_f1": f1_hn,
            "fp_rate_unvoiced": fp_hn,
        })

        print(f"  SNR {snr:>2} dB | RPA baseline {rpa_bn:.2f} | hybrid {rpa_hn:.2f}")
    
    # Process remaining tracks
    for track_id, technique, audio_path, f0_times, f0_freqs in track_iterator:
        if max_tracks is not None and count >= max_tracks:
            break
        count += 1

        print(f"\n=== {track_id} | technique: {technique} ===")
        print("Audio:", audio_path)

        audio, _ = librosa.load(audio_path, sr=SR)
        n_samples = len(audio)
        _, ref_f = resample_f0_to_hop(f0_times, f0_freqs, n_samples, sr=SR, hop_length=HOP_LENGTH)

        # clean baseline
        _, b_freq, _ = run_crepe_baseline(audio, SR, est)
        L = min(len(ref_f), len(b_freq))
        ref_clean = ref_f[:L]
        b_freq = b_freq[:L]
        rpa_b = calculate_rpa(ref_clean, b_freq)
        prec_b, rec_b, f1_b, fp_b = compute_voicing_stats(ref_clean, b_freq)

        results.append({
            "track_id": track_id,
            "technique": technique,
            "snr_db": None,
            "system": "baseline",
            "rpa": rpa_b,
            "voicing_precision": prec_b,
            "voicing_recall": rec_b,
            "voicing_f1": f1_b,
            "fp_rate_unvoiced": fp_b,
        })

        # clean hybrid
        _, h_freq, _ = run_crepe_hybrid(audio, SR, est)
        L = min(len(ref_f), len(h_freq))
        ref_clean_h = ref_f[:L]
        h_freq = h_freq[:L]
        rpa_h = calculate_rpa(ref_clean_h, h_freq)
        prec_h, rec_h, f1_h, fp_h = compute_voicing_stats(ref_clean_h, h_freq)

        results.append({
            "track_id": track_id,
            "technique": technique,
            "snr_db": None,
            "system": "hybrid",
            "rpa": rpa_h,
            "voicing_precision": prec_h,
            "voicing_recall": rec_h,
            "voicing_f1": f1_h,
            "fp_rate_unvoiced": fp_h,
        })

        print(f"  Clean | RPA baseline {rpa_b:.2f} | hybrid {rpa_h:.2f}")

        # noise robustness
        for snr in snr_levels:
            noisy = NoiseAugmenter.add_noise(audio, noise_type, snr_db=snr)

            # baseline noisy
            _, b_freq_n, _ = run_crepe_baseline(noisy, SR, est)
            L = min(len(ref_f), len(b_freq_n))
            ref_n = ref_f[:L]
            b_freq_n = b_freq_n[:L]
            rpa_bn = calculate_rpa(ref_n, b_freq_n)
            prec_bn, rec_bn, f1_bn, fp_bn = compute_voicing_stats(ref_n, b_freq_n)

            results.append({
                "track_id": track_id,
                "technique": technique,
                "snr_db": snr,
                "system": "baseline",
                "rpa": rpa_bn,
                "voicing_precision": prec_bn,
                "voicing_recall": rec_bn,
                "voicing_f1": f1_bn,
                "fp_rate_unvoiced": fp_bn,
            })

            # hybrid noisy
            _, h_freq_n, _ = run_crepe_hybrid(noisy, SR, est)
            L = min(len(ref_f), len(h_freq_n))
            ref_hn = ref_f[:L]
            h_freq_n = h_freq_n[:L]
            rpa_hn = calculate_rpa(ref_hn, h_freq_n)
            prec_hn, rec_hn, f1_hn, fp_hn = compute_voicing_stats(ref_hn, h_freq_n)

            results.append({
                "track_id": track_id,
                "technique": technique,
                "snr_db": snr,
                "system": "hybrid",
                "rpa": rpa_hn,
                "voicing_precision": prec_hn,
                "voicing_recall": rec_hn,
                "voicing_f1": f1_hn,
                "fp_rate_unvoiced": fp_hn,
            })

            print(f"  SNR {snr:>2} dB | RPA baseline {rpa_bn:.2f} | hybrid {rpa_hn:.2f}")

    print(f"\nTotal tracks processed: {count}")
    print(f"Total results: {len(results)}")
    
    if len(results) == 0:
        print("WARNING: No results generated!")
        return pd.DataFrame()
    
    return pd.DataFrame(results)


def plot_rpa_vs_snr(df):
    # Check if DataFrame is empty
    if df.empty:
        print("Error: DataFrame is empty. No data to plot.")
        return
    
    # Check if required columns exist
    if "snr_db" not in df.columns:
        print(f"Error: 'snr_db' column not found. Available columns: {df.columns.tolist()}")
        return
    
    if "rpa" not in df.columns:
        print(f"Error: 'rpa' column not found. Available columns: {df.columns.tolist()}")
        return
    
    # Filter for SNR data
    sub = df[df["snr_db"].notnull()]
    if sub.empty:
        print("No SNR data to plot (all snr_db values are null).")
        print(f"DataFrame shape: {df.shape}")
        print(f"DataFrame columns: {df.columns.tolist()}")
        if not df.empty:
            print(f"Sample data:\n{df.head()}")
        return
    
    plt.figure(figsize=(7,5))
    for system in ["baseline", "hybrid"]:
        d = sub[sub["system"] == system]
        if d.empty:
            print(f"Warning: No data for system '{system}'")
            continue
        means = d.groupby("snr_db")["rpa"].mean().sort_index(ascending=False)
        if len(means) > 0:
            plt.plot(means.index, means.values, "o-", label=system)
    plt.gca().invert_xaxis()
    plt.xlabel("SNR (dB)")
    plt.ylabel("Raw Pitch Accuracy (%)")
    plt.title("RPA vs SNR – all techniques")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# 5. DIAGNOSTIC FUNCTION (run this first to check paths)
# ============================================================

def diagnose_paths():
    """
    Run this function to check if your paths are set up correctly.
    """
    import os
    print("=" * 60)
    print("PATH DIAGNOSTICS")
    print("=" * 60)
    
    print(f"\n1. VOCALSET_TECH_ROOT: {VOCALSET_TECH_ROOT}")
    if os.path.exists(VOCALSET_TECH_ROOT):
        print("   ✓ Exists")
        tech_dirs = [d for d in glob.glob(VOCALSET_TECH_ROOT + "/*") if os.path.isdir(d)]
        print(f"   Found {len(tech_dirs)} technique directories:")
        for tech_dir in tech_dirs[:5]:  # Show first 5
            tech_name = os.path.basename(tech_dir)
            wav_count = len(glob.glob(tech_dir + "/*.wav"))
            print(f"     - {tech_name}: {wav_count} WAV files")
        if len(tech_dirs) > 5:
            print(f"     ... and {len(tech_dirs) - 5} more")
    else:
        print("   ✗ Does NOT exist!")
        print("   Please check the path and update VOCALSET_TECH_ROOT")
    
    print(f"\n2. ANNOTATED_ROOT: {ANNOTATED_ROOT}")
    if os.path.exists(ANNOTATED_ROOT):
        print("   ✓ Exists")
    else:
        print("   ✗ Does NOT exist!")
        print("   Please check the path and update ANNOTATED_ROOT")
    
    print(f"\n3. Checking RAW_DIRS:")
    total_csvs = 0
    for raw_dir in RAW_DIRS:
        print(f"   {raw_dir}")
        if os.path.exists(raw_dir):
            csv_files = glob.glob(raw_dir + "/*.csv")
            print(f"     ✓ Exists, found {len(csv_files)} CSV files")
            total_csvs += len(csv_files)
            if len(csv_files) > 0:
                print(f"     Sample files: {[os.path.basename(f) for f in csv_files[:3]]}")
        else:
            print(f"     ✗ Does NOT exist!")
    
    print(f"\n4. Annotation Index:")
    print(f"   Total CSV files indexed: {len(ANNOTATED_INDEX)}")
    if len(ANNOTATED_INDEX) > 0:
        print(f"   Sample track IDs: {list(ANNOTATED_INDEX.keys())[:5]}")
    else:
        print("   ⚠️  No annotations found! Check your RAW_DIRS paths.")
    
    print(f"\n5. Matching tracks:")
    matching_count = 0
    if os.path.exists(VOCALSET_TECH_ROOT):
        for tech_dir in glob.glob(VOCALSET_TECH_ROOT + "/*"):
            if not os.path.isdir(tech_dir):
                continue
            for wav_path in glob.glob(tech_dir + "/*.wav"):
                track_id = os.path.basename(wav_path).rsplit(".", 1)[0]
                if track_id in ANNOTATED_INDEX:
                    matching_count += 1
                    if matching_count <= 3:
                        print(f"   ✓ {track_id} (has both WAV and CSV)")
    print(f"   Total matching tracks: {matching_count}")
    
    print("\n" + "=" * 60)
    if len(ANNOTATED_INDEX) == 0:
        print("⚠️  ISSUE: No CSV files found. Check your ANNOTATED_ROOT path.")
    elif matching_count == 0:
        print("⚠️  ISSUE: No matching tracks found. Check that track IDs match.")
    else:
        print("✓ Paths look good! You should be able to run the evaluation.")
    print("=" * 60)


# ============================================================
# 6. SINGLE AUDIO FILE PROCESSING
# ============================================================

def process_single_audio(audio_path, output_dir="output", save_midi=True, save_plots=True):
    """
    Process a single audio file with both CREPE baseline and hybrid methods.
    
    Args:
        audio_path: Path to input WAV file
        output_dir: Directory to save outputs
        save_midi: Whether to save MIDI files
        save_plots: Whether to save visualization plots
    
    Returns:
        Dictionary with results from both methods
    """
    import os
    from pretty_midi import PrettyMIDI, Instrument, Note
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load audio
    print(f"Loading audio: {audio_path}")
    audio, sr = librosa.load(audio_path, sr=SR)
    print(f"  Duration: {len(audio)/sr:.2f} seconds")
    print(f"  Sample rate: {sr} Hz")
    
    # Initialize estimator
    estimator = PitchEstimator()
    
    # ========== BASELINE CREPE ==========
    print("\n" + "="*60)
    print("PROCESSING WITH CREPE BASELINE")
    print("="*60)
    est_time_base, est_freq_base, est_conf_base = run_crepe_baseline(audio, sr, estimator)
    
    n_voiced_base = np.sum(est_freq_base > 0)
    print(f"Voiced frames: {n_voiced_base}/{len(est_freq_base)} ({100*n_voiced_base/len(est_freq_base):.1f}%)")
    if n_voiced_base > 0:
        print(f"Frequency range: {est_freq_base[est_freq_base>0].min():.1f} - {est_freq_base[est_freq_base>0].max():.1f} Hz")
        print(f"Median confidence: {np.median(est_conf_base[est_freq_base>0]):.3f}")
    
    # ========== HYBRID CREPE ==========
    print("\n" + "="*60)
    print("PROCESSING WITH CREPE HYBRID (Gated)")
    print("="*60)
    est_time_hybrid, est_freq_hybrid, est_conf_hybrid = run_crepe_hybrid(audio, sr, estimator)
    
    n_voiced_hybrid = np.sum(est_freq_hybrid > 0)
    print(f"Voiced frames: {n_voiced_hybrid}/{len(est_freq_hybrid)} ({100*n_voiced_hybrid/len(est_freq_hybrid):.1f}%)")
    if n_voiced_hybrid > 0:
        print(f"Frequency range: {est_freq_hybrid[est_freq_hybrid>0].min():.1f} - {est_freq_hybrid[est_freq_hybrid>0].max():.1f} Hz")
        print(f"Median confidence: {np.median(est_conf_hybrid[est_freq_hybrid>0]):.3f}")
    
    # ========== COMPARISON ==========
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    print(f"Baseline:  {n_voiced_base:>5} voiced frames ({100*n_voiced_base/len(est_freq_base):>5.1f}%)")
    print(f"Hybrid:    {n_voiced_hybrid:>5} voiced frames ({100*n_voiced_hybrid/len(est_freq_hybrid):>5.1f}%)")
    print(f"Difference: {n_voiced_base - n_voiced_hybrid:>5} frames ({100*(n_voiced_base - n_voiced_hybrid)/len(est_freq_base):>5.1f}%)")
    
    # Get base name for output files
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    
    # ========== SAVE MIDI FILES ==========
    if save_midi:
        try:
            
            # Baseline MIDI
            midi_base = PrettyMIDI()
            inst_base = Instrument(program=0)
            midi_base.instruments.append(inst_base)
            
            current_note = None
            note_start = None
            for t, f in zip(est_time_base, est_freq_base):
                if f <= 0:
                    if current_note is not None:
                        inst_base.notes.append(Note(velocity=80, pitch=current_note, start=note_start, end=t))
                        current_note = None
                    continue
                
                midi_note = int(69 + 12 * np.log2(f / 440.0))
                midi_note = max(0, min(127, midi_note))
                
                if current_note != midi_note:
                    if current_note is not None:
                        inst_base.notes.append(Note(velocity=80, pitch=current_note, start=note_start, end=t))
                    current_note = midi_note
                    note_start = t
            
            if current_note is not None:
                inst_base.notes.append(Note(velocity=80, pitch=current_note, start=note_start, end=est_time_base[-1]))
            
            midi_base_path = os.path.join(output_dir, f"{base_name}_baseline.mid")
            midi_base.write(midi_base_path)
            print(f"\n✓ Saved baseline MIDI: {midi_base_path}")
            
            # Hybrid MIDI
            midi_hybrid = PrettyMIDI()
            inst_hybrid = Instrument(program=0)
            midi_hybrid.instruments.append(inst_hybrid)
            
            current_note = None
            note_start = None
            for t, f in zip(est_time_hybrid, est_freq_hybrid):
                if f <= 0:
                    if current_note is not None:
                        inst_hybrid.notes.append(Note(velocity=80, pitch=current_note, start=note_start, end=t))
                        current_note = None
                    continue
                
                midi_note = int(69 + 12 * np.log2(f / 440.0))
                midi_note = max(0, min(127, midi_note))
                
                if current_note != midi_note:
                    if current_note is not None:
                        inst_hybrid.notes.append(Note(velocity=80, pitch=current_note, start=note_start, end=t))
                    current_note = midi_note
                    note_start = t
            
            if current_note is not None:
                inst_hybrid.notes.append(Note(velocity=80, pitch=current_note, start=note_start, end=est_time_hybrid[-1]))
            
            midi_hybrid_path = os.path.join(output_dir, f"{base_name}_hybrid.mid")
            midi_hybrid.write(midi_hybrid_path)
            print(f"✓ Saved hybrid MIDI: {midi_hybrid_path}")
            
        except ImportError:
            print("\n⚠️  pretty_midi not installed. Skipping MIDI export.")
            print("   Install with: pip install pretty_midi")
        except Exception as e:
            print(f"\n⚠️  Error saving MIDI: {e}")
    
    # ========== SAVE PLOTS ==========
    if save_plots:
        try:
            # ===== PLOT 1: Time Series Comparison =====
            fig1, axes1 = plt.subplots(3, 1, figsize=(12, 10))
            
            # Plot 1.1: Frequency contours
            axes1[0].plot(est_time_base, est_freq_base, 'b-', alpha=0.7, label='Baseline CREPE', linewidth=0.5)
            axes1[0].plot(est_time_hybrid, est_freq_hybrid, 'r-', alpha=0.7, label='Hybrid CREPE', linewidth=0.5)
            axes1[0].set_xlabel('Time (s)')
            axes1[0].set_ylabel('Frequency (Hz)')
            axes1[0].set_title('Pitch Estimation Comparison (Time Series)')
            axes1[0].legend()
            axes1[0].grid(True, alpha=0.3)
            
            # Plot 1.2: Confidence scores
            axes1[1].plot(est_time_base, est_conf_base, 'b-', alpha=0.7, label='Baseline CREPE', linewidth=0.5)
            axes1[1].plot(est_time_hybrid, est_conf_hybrid, 'r-', alpha=0.7, label='Hybrid CREPE', linewidth=0.5)
            axes1[1].set_xlabel('Time (s)')
            axes1[1].set_ylabel('Confidence')
            axes1[1].set_title('Confidence Scores (Time Series)')
            axes1[1].legend()
            axes1[1].grid(True, alpha=0.3)
            axes1[1].set_ylim([0, 1])
            
            # Plot 1.3: Voicing comparison
            voiced_base = (est_freq_base > 0).astype(float)
            voiced_hybrid = (est_freq_hybrid > 0).astype(float)
            axes1[2].plot(est_time_base, voiced_base, 'b-', alpha=0.7, label='Baseline (voiced)', linewidth=0.5)
            axes1[2].plot(est_time_hybrid, voiced_hybrid, 'r-', alpha=0.7, label='Hybrid (voiced)', linewidth=0.5)
            axes1[2].set_xlabel('Time (s)')
            axes1[2].set_ylabel('Voiced (1) / Unvoiced (0)')
            axes1[2].set_title('Voicing Detection (Time Series)')
            axes1[2].legend()
            axes1[2].grid(True, alpha=0.3)
            axes1[2].set_ylim([-0.1, 1.1])
            
            plt.tight_layout()
            plot_path1 = os.path.join(output_dir, f"{base_name}_comparison_timeseries.png")
            plt.savefig(plot_path1, dpi=150, bbox_inches='tight')
            print(f"✓ Saved time series plot: {plot_path1}")
            plt.close()
            
            # ===== PLOT 2: Histogram Comparison =====
            fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
            
            # Histogram 2.1: Frequency Distribution (voiced frames only)
            freq_base_voiced = est_freq_base[est_freq_base > 0]
            freq_hybrid_voiced = est_freq_hybrid[est_freq_hybrid > 0]
            
            if len(freq_base_voiced) > 0 or len(freq_hybrid_voiced) > 0:
                max_freq = max(
                    np.max(freq_base_voiced) if len(freq_base_voiced) > 0 else 0,
                    np.max(freq_hybrid_voiced) if len(freq_hybrid_voiced) > 0 else 0
                )
                bins = np.linspace(0, min(max_freq, 2000), 50)
                
                axes2[0, 0].hist(freq_base_voiced, bins=bins, alpha=0.6, color='blue', 
                                label=f'Baseline (n={len(freq_base_voiced)})', density=True)
                axes2[0, 0].hist(freq_hybrid_voiced, bins=bins, alpha=0.6, color='red', 
                                label=f'Hybrid (n={len(freq_hybrid_voiced)})', density=True)
                axes2[0, 0].set_xlabel('Frequency (Hz)')
                axes2[0, 0].set_ylabel('Density')
                axes2[0, 0].set_title('Frequency Distribution (Voiced Frames Only)')
                axes2[0, 0].legend()
                axes2[0, 0].grid(True, alpha=0.3)
            
            # Histogram 2.2: Confidence Distribution (all frames)
            bins_conf = np.linspace(0, 1, 50)
            axes2[0, 1].hist(est_conf_base, bins=bins_conf, alpha=0.6, color='blue', 
                             label=f'Baseline (mean={np.mean(est_conf_base):.3f})', density=True)
            axes2[0, 1].hist(est_conf_hybrid, bins=bins_conf, alpha=0.6, color='red', 
                            label=f'Hybrid (mean={np.mean(est_conf_hybrid):.3f})', density=True)
            axes2[0, 1].set_xlabel('Confidence Score')
            axes2[0, 1].set_ylabel('Density')
            axes2[0, 1].set_title('Confidence Score Distribution (All Frames)')
            axes2[0, 1].legend()
            axes2[0, 1].grid(True, alpha=0.3)
            
            # Histogram 2.3: Confidence Distribution (voiced frames only)
            conf_base_voiced = est_conf_base[est_freq_base > 0]
            conf_hybrid_voiced = est_conf_hybrid[est_freq_hybrid > 0]
            
            if len(conf_base_voiced) > 0 or len(conf_hybrid_voiced) > 0:
                axes2[1, 0].hist(conf_base_voiced, bins=bins_conf, alpha=0.6, color='blue', 
                                label=f'Baseline (mean={np.mean(conf_base_voiced):.3f})', density=True)
                axes2[1, 0].hist(conf_hybrid_voiced, bins=bins_conf, alpha=0.6, color='red', 
                                label=f'Hybrid (mean={np.mean(conf_hybrid_voiced):.3f})', density=True)
                axes2[1, 0].set_xlabel('Confidence Score')
                axes2[1, 0].set_ylabel('Density')
                axes2[1, 0].set_title('Confidence Score Distribution (Voiced Frames Only)')
                axes2[1, 0].legend()
                axes2[1, 0].grid(True, alpha=0.3)
            
            # Bar Chart 2.4: Voicing Statistics
            categories = ['Voiced\nFrames', 'Unvoiced\nFrames', 'Total\nFrames']
            baseline_counts = [
                n_voiced_base,
                len(est_freq_base) - n_voiced_base,
                len(est_freq_base)
            ]
            hybrid_counts = [
                n_voiced_hybrid,
                len(est_freq_hybrid) - n_voiced_hybrid,
                len(est_freq_hybrid)
            ]
            
            x = np.arange(len(categories))
            width = 0.35
            
            axes2[1, 1].bar(x - width/2, baseline_counts, width, label='Baseline', color='blue', alpha=0.7)
            axes2[1, 1].bar(x + width/2, hybrid_counts, width, label='Hybrid', color='red', alpha=0.7)
            axes2[1, 1].set_xlabel('Frame Type')
            axes2[1, 1].set_ylabel('Count')
            axes2[1, 1].set_title('Voicing Statistics Comparison')
            axes2[1, 1].set_xticks(x)
            axes2[1, 1].set_xticklabels(categories)
            axes2[1, 1].legend()
            axes2[1, 1].grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for i, (b, h) in enumerate(zip(baseline_counts, hybrid_counts)):
                axes2[1, 1].text(i - width/2, b, str(b), ha='center', va='bottom', fontsize=9)
                axes2[1, 1].text(i + width/2, h, str(h), ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            plot_path2 = os.path.join(output_dir, f"{base_name}_comparison_histograms.png")
            plt.savefig(plot_path2, dpi=150, bbox_inches='tight')
            print(f"✓ Saved histogram comparison plot: {plot_path2}")
            plt.close()
            
            # ===== PLOT 3: Summary Statistics Table =====
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            ax3.axis('tight')
            ax3.axis('off')
            
            # Calculate statistics
            stats_data = [
                ['Metric', 'Baseline CREPE', 'Hybrid CREPE', 'Difference'],
                ['Total Frames', len(est_freq_base), len(est_freq_hybrid), len(est_freq_base) - len(est_freq_hybrid)],
                ['Voiced Frames', n_voiced_base, n_voiced_hybrid, n_voiced_base - n_voiced_hybrid],
                ['% Voiced', f'{100*n_voiced_base/len(est_freq_base):.2f}%', 
                 f'{100*n_voiced_hybrid/len(est_freq_hybrid):.2f}%',
                 f'{100*(n_voiced_base - n_voiced_hybrid)/len(est_freq_base):.2f}%'],
                ['Mean Confidence (All)', f'{np.mean(est_conf_base):.3f}', 
                 f'{np.mean(est_conf_hybrid):.3f}',
                 f'{np.mean(est_conf_base) - np.mean(est_conf_hybrid):.3f}'],
            ]
            
            if len(freq_base_voiced) > 0:
                stats_data.append(['Mean Freq (Hz)', f'{np.mean(freq_base_voiced):.2f}',
                                  f'{np.mean(freq_hybrid_voiced) if len(freq_hybrid_voiced) > 0 else 0:.2f}',
                                  f'{np.mean(freq_base_voiced) - (np.mean(freq_hybrid_voiced) if len(freq_hybrid_voiced) > 0 else 0):.2f}'])
                stats_data.append(['Median Freq (Hz)', f'{np.median(freq_base_voiced):.2f}',
                                  f'{np.median(freq_hybrid_voiced) if len(freq_hybrid_voiced) > 0 else 0:.2f}',
                                  f'{np.median(freq_base_voiced) - (np.median(freq_hybrid_voiced) if len(freq_hybrid_voiced) > 0 else 0):.2f}'])
            
            if len(conf_base_voiced) > 0:
                stats_data.append(['Mean Conf (Voiced)', f'{np.mean(conf_base_voiced):.3f}',
                                  f'{np.mean(conf_hybrid_voiced) if len(conf_hybrid_voiced) > 0 else 0:.3f}',
                                  f'{np.mean(conf_base_voiced) - (np.mean(conf_hybrid_voiced) if len(conf_hybrid_voiced) > 0 else 0):.3f}'])
            
            table = ax3.table(cellText=stats_data[1:], colLabels=stats_data[0],
                             cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
            
            # Style the header row
            for i in range(len(stats_data[0])):
                table[(0, i)].set_facecolor('#40466e')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            plt.title('Statistical Comparison Summary', fontsize=14, fontweight='bold', pad=20)
            plot_path3 = os.path.join(output_dir, f"{base_name}_comparison_stats.png")
            plt.savefig(plot_path3, dpi=150, bbox_inches='tight')
            print(f"✓ Saved statistics table: {plot_path3}")
            plt.close()
            
        except Exception as e:
            print(f"\n⚠️  Error saving plots: {e}")
            import traceback
            traceback.print_exc()
    
    # ========== RETURN RESULTS ==========
    results = {
        "audio_path": audio_path,
        "duration": len(audio) / sr,
        "baseline": {
            "time": est_time_base,
            "freq": est_freq_base,
            "conf": est_conf_base,
            "n_voiced": n_voiced_base,
            "pct_voiced": 100 * n_voiced_base / len(est_freq_base),
        },
        "hybrid": {
            "time": est_time_hybrid,
            "freq": est_freq_hybrid,
            "conf": est_conf_hybrid,
            "n_voiced": n_voiced_hybrid,
            "pct_voiced": 100 * n_voiced_hybrid / len(est_freq_hybrid),
        },
    }
    
    return results


# ============================================================
# 7. RUN - SINGLE AUDIO PROCESSING
# ============================================================

# ===== EDIT THIS: Path to your input WAV file =====
INPUT_AUDIO = "input.wav"  # Change this to your audio file path

# Process the audio file
if os.path.exists(INPUT_AUDIO):
    results = process_single_audio(
        audio_path=INPUT_AUDIO,
        output_dir="output",
        save_midi=True,
        save_plots=True
    )
    print("\n" + "="*60)
    print("PROCESSING COMPLETE!")
    print("="*60)
else:
    print(f"ERROR: Audio file not found: {INPUT_AUDIO}")
    print("Please update INPUT_AUDIO with the path to your WAV file.")


# ============================================================
# 8. OLD DATASET EVALUATION (commented out - uncomment if needed)
# ============================================================

# # UNCOMMENT THE LINE BELOW TO RUN DIAGNOSTICS FIRST:
# # diagnose_paths()
# 
# # df = evaluate_noise_robustness(
# #     snr_levels=SNR_LEVELS,
# #     noise_type="white",
# #     max_tracks=None,   # or small int for debugging
# # )
# 
# # df.to_csv("results_vocalset_noise_robustness.csv", index=False)
# # plot_rpa_vs_snr(df)
# # df.head()
