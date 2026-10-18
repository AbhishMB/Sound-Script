"""
Evaluation & Comparison Graphs for Hybrid CREPE Pipeline
=========================================================
Generates:
  1. RPA vs SNR graph  (baseline CREPE vs hybrid)
  2. Baseline vs Hybrid comparison plots
  3. Per-stem analysis

Usage:
    python evaluate_pipeline.py <audio_file> [--output_dir figures]
    python evaluate_pipeline.py <audio_file> --annotation <csv_file>
"""

import os
import argparse
import numpy as np
import librosa
import torch
import torchcrepe
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import butter, lfilter
from scipy.ndimage import median_filter

# Import from our pipeline
from hybrid_pipeline import (
    SR, HOP_LENGTH, N_MELS, CONF_THRESH, FMIN, FMAX, ALL_STEMS,
    PitchEstimator, VoicingDetector, FusionLogic,
    denoise_audio, SourceSeparator, process_single_stem,
)

# ---------------------------------------------------------------------------
# Noise Augmentation
# ---------------------------------------------------------------------------

class NoiseAugmenter:
    """Add synthetic noise at a specified SNR level."""

    @staticmethod
    def add_noise(audio, noise_type="white", snr_db=20):
        N = len(audio)
        if noise_type == "white":
            noise = np.random.randn(N)
        elif noise_type == "pink":
            b, a = butter(1, 0.04, btype="low")
            noise = lfilter(b, a, np.random.randn(N))
        else:  # brown
            noise = np.cumsum(np.random.randn(N))

        sig_power = np.mean(audio ** 2) + 1e-12
        sig_db = 10 * np.log10(sig_power)
        noise_db = sig_db - snr_db
        noise_power = 10 ** (noise_db / 10)
        current_noise_power = np.mean(noise ** 2) + 1e-12
        noise = noise * np.sqrt(noise_power / current_noise_power)
        return audio + noise


# ---------------------------------------------------------------------------
# Metric Helpers
# ---------------------------------------------------------------------------

def run_crepe_baseline(audio, sr, estimator):
    """Raw CREPE — no denoise, no mask, no gating."""
    return estimator.predict(audio, sr)


def run_crepe_hybrid(audio, sr, estimator, detector=None, fusion=None):
    """Full hybrid: denoise → CREPE + voicing mask → fusion gating."""
    if detector is None:
        detector = VoicingDetector()
    if fusion is None:
        fusion = FusionLogic()

    denoised = denoise_audio(audio, hop_length=HOP_LENGTH)
    est_time, est_freq, est_conf = estimator.predict(denoised, sr)
    mask = detector.get_voicing_mask(denoised, sr)
    est_freq_filtered = fusion.apply(est_time, est_freq, est_conf, mask)
    return est_time, est_freq_filtered, est_conf


def calculate_rpa(ref_freq, est_freq):
    """Raw Pitch Accuracy via mir_eval."""
    import mir_eval
    ref_voicing = (ref_freq > 0).astype(bool)
    est_voicing = (est_freq > 0).astype(bool)
    rpa = mir_eval.melody.raw_pitch_accuracy(
        ref_voicing, ref_freq, est_voicing, est_freq
    )
    return rpa * 100.0


def compute_voicing_stats(ref_freq, est_freq):
    """Precision, recall, F1, FP rate for voicing detection."""
    ref_v = ref_freq > 0
    est_v = est_freq > 0
    tp = np.sum(ref_v & est_v)
    fp = np.sum(~ref_v & est_v)
    fn = np.sum(ref_v & ~est_v)
    tn = np.sum(~ref_v & ~est_v)
    prec = tp / (tp + fp + 1e-8)
    rec  = tp / (tp + fn + 1e-8)
    f1   = 2 * prec * rec / (prec + rec + 1e-8)
    fp_rate = fp / (fp + tn + 1e-8)
    return prec, rec, f1, fp_rate


def resample_f0_to_hop(times, freqs, audio_len_samples,
                       sr=SR, hop_length=HOP_LENGTH):
    n_frames = int(np.ceil(audio_len_samples / hop_length))
    t_grid = np.arange(n_frames) * (hop_length / sr)
    freqs_clean = np.nan_to_num(freqs, nan=0.0)
    f_interp = np.interp(t_grid, times, freqs_clean, left=0.0, right=0.0)
    return t_grid, f_interp


# ---------------------------------------------------------------------------
# 1.  RPA vs SNR  Evaluation
# ---------------------------------------------------------------------------

SNR_LEVELS = [40, 30, 20, 10, 5, 0]


def evaluate_noise_robustness(audio_path, annotation_path=None,
                              snr_levels=None, noise_type="white"):
    """
    Evaluate baseline CREPE vs hybrid under varying noise conditions.

    If annotation_path is provided (CSV with columns time, freq), compute RPA.
    Otherwise use the clean hybrid output as pseudo-reference.

    Returns a pandas DataFrame with columns:
        snr_db, system, rpa, voicing_precision, voicing_recall,
        voicing_f1, fp_rate_unvoiced
    """
    if snr_levels is None:
        snr_levels = list(SNR_LEVELS)

    estimator = PitchEstimator()
    detector  = VoicingDetector()
    fusion    = FusionLogic()

    audio, _ = librosa.load(audio_path, sr=SR)
    n_samples = len(audio)

    # Load or generate reference f0
    if annotation_path and os.path.isfile(annotation_path):
        data = np.loadtxt(annotation_path, delimiter=",")
        ref_times, ref_freqs = data[:, 0], data[:, 1]
        _, ref_f = resample_f0_to_hop(ref_times, ref_freqs, n_samples)
    else:
        # Use clean hybrid as pseudo-reference
        _, ref_f, _ = run_crepe_hybrid(audio, SR, estimator, detector, fusion)
        print("  [info] No annotation file; using clean hybrid as reference.")

    results = []

    # --- Clean condition ---
    _, b_freq, _ = run_crepe_baseline(audio, SR, estimator)
    _, h_freq, _ = run_crepe_hybrid(audio, SR, estimator, detector, fusion)

    for tag, ef in [("baseline", b_freq), ("hybrid", h_freq)]:
        L = min(len(ref_f), len(ef))
        rpa = calculate_rpa(ref_f[:L], ef[:L])
        p, r, f1, fp = compute_voicing_stats(ref_f[:L], ef[:L])
        results.append(dict(snr_db=None, system=tag, rpa=rpa,
                            voicing_precision=p, voicing_recall=r,
                            voicing_f1=f1, fp_rate_unvoiced=fp))

    print(f"  Clean  | baseline RPA {results[0]['rpa']:.2f}  "
          f"| hybrid RPA {results[1]['rpa']:.2f}")

    # --- Noisy conditions ---
    for snr in snr_levels:
        noisy = NoiseAugmenter.add_noise(audio, noise_type, snr_db=snr)
        _, b_freq_n, _ = run_crepe_baseline(noisy, SR, estimator)
        _, h_freq_n, _ = run_crepe_hybrid(noisy, SR, estimator, detector, fusion)

        for tag, ef in [("baseline", b_freq_n), ("hybrid", h_freq_n)]:
            L = min(len(ref_f), len(ef))
            rpa = calculate_rpa(ref_f[:L], ef[:L])
            p, r, f1, fp = compute_voicing_stats(ref_f[:L], ef[:L])
            results.append(dict(snr_db=snr, system=tag, rpa=rpa,
                                voicing_precision=p, voicing_recall=r,
                                voicing_f1=f1, fp_rate_unvoiced=fp))

        b_rpa = [r for r in results if r["snr_db"] == snr and r["system"] == "baseline"][0]["rpa"]
        h_rpa = [r for r in results if r["snr_db"] == snr and r["system"] == "hybrid"][0]["rpa"]
        print(f"  SNR {snr:>2} | baseline RPA {b_rpa:.2f}  | hybrid RPA {h_rpa:.2f}")

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# 2.  Plotting Functions
# ---------------------------------------------------------------------------

def plot_rpa_vs_snr(df, output_path="rpa_vs_snr.png"):
    """Plot RPA vs SNR for baseline and hybrid systems."""
    sub = df[df["snr_db"].notnull()]
    if sub.empty:
        print("No SNR data to plot.")
        return

    plt.figure(figsize=(8, 5))
    colors = {"baseline": "#3b82f6", "hybrid": "#ef4444"}
    markers = {"baseline": "o", "hybrid": "s"}

    for system in ["baseline", "hybrid"]:
        d = sub[sub["system"] == system]
        means = d.groupby("snr_db")["rpa"].mean().sort_index(ascending=False)
        plt.plot(means.index, means.values, f"{markers[system]}-",
                 color=colors[system], label=f"{system.capitalize()} CREPE",
                 linewidth=2, markersize=8)

    # Add clean reference point
    for system in ["baseline", "hybrid"]:
        clean = df[(df["snr_db"].isnull()) & (df["system"] == system)]
        if not clean.empty:
            rpa_val = clean["rpa"].values[0]
            plt.axhline(y=rpa_val, color=colors[system], linestyle="--",
                        alpha=0.4, label=f"{system.capitalize()} (clean)")

    plt.gca().invert_xaxis()
    plt.xlabel("SNR (dB)", fontsize=12)
    plt.ylabel("Raw Pitch Accuracy (%)", fontsize=12)
    plt.title("RPA vs SNR — Baseline CREPE vs Hybrid CREPE", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {output_path}")


def plot_voicing_f1_vs_snr(df, output_path="voicing_f1_vs_snr.png"):
    """Plot Voicing F1 score vs SNR."""
    sub = df[df["snr_db"].notnull()]
    if sub.empty:
        return

    plt.figure(figsize=(8, 5))
    colors = {"baseline": "#3b82f6", "hybrid": "#ef4444"}

    for system in ["baseline", "hybrid"]:
        d = sub[sub["system"] == system]
        means = d.groupby("snr_db")["voicing_f1"].mean().sort_index(ascending=False)
        plt.plot(means.index, means.values, "o-", color=colors[system],
                 label=f"{system.capitalize()} CREPE", linewidth=2, markersize=8)

    plt.gca().invert_xaxis()
    plt.xlabel("SNR (dB)", fontsize=12)
    plt.ylabel("Voicing F1 Score", fontsize=12)
    plt.title("Voicing F1 vs SNR — Baseline CREPE vs Hybrid CREPE", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {output_path}")


def plot_baseline_vs_hybrid_comparison(audio_path, output_path="comparison.png"):
    """
    Side-by-side comparison of baseline CREPE vs hybrid CREPE
    on a single audio file.  Generates 3 subplots:
      1. Pitch contours
      2. Confidence scores
      3. Voicing bar chart
    """
    audio, _ = librosa.load(audio_path, sr=SR)
    estimator = PitchEstimator()
    detector  = VoicingDetector()
    fusion    = FusionLogic()

    t_b, f_b, c_b = run_crepe_baseline(audio, SR, estimator)
    t_h, f_h, c_h = run_crepe_hybrid(audio, SR, estimator, detector, fusion)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # 1. Pitch contours
    axes[0].plot(t_b, f_b, "b-", alpha=0.6, lw=0.5, label="Baseline CREPE")
    axes[0].plot(t_h, f_h, "r-", alpha=0.6, lw=0.5, label="Hybrid CREPE")
    axes[0].set_ylabel("Frequency (Hz)")
    axes[0].set_title("Pitch Estimation Comparison")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. Confidence
    axes[1].plot(t_b, c_b, "b-", alpha=0.6, lw=0.5, label="Baseline")
    axes[1].plot(t_h, c_h, "r-", alpha=0.6, lw=0.5, label="Hybrid")
    axes[1].set_ylabel("Confidence")
    axes[1].set_ylim([0, 1])
    axes[1].set_title("Periodicity Confidence")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 3. Voicing bar chart
    n_v_b = int(np.sum(f_b > 0))
    n_v_h = int(np.sum(f_h > 0))
    n_total = len(f_b)
    categories = ["Voiced", "Unvoiced", "Total"]
    base_vals = [n_v_b, n_total - n_v_b, n_total]
    hyb_vals  = [n_v_h, n_total - n_v_h, n_total]
    x = np.arange(len(categories))
    w = 0.35
    axes[2].bar(x - w/2, base_vals, w, label="Baseline", color="#3b82f6", alpha=0.8)
    axes[2].bar(x + w/2, hyb_vals,  w, label="Hybrid",   color="#ef4444", alpha=0.8)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(categories)
    axes[2].set_ylabel("Frame Count")
    axes[2].set_title("Voicing Statistics")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3, axis="y")
    for i, (bv, hv) in enumerate(zip(base_vals, hyb_vals)):
        axes[2].text(i - w/2, bv, str(bv), ha="center", va="bottom", fontsize=9)
        axes[2].text(i + w/2, hv, str(hv), ha="center", va="bottom", fontsize=9)

    axes[2].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {output_path}")


def plot_per_stem_analysis(input_file, output_path="per_stem_analysis.png"):
    """
    Run the full pipeline on all 4 stems and plot voiced-frame counts
    and ghost-note reduction per stem.
    """
    estimator = PitchEstimator()
    detector  = VoicingDetector()
    fusion    = FusionLogic()

    filename = os.path.splitext(os.path.basename(input_file))[0]
    stem_dir = os.path.join("separated_stems", "htdemucs", filename)

    if not os.path.isdir(stem_dir):
        print(f"Stems not found at {stem_dir}. Run hybrid_pipeline.py first.")
        return

    stem_names, raw_voiced, fused_voiced, ghost_removed = [], [], [], []

    for stem in ALL_STEMS:
        sp = os.path.join(stem_dir, f"{stem}.wav")
        if not os.path.isfile(sp):
            continue
        res = process_single_stem(sp, stem, estimator, detector, fusion,
                                  verbose=False)
        n_raw   = int(np.sum(res["est_freq_raw"] > 0))
        n_fused = int(np.sum(res["est_freq_filtered"] > 0))
        stem_names.append(stem.capitalize())
        raw_voiced.append(n_raw)
        fused_voiced.append(n_fused)
        ghost_removed.append(n_raw - n_fused)

    if not stem_names:
        print("No stems processed.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(stem_names))
    w = 0.35
    axes[0].bar(x - w/2, raw_voiced,   w, label="Baseline (raw)", color="#3b82f6")
    axes[0].bar(x + w/2, fused_voiced, w, label="Hybrid (fused)", color="#ef4444")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(stem_names)
    axes[0].set_ylabel("Voiced Frames")
    axes[0].set_title("Voiced Frames per Stem")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis="y")

    axes[1].bar(x, ghost_removed, color="#f59e0b")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(stem_names)
    axes[1].set_ylabel("Frames Removed")
    axes[1].set_title("Ghost Notes Removed by Fusion Gating")
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {output_path}")


# ---------------------------------------------------------------------------
# 3.  Generate All Figures
# ---------------------------------------------------------------------------

def generate_all_figures(audio_path, output_dir="figures",
                         annotation_path=None):
    """Generate all paper figures in one call."""
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 60)
    print("Generating evaluation figures")
    print("=" * 60)

    # 1. RPA vs SNR
    print("\n[1/4] RPA vs SNR evaluation ...")
    df = evaluate_noise_robustness(audio_path, annotation_path)
    csv_path = os.path.join(output_dir, "noise_robustness_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"  ✓ Results saved: {csv_path}")

    plot_rpa_vs_snr(df, os.path.join(output_dir, "rpa_vs_snr.png"))
    plot_voicing_f1_vs_snr(df, os.path.join(output_dir, "voicing_f1_vs_snr.png"))

    # 2. Baseline vs Hybrid comparison
    print("\n[2/4] Baseline vs Hybrid comparison ...")
    plot_baseline_vs_hybrid_comparison(
        audio_path, os.path.join(output_dir, "baseline_vs_hybrid.png")
    )

    # 3. Per-stem analysis (requires prior separation)
    print("\n[3/4] Per-stem analysis ...")
    plot_per_stem_analysis(
        audio_path, os.path.join(output_dir, "per_stem_analysis.png")
    )

    # 4. Summary table
    print("\n[4/4] Summary statistics ...")
    plot_summary_table(df, os.path.join(output_dir, "summary_table.png"))

    print("\n" + "=" * 60)
    print(f"All figures saved to: {os.path.abspath(output_dir)}")
    print("=" * 60)


def plot_summary_table(df, output_path="summary_table.png"):
    """Render a comparison table as an image."""
    clean = df[df["snr_db"].isnull()]
    if clean.empty:
        return

    rows = []
    for _, row in clean.iterrows():
        rows.append([
            row["system"].capitalize(),
            f"{row['rpa']:.2f}%",
            f"{row['voicing_precision']:.3f}",
            f"{row['voicing_recall']:.3f}",
            f"{row['voicing_f1']:.3f}",
            f"{row['fp_rate_unvoiced']:.3f}",
        ])

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis("off")
    headers = ["System", "RPA", "Precision", "Recall", "F1", "FP Rate"]
    table = ax.table(cellText=rows, colLabels=headers,
                     cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)

    for i in range(len(headers)):
        table[(0, i)].set_facecolor("#40466e")
        table[(0, i)].set_text_props(weight="bold", color="white")

    plt.title("Clean Audio — Baseline vs Hybrid", fontsize=13,
              fontweight="bold", pad=20)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Hybrid CREPE Pipeline & Generate Graphs"
    )
    parser.add_argument("audio_file", help="Path to input audio WAV")
    parser.add_argument("--annotation", default=None,
                        help="CSV with ground-truth f0 (time, freq)")
    parser.add_argument("--output_dir", default="figures",
                        help="Directory for output figures (default: figures)")
    args = parser.parse_args()

    generate_all_figures(
        audio_path=args.audio_file,
        output_dir=args.output_dir,
        annotation_path=args.annotation,
    )


if __name__ == "__main__":
    main()
