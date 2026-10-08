"""Branch B: Mel / chroma voicing mask.

The paper builds a binary voicing mask from a 256-band Mel spectrogram and a
12-class chroma profile. Two details of that description do not work as
written, and are handled explicitly here:

* Chroma has to be derived from the Mel bins' *centre frequencies*. Treating
  Mel bins as if they were linear-frequency bins shifts every pitch class
  (a 220 Hz A comes out as C#).
* Thresholding the mean chroma energy at its 10th percentile keeps ~90 % of
  the frames by construction, for a pure tone and for white noise alike. It
  cannot tell tonal frames from noise. The "harmonic" mask therefore uses how
  *concentrated* the chroma profile is (harmonic sounds put their energy in a
  few pitch classes, noise spreads it evenly), with the decision threshold
  calibrated against synthetic white noise so the false-alarm rate is fixed.
  The literal percentile rule is kept as mode="paper_literal" for comparison.
"""
from dataclasses import dataclass

import librosa
import numpy as np

from .config import (
    CHROMA_FMAX,
    CHROMA_FMIN,
    HOP_LENGTH,
    MASK_ABS_FLOOR_DBFS,
    MASK_ENERGY_FLOOR_DB,
    MASK_FALSE_ALARM_RATE,
    N_CHROMA_BINS,
    N_FFT,
    N_MEL_FILTERS,
    SAMPLE_RATE,
    VOICING_PERCENTILE,
)


@dataclass
class VoicingFeatures:
    mel: np.ndarray  # [n_mels, frames] power
    chroma: np.ndarray  # [12, frames] power, row 0 = C
    energy_db: np.ndarray  # [frames] chroma-profile energy in dB
    concentration: np.ndarray  # [frames] in [0, 1]; 0 = flat (noise), 1 = single class
    rms_dbfs: np.ndarray  # [frames] frame RMS in dB re full scale


def mel_to_chroma_matrix(
    mel_freqs: np.ndarray, n_chroma: int = N_CHROMA_BINS, sigma: float = 0.5
) -> np.ndarray:
    """[n_chroma, n_mels] soft assignment of Mel-bin centre frequencies to pitch classes."""
    semitones = 12.0 * np.log2(np.maximum(mel_freqs, 1e-3) / 440.0) + 9.0  # C = 0
    classes = np.arange(n_chroma)[:, None] * (12.0 / n_chroma)
    dist = np.abs(((semitones[None, :] - classes) + 6.0) % 12.0 - 6.0)
    return np.exp(-0.5 * (dist / sigma) ** 2)


class VoicingDetector:
    def __init__(
        self,
        sr: int = SAMPLE_RATE,
        hop_length: int = HOP_LENGTH,
        n_fft: int = N_FFT,
        n_mels: int = N_MEL_FILTERS,
        fmin: float = CHROMA_FMIN,
        fmax: float = CHROMA_FMAX,
        mode: str = "harmonic",
        false_alarm_rate: float = MASK_FALSE_ALARM_RATE,
        energy_floor_db: float = MASK_ENERGY_FLOOR_DB,
        abs_floor_dbfs: float = MASK_ABS_FLOOR_DBFS,
        percentile: float = VOICING_PERCENTILE,
    ):
        assert mode in ("harmonic", "paper_literal")
        self.sr, self.hop, self.n_fft, self.n_mels = sr, hop_length, n_fft, n_mels
        self.fmin, self.fmax = fmin, fmax
        self.mode = mode
        self.far = false_alarm_rate
        self.energy_floor_db = energy_floor_db
        self.abs_floor_dbfs = abs_floor_dbfs
        self.percentile = percentile
        mel_edges = librosa.mel_frequencies(n_mels + 2, fmin=fmin, fmax=fmax)
        self._chroma_w = mel_to_chroma_matrix(mel_edges[1:-1])
        self._noise_threshold: float | None = None

    # ---- features -----------------------------------------------------
    def features(self, y: np.ndarray) -> VoicingFeatures:
        y = np.ascontiguousarray(y, dtype=np.float32)
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
            power=2.0,
        )
        chroma = self._chroma_w @ mel
        total = chroma.sum(axis=0)
        p = chroma / np.maximum(total, 1e-20)
        entropy = -(p * np.log(np.maximum(p, 1e-20))).sum(axis=0)
        concentration = 1.0 - entropy / np.log(self.n_chroma_classes)
        rms = librosa.feature.rms(y=y, frame_length=self.n_fft, hop_length=self.hop)[0]
        return VoicingFeatures(
            mel=mel,
            chroma=chroma,
            energy_db=10.0 * np.log10(np.maximum(total / self.n_chroma_classes, 1e-20)),
            concentration=concentration,
            rms_dbfs=20.0 * np.log10(np.maximum(rms[: mel.shape[1]], 1e-10)),
        )

    @property
    def n_chroma_classes(self) -> int:
        return self._chroma_w.shape[0]

    # ---- calibration ---------------------------------------------------
    def noise_threshold(self) -> float:
        """Concentration value that `false_alarm_rate` of white-noise frames exceed."""
        if self._noise_threshold is None:
            rng = np.random.default_rng(0)
            noise = rng.standard_normal(self.sr * 30).astype(np.float32)
            conc = self.features(noise).concentration
            self._noise_threshold = float(np.quantile(conc, 1.0 - self.far))
        return self._noise_threshold

    # ---- mask ----------------------------------------------------------
    def mask(self, y: np.ndarray, feats: VoicingFeatures | None = None) -> np.ndarray:
        """Boolean voiced/unvoiced mask, one value per 10 ms frame."""
        f = feats or self.features(y)
        if self.mode == "paper_literal":
            return f.energy_db > np.percentile(f.energy_db, self.percentile)
        loud = np.percentile(f.energy_db, 95)
        return (
            (f.concentration > self.noise_threshold())
            & (f.energy_db > loud - self.energy_floor_db)
            & (f.rms_dbfs > self.abs_floor_dbfs)
        )
