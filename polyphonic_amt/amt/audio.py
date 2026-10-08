import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF


def load_audio(path: str) -> tuple[np.ndarray, int]:
    """Load an audio file as float32 [channels, samples] at its native rate."""
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    return data.T.copy(), sr


def resample(wave: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Resample [..., samples] audio."""
    if sr_in == sr_out:
        return wave
    out = AF.resample(torch.from_numpy(np.ascontiguousarray(wave)), sr_in, sr_out)
    return out.numpy()


def to_mono(wave: np.ndarray) -> np.ndarray:
    """Collapse [channels, samples] (or already 1-D) audio to 1-D."""
    return wave if wave.ndim == 1 else wave.mean(axis=0)


def to_stereo(wave: np.ndarray) -> np.ndarray:
    """Return [2, samples]; mono is duplicated, extra channels are dropped."""
    if wave.ndim == 1:
        wave = wave[None]
    if wave.shape[0] == 1:
        wave = np.repeat(wave, 2, axis=0)
    return wave[:2]


def prepare_pitch_input(wave: np.ndarray, sr_in: int, sr_out: int = 16000) -> np.ndarray:
    """Any-rate, any-channel audio -> 1-D float32 at the pitch/feature rate."""
    return resample(to_mono(wave), sr_in, sr_out).astype(np.float32)
