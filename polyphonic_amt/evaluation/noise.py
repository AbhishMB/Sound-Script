import zlib

import numpy as np

PITCH_BAND_HZ = 8000.0  # Nyquist of the 16 kHz pitch/feature branches


def seed_for(*parts) -> int:
    return zlib.crc32("|".join(map(str, parts)).encode())


def add_white_noise(wave: np.ndarray, sr: int, snr_db: float | None, seed: int) -> np.ndarray:
    """Add white Gaussian noise to [C, T] audio.

    SNR is measured over the 0-8 kHz band the 16 kHz pitch branch sees, so a
    plain-CREPE baseline and the Demucs path face the same in-band noise level.
    The noise itself is full-band at `sr`, i.e. Demucs sees more total noise
    power than the baseline (the harder, more realistic case for the hybrid).
    """
    if snr_db is None:
        return wave
    rng = np.random.default_rng(seed)
    p_sig = float(np.mean(wave ** 2))
    p_noise_inband = p_sig / 10 ** (snr_db / 10.0)
    var = p_noise_inband * (sr / 2.0) / PITCH_BAND_HZ
    return wave + rng.standard_normal(wave.shape).astype(np.float32) * np.float32(np.sqrt(var))


def mix_with_accompaniment(
    vocal: np.ndarray, accompaniment: list[np.ndarray], vocal_to_acc_db: float, rng: np.random.Generator
) -> np.ndarray:
    """[C, T] vocal + a random concatenation of 7 s accompaniment excerpts.

    The accompaniment is scaled so that 10*log10(P_vocal / P_acc) equals
    `vocal_to_acc_db`, where both powers are whole-signal means.
    """
    T = vocal.shape[1]
    pieces, total = [], 0
    while total < T:
        a = accompaniment[rng.integers(len(accompaniment))]
        pieces.append(a)
        total += a.shape[1]
    acc = np.concatenate(pieces, axis=1)[:, :T]
    vocal2 = vocal if vocal.shape[0] == 2 else np.repeat(vocal, 2, axis=0)
    p_v, p_a = np.mean(vocal2 ** 2), np.mean(acc ** 2) + 1e-12
    acc = acc * np.sqrt(p_v / (p_a * 10 ** (vocal_to_acc_db / 10.0)))
    return (vocal2 + acc).astype(np.float32)
