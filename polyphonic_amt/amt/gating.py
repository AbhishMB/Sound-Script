import numpy as np

from .config import CREPE_CONFIDENCE_THRESH


def apply_gating(
    f0: np.ndarray,
    periodicity: np.ndarray,
    voicing_mask: np.ndarray | None = None,
    confidence_threshold: float | None = CREPE_CONFIDENCE_THRESH,
) -> np.ndarray:
    """Fusion logic: f0(t) is kept only where every enabled gate passes.

        f0_gated(t) = f0(t)  if Mask(t) and C(t) >= threshold  else 0

    Passing `voicing_mask=None` disables the mask gate and
    `confidence_threshold=None` disables the confidence gate, which is what the
    gate-ablation experiments use.
    """
    n = len(f0)
    keep = np.ones(n, dtype=bool)
    if confidence_threshold is not None:
        keep &= periodicity[:n] >= confidence_threshold
    if voicing_mask is not None:
        m = np.zeros(n, dtype=bool)  # frames past the end of the mask are unvoiced
        k = min(n, len(voicing_mask))
        m[:k] = voicing_mask[:k]
        keep &= m
    return np.where(keep, f0, 0.0)
