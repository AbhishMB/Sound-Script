import numpy as np
import pytest

from amt.voicing import VoicingDetector

SR = 16000


def tone(freq, seconds=3.0, harmonics=6):
    t = np.arange(int(SR * seconds)) / SR
    y = sum(np.sin(2 * np.pi * freq * h * t) / h for h in range(1, harmonics + 1))
    return (0.2 * y).astype(np.float32)


@pytest.fixture(scope="module")
def det():
    return VoicingDetector()


@pytest.mark.parametrize("freq,pitch_class", [(261.63, 0), (220.0, 9), (329.63, 4), (392.0, 7)])
def test_chroma_peaks_at_the_right_pitch_class(det, freq, pitch_class):
    """Chroma must come from Mel centre frequencies: 220 Hz is A (9), not C# (1)."""
    y = np.sin(2 * np.pi * freq * np.arange(SR * 2) / SR).astype(np.float32) * 0.3
    chroma = det.features(y).chroma.mean(axis=1)
    assert chroma.argmax() == pitch_class


def test_harmonic_mask_keeps_tone_and_rejects_white_noise(det):
    noise = (0.1 * np.random.default_rng(1).standard_normal(SR * 4)).astype(np.float32)
    assert det.mask(tone(220.0)).mean() > 0.9
    assert det.mask(noise).mean() < 0.15


def test_harmonic_mask_rejects_silence(det):
    assert det.mask(np.zeros(SR * 2, np.float32)).sum() == 0
    assert det.mask(np.full(SR * 2, 1e-6, np.float32)).sum() == 0


def test_noise_threshold_gives_requested_false_alarm_rate():
    d = VoicingDetector(false_alarm_rate=0.05)
    noise = np.random.default_rng(7).standard_normal(SR * 20).astype(np.float32)
    f = d.features(noise)
    assert abs((f.concentration > d.noise_threshold()).mean() - 0.05) < 0.03


def test_paper_literal_percentile_mask_cannot_separate_tone_from_noise():
    """Documents why the literal 10th-percentile rule is not the default:
    it keeps ~90 % of frames whatever the content, by construction."""
    d = VoicingDetector(mode="paper_literal")
    noise = (0.1 * np.random.default_rng(2).standard_normal(SR * 4)).astype(np.float32)
    assert d.mask(noise).mean() > 0.85
    assert d.mask(tone(220.0)).mean() > 0.7


def test_mask_has_one_value_per_crepe_frame(det):
    y = tone(220.0, 2.0)
    assert abs(len(det.mask(y)) - (1 + len(y) // 160)) <= 1
