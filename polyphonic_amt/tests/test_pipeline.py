"""Model-backed tests (Demucs + torchcrepe). Run on CPU unless AMT_TEST_DEVICE=cuda."""
import os

import numpy as np
import pytest

from amt.audio import prepare_pitch_input, to_stereo
from amt.config import HOP_LENGTH, SAMPLE_RATE

DEVICE = os.environ.get("AMT_TEST_DEVICE", "cpu")


def harmonic(freq, seconds, sr, harmonics=8):
    t = np.arange(int(sr * seconds)) / sr
    return (0.2 * sum(np.sin(2 * np.pi * freq * h * t) / h for h in range(1, harmonics + 1))).astype(np.float32)


@pytest.fixture(scope="module")
def tracker():
    from amt.pitch import PitchTracker

    return PitchTracker(device=DEVICE)


@pytest.fixture(scope="module")
def detector():
    from amt.voicing import VoicingDetector

    return VoicingDetector()


def run_branches(tracker, detector, y, fmin=60.0, fmax=1100.0):
    from amt.gating import apply_gating

    f0, per = tracker.predict(y, fmin, fmax)
    return f0, apply_gating(f0, per, detector.mask(y), 0.15)


def test_tcu05_silence_gives_no_pitch(tracker, detector):
    _, gated = run_branches(tracker, detector, np.zeros(SAMPLE_RATE * 2, np.float32))
    assert gated.max() == 0.0


def test_tcu04_noise_is_suppressed(tracker, detector):
    noise = (0.1 * np.random.default_rng(3).standard_normal(SAMPLE_RATE * 3)).astype(np.float32)
    _, gated = run_branches(tracker, detector, noise)
    assert np.count_nonzero(gated) / len(gated) < 0.05


@pytest.mark.parametrize("freq", [110.0, 220.0, 440.0])
def test_tcu06_same_melody_in_different_octaves(tracker, detector, freq):
    y = harmonic(freq, 2.0, SAMPLE_RATE)
    f0, gated = run_branches(tracker, detector, y)
    voiced = gated[gated > 0]
    assert len(voiced) > 0.8 * len(gated)
    assert abs(1200 * np.log2(np.median(voiced) / freq)) < 50


def test_pitch_frame_rate_is_10ms(tracker):
    y = harmonic(220.0, 2.0, SAMPLE_RATE)
    f0, per = tracker.predict(y, 60.0, 1100.0)
    assert abs(len(f0) - (1 + len(y) // HOP_LENGTH)) <= 1 and len(f0) == len(per)


def test_separator_returns_mono_16k_stems_for_stereo_44k_input():
    from amt.separator import AudioSeparator

    sep = AudioSeparator(device=DEVICE)
    assert sep.model_sr == 44100
    sr = 44100
    wave = to_stereo(harmonic(220.0, 3.0, sr))
    stems = sep.separate(wave, sr)
    assert set(stems) == {"drums", "bass", "other", "vocals"}
    for s in stems.values():
        assert s.ndim == 1  # mono, not [2, T]
        assert abs(len(s) - 3 * SAMPLE_RATE) <= 2  # 16 kHz, same duration as the input


def test_separator_resamples_16k_input_to_model_rate():
    from amt.separator import AudioSeparator

    sep = AudioSeparator(device=DEVICE)
    sr = 16000
    stereo = sep.separate_stereo(to_stereo(harmonic(220.0, 2.0, sr)), sr)
    assert all(v.shape[0] == 2 and abs(v.shape[1] - 2 * 44100) <= 2 for v in stereo.values())


def test_prepare_pitch_input_downmixes_and_resamples():
    st = np.stack([harmonic(220.0, 1.0, 44100), harmonic(220.0, 1.0, 44100)])
    y = prepare_pitch_input(st, 44100)
    assert y.ndim == 1 and abs(len(y) - 16000) <= 1
