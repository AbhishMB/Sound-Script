import numpy as np

from evaluation.metrics import melody_metrics

T = np.arange(0, 2, 0.01)
REF = np.where(T < 1.5, 220.0, 0.0)  # voiced for the first 1.5 s


def test_perfect_estimate():
    m = melody_metrics(T, REF, T, np.full(len(T), 220.0), REF > 0)
    assert m["rpa"] == 1.0 and m["rpa_gated"] == 1.0 and m["precision"] == 1.0 and m["vfa"] == 0.0


def test_gating_does_not_change_standard_rpa_but_changes_gated_rpa():
    est = np.full(len(T), 220.0)
    ungated = melody_metrics(T, REF, T, est, np.ones(len(T), bool))
    all_off = melody_metrics(T, REF, T, est, np.zeros(len(T), bool))
    assert ungated["rpa"] == all_off["rpa"] == 1.0
    assert ungated["rpa_gated"] == 1.0 and all_off["rpa_gated"] == 0.0
    assert ungated["vfa"] == 1.0 and all_off["vfa"] == 0.0


def test_octave_error_counts_for_chroma_but_not_pitch_accuracy():
    m = melody_metrics(T, REF, T, np.full(len(T), 440.0), REF > 0)
    assert m["rpa"] == 0.0 and m["rca"] == 1.0


def test_within_50_cents_is_correct():
    near = 220.0 * 2 ** (40 / 1200)
    far = 220.0 * 2 ** (60 / 1200)
    assert melody_metrics(T, REF, T, np.full(len(T), near), REF > 0)["rpa"] == 1.0
    assert melody_metrics(T, REF, T, np.full(len(T), far), REF > 0)["rpa"] == 0.0


def test_estimate_on_a_different_frame_grid_is_resampled():
    t2 = np.arange(0, 2, 0.0116)
    ref = np.where(t2 < 1.5, 220.0, 0.0)
    m = melody_metrics(t2, ref, T, np.full(len(T), 220.0), T < 1.5)
    assert m["rpa"] > 0.99
