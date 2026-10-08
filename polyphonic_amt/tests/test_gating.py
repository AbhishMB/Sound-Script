import numpy as np

from amt.gating import apply_gating

F0 = np.array([100.0, 200.0, 300.0, 400.0])
PER = np.array([0.9, 0.9, 0.05, 0.9])
MASK = np.array([True, False, True, True])


def test_both_gates_must_pass():
    assert apply_gating(F0, PER, MASK, 0.15).tolist() == [100.0, 0.0, 0.0, 400.0]


def test_confidence_gate_alone():
    assert apply_gating(F0, PER, None, 0.15).tolist() == [100.0, 200.0, 0.0, 400.0]


def test_mask_gate_alone():
    assert apply_gating(F0, PER, MASK, None).tolist() == [100.0, 0.0, 300.0, 400.0]


def test_no_gates_is_identity():
    assert apply_gating(F0, PER, None, None).tolist() == F0.tolist()


def test_threshold_is_inclusive_at_015():
    out = apply_gating(np.array([100.0, 100.0]), np.array([0.15, 0.1499]), None, 0.15)
    assert out.tolist() == [100.0, 0.0]


def test_short_mask_marks_remaining_frames_unvoiced():
    out = apply_gating(F0, np.ones(4), np.array([True, True]), None)
    assert out.tolist() == [100.0, 200.0, 0.0, 0.0]
