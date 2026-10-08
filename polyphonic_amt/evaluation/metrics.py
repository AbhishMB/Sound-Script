import mir_eval
import numpy as np

CENT_TOL = 50.0


def melody_metrics(ref_t, ref_f, est_t, est_f_raw, est_voiced) -> dict:
    """Frame-level melody metrics for one clip.

    est_f_raw: raw pitch track (Hz, > 0 where the tracker produced a value).
    est_voiced: boolean per-frame decision from the system under test.

    Pitch values are always kept and voicing is carried by the sign (the
    standard mir_eval convention), so:

    * rpa        standard mir_eval raw pitch accuracy: pitch within 50 cents on
                 reference-voiced frames, independent of the voicing decision.
                 A gate cannot change it; only a better pitch track can.
    * rpa_gated  fraction of reference-voiced frames that are marked voiced AND
                 within 50 cents. This is the operating point the MIDI output
                 reflects, because gated-out frames emit no note.
    * precision  correct voiced frames / voiced frames emitted.
    * recall     = rpa_gated.
    * vfa        voicing false-alarm rate: unvoiced reference frames marked voiced.
    """
    est_signed = np.where(est_voiced, est_f_raw, -est_f_raw)
    est_signed = np.where(est_f_raw > 0, est_signed, -1.0)  # no pitch at all -> unvoiced
    ref_v, ref_c, est_v, est_c = mir_eval.melody.to_cent_voicing(ref_t, ref_f, est_t, est_signed)
    ref_v = ref_v.astype(bool)
    est_v = est_v.astype(bool)
    diff = np.abs(ref_c - est_c)
    close = diff < CENT_TOL
    n_ref = max(int(ref_v.sum()), 1)
    tp = int((ref_v & est_v & close).sum())
    recall = tp / n_ref
    precision = tp / max(int(est_v.sum()), 1)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    chroma_close = np.abs(((ref_c - est_c) + 600.0) % 1200.0 - 600.0) < CENT_TOL
    return {
        "rpa": float((ref_v & close).sum() / n_ref),
        "rca": float((ref_v & chroma_close).sum() / n_ref),
        "rpa_gated": recall,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "vr": float((ref_v & est_v).sum() / n_ref),
        "vfa": float((~ref_v & est_v).sum() / max(int((~ref_v).sum()), 1)),
        "voiced_rate": float(est_v.mean()),
    }


def note_metrics(ref_notes: np.ndarray, est_notes: list[tuple[int, float, float]]) -> dict:
    """Note-level P/R/F1 (onset +-50 ms, pitch +-50 cents, offsets ignored)."""
    if len(est_notes) == 0 or len(ref_notes) == 0:
        return {"note_p": 0.0, "note_r": 0.0, "note_f1": 0.0}
    est = np.array(est_notes, dtype=float)
    est_hz = 440.0 * 2.0 ** ((est[:, 0] - 69.0) / 12.0)
    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_notes[:, :2], ref_notes[:, 2], est[:, 1:3], est_hz, offset_ratio=None
    )
    return {"note_p": float(p), "note_r": float(r), "note_f1": float(f)}
