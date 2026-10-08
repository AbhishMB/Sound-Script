"""Noise / polyphonic evaluation of every system variant on the same inputs.

For each clip and condition the script runs

    mix path : audio -> 16 kHz mono -> torchcrepe                 ("crepe")
    sep path : audio -> Demucs vocals stem -> torchcrepe           ("sep")

each with the pipeline's vocal search range (60-1100 Hz) and, tagged "F"
(crepeF / sepF), torchcrepe's default 50-2006 Hz range.

and then applies each gate configuration to the *same* f0 / periodicity /
mask arrays, so differences between rows come only from the gate or from the
separation stage:

    raw          no gate                       (plain CREPE for the mix path)
    conf50       periodicity >= 0.5            (conventional CREPE voicing cut-off)
    conf         periodicity >= 0.15           (paper's confidence gate)
    mask         harmonic voicing mask only
    conf+mask    paper's fusion logic
    pmask        paper-literal 10th-percentile mask only
    conf+pmask   confidence + paper-literal mask

Results are appended to a JSONL file as they are produced, and finished
(clip, condition) pairs are skipped on restart.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from amt.audio import load_audio, prepare_pitch_input  # noqa: E402
from amt.config import HOP_LENGTH, SAMPLE_RATE, STEM_F0_RANGE  # noqa: E402
from amt.gating import apply_gating  # noqa: E402
from amt.midi import frames_to_notes  # noqa: E402
from amt.pitch import PitchTracker  # noqa: E402
from amt.separator import AudioSeparator  # noqa: E402
from amt.voicing import VoicingDetector  # noqa: E402
from evaluation.datasets import (  # noqa: E402
    VOCALSET_DEV_SINGERS,
    list_vocadito,
    list_vocalset,
    load_musdb_accompaniments,
    stratified_subset,
)
from evaluation.metrics import melody_metrics, note_metrics  # noqa: E402
from evaluation.noise import add_white_noise, mix_with_accompaniment, seed_for  # noqa: E402

FRAME = HOP_LENGTH / SAMPLE_RATE
GATES = {
    "raw": dict(conf=None, mask=None),
    "conf50": dict(conf=0.5, mask=None),
    "conf": dict(conf=0.15, mask=None),
    "mask": dict(conf=None, mask="h"),
    "conf+mask": dict(conf=0.15, mask="h"),
    "pmask": dict(conf=None, mask="p"),
    "conf+pmask": dict(conf=0.15, mask="p"),
}


def condition_key(clip, cond) -> str:
    return f"{clip.dataset}/{clip.name}/{cond}"


RANGES = {
    "": STEM_F0_RANGE["vocals"],  # pipeline search range for the vocals stem
    "F": (50.0, 2006.0),  # torchcrepe default range ("plain CREPE")
}


def evaluate_signal(clip, sig16, tracker, det_h, det_p, with_notes: bool):
    """Per-range, per-gate metrics for one 16 kHz signal against one clip's reference.

    Returns {range_tag: {gate: metrics}}; the mask is computed once per signal.
    """
    feats = det_h.features(sig16)
    masks = {"h": det_h.mask(sig16, feats), "p": det_p.mask(sig16, feats)}
    out = {}
    for tag, (fmin, fmax) in RANGES.items():
        f0, per = tracker.predict(sig16, fmin, fmax)
        est_t = np.arange(len(f0)) * FRAME
        res = {}
        for name, g in GATES.items():
            gated = apply_gating(f0, per, masks[g["mask"]] if g["mask"] else None, g["conf"])
            m = melody_metrics(clip.ref_times, clip.ref_freqs, est_t, f0, gated > 0)
            if with_notes and clip.notes is not None:
                m.update(note_metrics(clip.notes, frames_to_notes(gated)))
            res[name] = m
        out[tag] = res
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("AMT_DATA", "data"),
                    help="dataset root created by evaluation/get_data.py (env: AMT_DATA)")
    ap.add_argument("--experiment", choices=["noise", "polyphonic"], default="noise")
    ap.add_argument("--dataset", choices=["vocalset", "vocadito"], default="vocalset")
    ap.add_argument("--split", choices=["test", "dev"], default="test")
    ap.add_argument("--per-singer", type=int, default=6, help="VocalSet clips per singer (0 = all)")
    ap.add_argument("--snr", type=float, nargs="*", default=[40, 30, 20, 10, 5, 0],
                    help="noise SNR(dB) list; 'clean' is always run for the noise experiment")
    ap.add_argument("--vac", type=float, nargs="*", default=[10, 5, 0, -5],
                    help="polyphonic: vocal-to-accompaniment ratios in dB")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.dataset == "vocalset":
        if args.split == "dev":
            clips = list_vocalset(args.data, singers=VOCALSET_DEV_SINGERS)
        else:
            clips = list_vocalset(args.data, exclude_singers=VOCALSET_DEV_SINGERS)
        if args.per_singer:
            clips = stratified_subset(clips, args.per_singer)
    else:
        clips = list_vocadito(args.data)
    if args.limit:
        clips = clips[: args.limit]

    accompaniment = None
    if args.experiment == "polyphonic":
        import imageio_ffmpeg

        accompaniment = load_musdb_accompaniments(
            args.data, imageio_ffmpeg.get_ffmpeg_exe(), os.path.join(args.data, "cache_musdb")
        )
        conditions = [f"vac{v:g}" for v in args.vac]
    else:
        conditions = ["clean"] + [f"snr{s:g}" for s in args.snr]

    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            for line in f:
                r = json.loads(line)
                done.add((r["dataset"], r["clip"], r["condition"]))

    separator = AudioSeparator()
    tracker = PitchTracker()
    det_h = VoicingDetector(mode="harmonic")
    det_p = VoicingDetector(mode="paper_literal")

    t0 = time.time()
    n_total = len(clips) * len(conditions)
    n_done = 0
    with open(args.out, "a") as out:
        for clip in clips:
            wave, sr = load_audio(clip.audio_path)
            wave = wave[:1]
            for cond in conditions:
                n_done += 1
                if (clip.dataset, clip.name, cond) in done:
                    continue
                seed = seed_for(clip.dataset, clip.name, cond)
                if args.experiment == "noise":
                    snr = None if cond == "clean" else float(cond[3:])
                    noisy = add_white_noise(wave, sr, snr, seed)
                else:
                    rng = np.random.default_rng(seed)
                    noisy = mix_with_accompaniment(wave, accompaniment, float(cond[3:]), rng)

                mix16 = prepare_pitch_input(noisy, sr)
                vocals16 = separator.separate(noisy, sr)["vocals"]
                rows = []
                for src, sig in (("crepe", mix16), ("sep", vocals16)):
                    res = evaluate_signal(clip, sig, tracker, det_h, det_p, clip.notes is not None)
                    for tag, gates in res.items():
                        for gate, m in gates.items():
                            rows.append({"system": f"{src}{tag}:{gate}", **m})
                for r in rows:
                    out.write(json.dumps({"dataset": clip.dataset, "clip": clip.name, "singer": clip.singer,
                                          "condition": cond, **r}) + "\n")
                out.flush()
            el = time.time() - t0
            print(f"[{n_done}/{n_total}] {clip.name} elapsed {el/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
