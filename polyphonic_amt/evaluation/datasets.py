"""Loaders for Annotated-VocalSet, vocadito and MUSDB18 accompaniment excerpts."""
import glob
import os
import subprocess
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Held-out split by singer: exploratory work (sanity checks, sensitivity runs)
# uses DEV singers only; reported numbers use the remaining singers.
VOCALSET_DEV_SINGERS = {"f2", "f6", "m3", "m8"}


@dataclass
class Clip:
    name: str
    dataset: str
    audio_path: str
    ref_times: np.ndarray
    ref_freqs: np.ndarray  # Hz, 0 = unvoiced
    singer: str = ""
    notes: np.ndarray | None = None  # [n, 3] onset_s, offset_s, pitch_hz (vocadito only)


def _read_vocalset_csv(path: str):
    df = pd.read_csv(path, skipinitialspace=True, usecols=[0, 1])
    t = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy()
    f = pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy()
    ok = np.isfinite(t) & np.isfinite(f)
    t, f = t[ok], f[ok]
    keep = np.r_[True, np.diff(t) > 0]  # mir_eval needs strictly increasing times
    return t[keep], np.maximum(f[keep], 0.0)


def list_vocalset(data_root: str, singers=None, exclude_singers=None) -> list[Clip]:
    audio = {
        os.path.splitext(os.path.basename(p))[0]: p
        for p in glob.glob(os.path.join(data_root, "raw", "vocalset_audio", "FULL", "**", "*.wav"), recursive=True)
    }
    csvs = glob.glob(
        os.path.join(data_root, "raw", "vocalset_x", "Annotated VocalSet", "raw 1", "csv", "**", "*.csv"),
        recursive=True,
    )
    clips = []
    for c in sorted(csvs):
        name = os.path.splitext(os.path.basename(c))[0]
        if name not in audio:
            continue
        singer = name.split("_")[0]
        if singers and singer not in singers:
            continue
        if exclude_singers and singer in exclude_singers:
            continue
        t, f = _read_vocalset_csv(c)
        clips.append(Clip(name, "vocalset", audio[name], t, f, singer))
    return clips


def list_vocadito(data_root: str) -> list[Clip]:
    root = os.path.join(data_root, "raw", "vocadito", "zenodo_data")
    clips = []
    for wav in sorted(glob.glob(os.path.join(root, "Audio", "*.wav"))):
        name = os.path.splitext(os.path.basename(wav))[0]
        arr = np.loadtxt(os.path.join(root, "Annotations", "F0", f"{name}_f0.csv"), delimiter=",")
        notes_path = os.path.join(root, "Annotations", "Notes", f"{name}_notesA1.csv")
        n = np.loadtxt(notes_path, delimiter=",", ndmin=2)
        notes = np.c_[n[:, 0], n[:, 0] + n[:, 2], n[:, 1]]
        clips.append(Clip(name, "vocadito", wav, arr[:, 0], np.maximum(arr[:, 1], 0.0), "", notes))
    return clips


def stratified_subset(clips: list[Clip], per_singer: int, seed: int = 0) -> list[Clip]:
    """Deterministic sample with the same number of clips for every singer."""
    rng = np.random.default_rng(seed)
    by_singer: dict[str, list[Clip]] = {}
    for c in clips:
        by_singer.setdefault(c.singer, []).append(c)
    out = []
    for s in sorted(by_singer):
        group = by_singer[s]
        idx = rng.permutation(len(group))[:per_singer]
        out += [group[i] for i in sorted(idx)]
    return out


def load_musdb_accompaniments(data_root: str, ffmpeg_exe: str, cache_dir: str, sr: int = 44100) -> list[np.ndarray]:
    """Decode every MUSDB18-7s stem file to a [2, T] accompaniment (drums + bass + other).

    Stem order inside *.stem.mp4 is mixture, drums, bass, other, vocals.
    Decoded stems are cached as .npy.
    """
    os.makedirs(cache_dir, exist_ok=True)
    out = []
    for p in sorted(glob.glob(os.path.join(data_root, "raw", "musdb7", "**", "*.stem.mp4"), recursive=True)):
        cache = os.path.join(cache_dir, os.path.basename(p).replace(".stem.mp4", ".npy"))
        if not os.path.exists(cache):
            parts = []
            for stream in (1, 2, 3):  # drums, bass, other
                raw = subprocess.run(
                    [ffmpeg_exe, "-v", "error", "-i", p, "-map", f"0:{stream}", "-f", "f32le",
                     "-ac", "2", "-ar", str(sr), "-"],
                    capture_output=True, check=True,
                ).stdout
                parts.append(np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).T)
            n = min(x.shape[1] for x in parts)
            np.save(cache, sum(x[:, :n] for x in parts).astype(np.float32))
        out.append(np.load(cache))
    return out
