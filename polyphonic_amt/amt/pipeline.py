"""End-to-end modular pipeline: Demucs -> {torchcrepe, voicing mask} -> gating -> MIDI."""
from dataclasses import dataclass

import numpy as np

from .audio import load_audio, prepare_pitch_input
from .config import (
    CREPE_CONFIDENCE_THRESH,
    DEVICE,
    STEM_F0_RANGE,
    TARGET_STEMS,
)
from .gating import apply_gating
from .midi import build_midi, frames_to_notes
from .pitch import PitchTracker
from .separator import AudioSeparator
from .voicing import VoicingDetector


@dataclass
class StemResult:
    f0: np.ndarray  # raw torchcrepe f0 (Hz)
    periodicity: np.ndarray
    mask: np.ndarray
    gated_f0: np.ndarray
    notes: list


class TranscriptionPipeline:
    def __init__(self, device: str = DEVICE, mask_mode: str = "harmonic"):
        self.separator = AudioSeparator(device=device)
        self.tracker = PitchTracker(device=device)
        self.voicing = VoicingDetector(mode=mask_mode)

    def process_stem(self, stem: np.ndarray, name: str) -> StemResult:
        fmin, fmax = STEM_F0_RANGE[name]
        f0, per = self.tracker.predict(stem, fmin, fmax)  # Branch A
        mask = self.voicing.mask(stem)  # Branch B
        gated = apply_gating(f0, per, mask, CREPE_CONFIDENCE_THRESH)  # fusion
        return StemResult(f0, per, mask, gated, frames_to_notes(gated))

    def transcribe(self, audio_path: str, output_midi: str | None = None, stems=TARGET_STEMS):
        wave, sr = load_audio(audio_path)
        separated = self.separator.separate(wave, sr)
        results = {s: self.process_stem(separated[s], s) for s in stems}
        pm = build_midi({s: r.notes for s, r in results.items()})
        if output_midi:
            pm.write(output_midi)
        return pm, results
