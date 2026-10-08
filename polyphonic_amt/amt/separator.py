import numpy as np
import torch
from demucs.apply import apply_model
from demucs.pretrained import get_model

from .audio import prepare_pitch_input, resample, to_stereo
from .config import DEMUCS_MODEL, DEVICE, SAMPLE_RATE


class AudioSeparator:
    """Demucs v4 (htdemucs) source separation.

    Input audio is converted to the model's native rate (44.1 kHz) and channel
    layout (stereo) before separation. Stems come back as mono 16 kHz arrays
    so they can be fed straight into the pitch and voicing branches.
    """

    def __init__(self, model_name: str = DEMUCS_MODEL, device: str = DEVICE, shifts: int = 0):
        self.device = device
        self.shifts = shifts
        self.model = get_model(model_name).to(device).eval()
        self.model_sr = self.model.samplerate
        self.sources = list(self.model.sources)

    def _apply(self, batch: torch.Tensor) -> torch.Tensor:
        """apply_model with a fallback to smaller segments if the GPU runs out of memory."""
        for segment in (None, 4.0, 2.0):
            try:
                return apply_model(
                    self.model,
                    batch.to(self.device),
                    shifts=self.shifts,
                    split=True,
                    overlap=0.25,
                    segment=segment,
                    device=self.device,
                )
            except torch.OutOfMemoryError:
                if self.device != "cuda" or segment == 2.0:
                    raise
                torch.cuda.empty_cache()
        raise RuntimeError("unreachable")

    @torch.no_grad()
    def separate_stereo(self, wave: np.ndarray, sr: int) -> dict[str, np.ndarray]:
        """[C, T] audio at `sr` -> {stem: [2, T'] float32 at the model rate}."""
        wav = torch.from_numpy(to_stereo(resample(wave, sr, self.model_sr))).float()
        # Same input normalisation the reference demucs CLI applies.
        ref = wav.mean(0)
        mean, std = ref.mean(), ref.std() + 1e-8
        wav = (wav - mean) / std
        out = self._apply(wav[None])[0]
        out = (out * std + mean).cpu().numpy()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        return {name: out[i] for i, name in enumerate(self.sources)}

    def separate(self, wave: np.ndarray, sr: int, out_sr: int = SAMPLE_RATE) -> dict[str, np.ndarray]:
        """[C, T] audio at `sr` -> {stem: 1-D mono float32 at out_sr}."""
        stems = self.separate_stereo(wave, sr)
        return {k: prepare_pitch_input(v, self.model_sr, out_sr) for k, v in stems.items()}
