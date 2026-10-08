import numpy as np
import torch
import torchcrepe

from .config import CREPE_DECODER, CREPE_MODEL_CAPACITY, DEVICE, HOP_LENGTH, SAMPLE_RATE


class PitchTracker:
    """torchcrepe wrapper (Branch A): per-frame f0 and periodicity confidence."""

    def __init__(
        self,
        model_capacity: str = CREPE_MODEL_CAPACITY,
        device: str = DEVICE,
        decoder: str = CREPE_DECODER,
        batch_size: int = 128,
    ):
        self.model_capacity = model_capacity
        self.device = device
        self.batch_size = batch_size
        self.decoder = {
            "viterbi": torchcrepe.decode.viterbi,
            "argmax": torchcrepe.decode.argmax,
        }[decoder]

    @torch.no_grad()
    def predict(self, audio: np.ndarray, fmin: float = 50.0, fmax: float = 2000.0):
        """1-D float32 audio at 16 kHz -> (f0_hz, periodicity), both [frames].

        Frame i is centred on time i * HOP_LENGTH / SAMPLE_RATE.
        """
        x = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))[None].to(self.device)
        batch = self.batch_size
        while True:
            try:
                f0, per = torchcrepe.predict(
                    x,
                    SAMPLE_RATE,
                    HOP_LENGTH,
                    fmin,
                    fmax,
                    self.model_capacity,
                    decoder=self.decoder,
                    return_periodicity=True,
                    batch_size=batch,
                    device=self.device,
                )
                break
            except torch.OutOfMemoryError:
                if self.device != "cuda" or batch <= 8:
                    raise
                torch.cuda.empty_cache()
                batch //= 2  # retry with smaller batches
        return f0[0].cpu().numpy(), per[0].cpu().numpy()
