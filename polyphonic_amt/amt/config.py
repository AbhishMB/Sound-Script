"""Central settings for the polyphonic AMT pipeline.

Values that come from the paper are marked [paper]; everything else is an
implementation choice and is documented in the README.
"""
import torch

SAMPLE_RATE = 16000  # [paper] pitch / feature branches run at 16 kHz
HOP_LENGTH = 160  # [paper] 10 ms hop shared by CREPE and the spectrogram features

# Branch A: torchcrepe
CREPE_MODEL_CAPACITY = "full"  # [paper]
CREPE_CONFIDENCE_THRESH = 0.15  # [paper] C(t) >= 0.15
CREPE_DECODER = "viterbi"

# Per-stem search ranges (Hz). CREPE covers roughly 32 Hz - 2 kHz; narrowing it
# per stem removes octave errors that cannot occur for that instrument.
STEM_F0_RANGE = {
    "vocals": (60.0, 1100.0),
    "bass": (30.0, 400.0),
    "other": (60.0, 2000.0),
    "drums": (60.0, 2000.0),
}

# Branch B: voicing mask
N_MEL_FILTERS = 256  # [paper]
N_CHROMA_BINS = 12  # [paper]
N_FFT = 2048
VOICING_PERCENTILE = 10  # [paper] only used by the "paper_literal" mask variant
CHROMA_FMIN = 100.0
CHROMA_FMAX = 4000.0
MASK_FALSE_ALARM_RATE = 0.05  # fraction of white-noise frames the mask may accept
MASK_ENERGY_FLOOR_DB = 45.0  # frames this far below the loud frames are unvoiced
MASK_ABS_FLOOR_DBFS = -65.0  # absolute silence floor (a near-silent stem has no 'loud' reference)

# Demucs
DEMUCS_MODEL = "htdemucs"  # [paper]
DEMUCS_SAMPLE_RATE = 44100

TARGET_STEMS = ["vocals", "bass", "other"]  # [paper] drums skipped for pitch
STEM_INSTRUMENT_MAP = {  # [paper] General MIDI programs
    "vocals": 52,  # Choir Aahs
    "bass": 33,  # Electric Bass (finger)
    "other": 0,  # Acoustic Grand Piano
    "drums": 118,
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
