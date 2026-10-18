# Sound-Script: polyphonic transcription with source separation and harmonic-gated pitch tracking

Code for the paper **"A modular approach to polyphonic music transcription using source separation and harmonic-gated pitch tracking"**, *Journal of Integrated Science and Technology*, 14(5):1582, 2026. [doi:10.62110/sciencein.jist.2026.v14.1582](https://doi.org/10.62110/sciencein.jist.2026.v14.1582)

CREPE is an excellent pitch tracker on a single voice, but on mixed or noisy audio it reports confident pitches where nothing is sounding ("ghost notes"). This pipeline breaks a polyphonic recording back down into single sources first, then keeps a pitch only where two independent signals agree: CREPE's own confidence, and a spectral check that the frame actually carries tonal energy.

## Pipeline

```
mixture.wav
   │
   ▼  Demucs v4 (htdemucs)            vocals / bass / drums / other
   │
   ├─► for each stem:
   │      light spectral-subtraction denoise
   │      Path A: torchcrepe ("full")  ──►  f0(t), confidence C(t)
   │      Path B: 256-band Mel → chroma ──►  voicing mask M(t)
   │                                         (frame energy above the 10th percentile)
   │      Fusion:  keep f0(t) only if  M(t) = 1  and  C(t) ≥ 0.15
   │      gated contour ──► MIDI notes
   │
   ▼
one MIDI file per stem + a combined multi-track MIDI
```

Everything runs at 16 kHz with a 10 ms hop (160 samples), so CREPE, the Mel spectrogram and the chroma frames line up.

## Result

As reported in the paper, on Annotated-VocalSet (solo singing) with white noise added from 40 dB down to 0 dB SNR:

| Condition | Raw pitch accuracy, CREPE baseline | Raw pitch accuracy, gated pipeline |
|---|---|---|
| Clean (40 dB SNR) | 92.1% | 96.2% |
| Noisy (10 dB SNR) | 65.0% | 82.1% |

The gap widens as noise increases. Gating trades a little recall for precision.

## Repository layout

```
hybrid_pipeline.py          the full pipeline (separation → dual path → fusion → MIDI), CLI entry point
crepe_baseline_pipeline.py  CREPE only, no denoising, mask or gating: the baseline
test_crepe_vs_hybrid.py     quick side-by-side of baseline and gated output on one stem
evaluate_pipeline.py        noise-robustness evaluation: RPA and voicing F1 vs SNR, comparison figures
experiments/
  vocalset_snr_sweep_colab.py   Colab script for the VocalSet / Annotated-VocalSet noise sweep
demo/
  app.py                    Streamlit app for the monophonic CREPE baseline (audio or video in, MIDI/MusicXML out)
exploration/cnn_transformer/   an earlier end-to-end attempt, kept for reference (see below)
```

## Quick start

Python 3.10+ is recommended. A GPU helps (Demucs and CREPE "full" are heavy) but is not required.

```bash
git clone https://github.com/AbhishMB/Sound-Script.git
cd Sound-Script
pip install -r requirements.txt

# transcribe all four stems of a song
python hybrid_pipeline.py song.wav --output_dir output

# only some stems, reusing stems that are already separated
python hybrid_pipeline.py song.wav --stems vocals,bass --no_separation

# compare baseline and gated output on the vocal stem
# (uses separated_stems/ if the song was already separated, otherwise the file itself)
python test_crepe_vs_hybrid.py song.wav vocals
```

Separated stems are written to `separated_stems/htdemucs/<song>/`; MIDI files go to `output/`.

## Reproducing the noise evaluation

```bash
# with ground truth: a CSV of time (s), f0 (Hz), two columns, no header row
python evaluate_pipeline.py vocals.wav --annotation f0.csv --output_dir figures

# without ground truth, the clean gated output is used as a pseudo-reference
python evaluate_pipeline.py vocals.wav --output_dir figures
```

For the dataset-level sweep, open `experiments/vocalset_snr_sweep_colab.py` in Colab, set `VOCALSET_TECH_ROOT` and `ANNOTATED_ROOT` to your copies of VocalSet and Annotated-VocalSet, and run it. It evaluates both systems at 40, 30, 20, 10, 5 and 0 dB SNR.

## Known limits

- **Each stem is treated as monophonic.** If a stem still contains chords (a piano or guitar), CREPE follows the dominant fundamental only. Full chordal transcription would need a multi-pitch estimator per stem.
- **Separation errors propagate.** Leakage between Demucs stems reaches the pitch tracker; the voicing mask is the only defence against it.
- **Thresholds are hand-set.** The 0.15 confidence gate and the 10th-percentile mask were chosen empirically and may need re-tuning for other styles or recording conditions.
- **Two details the paper does not spell out:** each stem gets a light spectral-subtraction denoise (noise floor estimated from the first ten frames) before pitch tracking, and the CREPE search range is limited to 50–2000 Hz.

## Demo app (monophonic baseline)

`demo/app.py` is a Streamlit front end for the CREPE baseline on single-voice recordings. It accepts audio or video, shows the waveform, spectrogram and piano roll, and exports MIDI and MusicXML. It uses the TensorFlow `crepe` package, so it has its own requirements:

```bash
pip install -r demo/requirements.txt
streamlit run demo/app.py
```

## Earlier attempt: end-to-end CNN + Transformer

`exploration/cnn_transformer/` holds the approach tried before the decomposition pipeline: a CNN front end with a Transformer encoder, trained to predict 88-key piano rolls from log-Mel features (with deltas) on MusicNet. It is kept for reference and is not part of the paper's method.

## Citation

```bibtex
@article{rashmi2026modular,
  title   = {A modular approach to polyphonic music transcription using source separation and harmonic-gated pitch tracking},
  author  = {Rashmi, M. R. and Vatsala, B. R. and Vijay, C. P. and Harshitha, H. S. and Bilagi, Abhishek Manohar and Hegde, Mahima Shripad},
  journal = {Journal of Integrated Science and Technology},
  volume  = {14},
  number  = {5},
  pages   = {1582},
  year    = {2026},
  doi     = {10.62110/sciencein.jist.2026.v14.1582}
}
```

## Acknowledgements

[CREPE](https://github.com/marl/crepe) and [torchcrepe](https://github.com/maxrmorrison/torchcrepe) for pitch estimation, [Demucs](https://github.com/facebookresearch/demucs) for source separation, and the VocalSet and Annotated-VocalSet datasets for evaluation.
