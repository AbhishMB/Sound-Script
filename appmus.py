import os
import base64
from pathlib import Path
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import streamlit as st
import crepe
import pretty_midi
import soundfile as sf

# Optional: music21 is used for MIDI -> MusicXML (no external engraver needed for MusicXML)
try:
    from music21 import converter

    MUSIC21_AVAILABLE = True
except Exception:
    MUSIC21_AVAILABLE = False

# Page config: wide layout, calm branding
st.set_page_config(
    page_title="Muse Notation Studio",
    page_icon="🎼",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Theme-aligned CSS (Apple-inspired: SF stack, glass panels, soft neutrals, blue accent)
st.markdown(
    """
<style>
:root{
  --radius: 16px;
  --hairline: rgba(255,255,255,0.08);
  --glass-bg: rgba(17, 25, 40, 0.55);
  --glass-stroke: rgba(255,255,255,0.06);
  --bg: #0B0F14;
  --bg-card: #121826;
  --text: #E6EEF7;
  --text-dim: rgba(230,238,247,0.72);
  --accent: #007AFF; /* Apple-like blue */
  --grad-1: #0A84FF;
  --grad-2: #6E5AE6;
}
html, body, [class*="css"]  {
  background: var(--bg);
  color: var(--text);
  font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "SF Pro Display",
               "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
.block-container { padding-top: 2rem; padding-bottom: 4rem; }
section.main > div { max-width: 1200px; margin: 0 auto; }

/* Glass panel */
.glass {
  background: var(--glass-bg);
  border: 1px solid var(--glass-stroke);
  border-radius: var(--radius);
  -webkit-backdrop-filter: saturate(180%) blur(20px);
  backdrop-filter: saturate(180%) blur(20px);
  box-shadow: 0 10px 30px rgba(0,0,0,0.35);
  overflow: hidden;
  margin-bottom: 18px;
}
.card-header {
  padding: 18px 20px 0 20px;
  color: var(--text);
  font-weight: 600;
  font-size: 1.05rem;
  letter-spacing: 0.2px;
}
.card-body { padding: 12px 20px 20px 20px; }
.hairline { border-top: 1px solid var(--hairline); margin: 12px 0; }

/* Accent button styling for Streamlit buttons & downloads */
.stButton>button, .stDownloadButton>button {
  background: linear-gradient(135deg, var(--grad-1), var(--grad-2));
  color: #ffffff;
  border-radius: 999px;
  padding: 8px 16px;
  border: 0;
  font-weight: 600;
}
.stDownloadButton>button:hover, .stButton>button:hover {
  filter: brightness(1.05);
}

/* Tweak audio element */
audio { width: 100%; outline: none; border-radius: 10px; }

/* Matplotlib figure background harmonization */
figure, canvas { background: transparent !important; }

/* Small helper text */
small.helper { color: var(--text-dim); }

/* Headline */
.hero h1 {
  margin-bottom: .25rem;
  font-weight: 700;
  letter-spacing: 0.2px;
  background: linear-gradient(135deg, #E6EEF7, #A0C6FF);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
}
.hero p { color: var(--text-dim); margin-top: 0; }
</style>
""",
    unsafe_allow_html=True,
)

# Hero
st.markdown(
    """
<div class="hero">
  <h1>Muse Notation Studio</h1>
  <p>Transcribe, visualize, and export your music with clarity and polish.</p>
</div>
""",
    unsafe_allow_html=True,
)

# Accept common audio and video types
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm"}
ALL_TYPES = sorted(
    {*(e.strip(".") for e in AUDIO_EXTS), *(e.strip(".") for e in VIDEO_EXTS)}
)

# Upload card
st.markdown('<div class="glass">', unsafe_allow_html=True)
st.markdown('<div class="card-header">Upload</div>', unsafe_allow_html=True)
st.markdown('<div class="card-body">', unsafe_allow_html=True)
uploaded_file = st.file_uploader(" ", type=ALL_TYPES, label_visibility="collapsed")
st.markdown(
    '<small class="helper">Drag & drop a file or browse. Supports audio and video.</small>',
    unsafe_allow_html=True,
)
st.markdown("</div></div>", unsafe_allow_html=True)


def plot_color_wave_and_mel(y, sr):
    # Downsample waveform for rendering speed
    max_points = 1800
    N = len(y)
    if N > max_points:
        factor = int(np.ceil(N / max_points))
        y_ds = y[: (N // factor) * factor].reshape(-1, factor).mean(axis=1)
        t = np.linspace(0, N / sr, num=len(y_ds))
    else:
        y_ds = y
        t = np.arange(N) / sr

    amp = np.abs(y_ds)
    if amp.max() > 0:
        amp = amp / amp.max()

    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 2], hspace=0.32)

    # Top: colorful bar waveform
    ax1 = fig.add_subplot(gs[0])
    cmap = plt.get_cmap("plasma")
    colors = cmap(amp)
    width = (t[1] - t[0]) if len(t) > 1 else 0.01
    ax1.bar(t, y_ds, width=width, color=colors, edgecolor="none")
    ax1.set_xlim(0, t[-1] if len(t) else 0)
    ax1.set_ylabel("Amplitude", color="#c7d0dd")
    ax1.set_title("Color Waveform", color="#e6eef7")
    ax1.set_facecolor("#0b1020")
    for spine in ax1.spines.values():
        spine.set_color("#324152")
    ax1.tick_params(colors="#c7d0dd")

    # Bottom: mel spectrogram + RMS overlay
    ax2 = fig.add_subplot(gs[1])
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=96, fmax=8000, power=2.0
    )
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(
        S_dB,
        x_axis="time",
        y_axis="mel",
        sr=sr,
        hop_length=256,
        fmax=8000,
        cmap="magma",
        ax=ax2,
    )
    cbar = fig.colorbar(img, ax=ax2, format="%+2.0f dB")
    cbar.ax.tick_params(colors="#c7d0dd")
    ax2.set_title("Mel Spectrogram", color="#e6eef7")
    ax2.set_facecolor("#0b1020")
    for spine in ax2.spines.values():
        spine.set_color("#324152")
    ax2.tick_params(colors="#c7d0dd")

    # RMS overlay
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=256).flatten()
    if rms.max() > 0:
        rms_norm = rms / rms.max()
        rms_time = librosa.frames_to_time(
            np.arange(len(rms_norm)), sr=sr, hop_length=256
        )
        y_curve = rms_norm * (S_dB.shape[0] * 0.9)
        ax2.plot(rms_time, y_curve, color="white", linewidth=1.0, alpha=0.7)

    plt.tight_layout()
    return fig


if uploaded_file is not None:
    # Save raw upload
    raw_name = uploaded_file.name
    raw_ext = Path(raw_name).suffix.lower()
    raw_path = f"temp_input_raw{raw_ext}"
    with open(raw_path, "wb") as f:
        f.write(uploaded_file.read())

    # Normalize to WAV (16 kHz mono PCM16) for player + visualization + transcription
    audio_path = "temp_input.wav"
    if raw_ext in VIDEO_EXTS:
        from moviepy.editor import VideoFileClip

        with VideoFileClip(raw_path) as clip:
            if clip.audio is None:
                st.error("Uploaded video has no audio track.")
                st.stop()
            clip.audio.write_audiofile(
                audio_path, codec="pcm_s16le", fps=16000, verbose=False, logger=None
            )
    else:
        y, sr_in = librosa.load(raw_path, sr=16000, mono=True)
        sf.write(audio_path, y, 16000, subtype="PCM_16")

    # Load normalized audio
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    audio = librosa.util.normalize(audio)

    # Player + Visuals: two-column glass layout
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">Player</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-body">', unsafe_allow_html=True)
        with open(audio_path, "rb") as f:
            st.audio(f.read(), format="audio/wav")
        st.markdown("</div></div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">Visuals</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-body">', unsafe_allow_html=True)
        fig_vis = plot_color_wave_and_mel(audio, sr)
        st.pyplot(fig_vis, use_container_width=True)
        st.markdown("</div></div>", unsafe_allow_html=True)

    # Processing card
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">Transcription</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-body">', unsafe_allow_html=True)
    st.info("🎧 Running CREPE pitch tracking...")
    time, frequency, confidence, activation = crepe.predict(
        audio, sr, viterbi=True, step_size=10
    )

    # Confidence filter
    conf_threshold = 0.5
    valid_idx = confidence > conf_threshold
    time = time[valid_idx]
    frequency = frequency[valid_idx]

    if len(frequency) == 0:
        st.error(
            "❌ No confident pitch detected. Try a clearer monophonic input (≥1s long)."
        )
    else:
        # PrettyMIDI assembly
        midi_notes = librosa.hz_to_midi(frequency)
        pm = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=0)
        prev_time = float(time[0])
        prev_note = float(midi_notes[0])
        min_change = 0.5
        default_dur = 0.1

        for t, n in zip(time, midi_notes):
            t = float(t)
            n = float(n)
            if abs(n - prev_note) > min_change:
                note = pretty_midi.Note(
                    velocity=100, pitch=int(round(prev_note)), start=prev_time, end=t
                )
                instrument.notes.append(note)
                prev_time = t
                prev_note = n

        last_end = float(time[-1]) + default_dur
        note = pretty_midi.Note(
            velocity=100, pitch=int(round(prev_note)), start=prev_time, end=last_end
        )
        instrument.notes.append(note)
        pm.instruments.append(instrument)

        # Save MIDI
        midi_path = "output.mid"
        pm.write(midi_path)

        # MusicXML export
        musicxml_path = "output.musicxml"
        musicxml_ok = False
        if MUSIC21_AVAILABLE:
            try:
                score = converter.parse(midi_path)
                score.write("musicxml", fp=musicxml_path)
                musicxml_ok = os.path.exists(musicxml_path)
            except Exception:
                musicxml_ok = False
        else:
            st.info("Install music21 to enable MusicXML export: pip install music21")

        st.success("✅ Pitch tracking complete! Visualizing piano-roll...")

        # Piano roll card
        st.markdown('<div class="hairline"></div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(14, 8))
        piano_roll = pm.get_piano_roll(fs=100)
        if piano_roll.size == 0 or np.all(piano_roll == 0):
            st.warning("⚠ No valid notes detected in piano-roll visualization.")
        else:
            note_min = int(np.floor(np.min(midi_notes))) - 2
            note_max = int(np.ceil(np.max(midi_notes))) + 2
            note_min = max(0, note_min)
            note_max = min(127, note_max)
            roll_slice = piano_roll[note_min : note_max + 1, :]
            midi_ticks = np.arange(note_min, note_max + 1)
            note_labels = [pretty_midi.note_number_to_name(n) for n in midi_ticks]

            librosa.display.specshow(
                roll_slice, x_axis="time", y_axis=None, ax=ax, cmap="inferno"
            )
            tick_idx = np.arange(0, len(midi_ticks), 3)
            ax.set_yticks(tick_idx)
            ax.set_yticklabels([note_labels[i] for i in tick_idx])
            ax.set_ylabel("Musical Notes")
            ax.set_xlabel("Time (s)")
            ax.set_title("🎼 Piano-Roll Visualization")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

        # Exports card
        st.markdown('<div class="hairline"></div>', unsafe_allow_html=True)
        st.markdown("<div>", unsafe_allow_html=True)
        cols = st.columns(3)
        with cols[0]:
            if os.path.exists(midi_path):
                with open(midi_path, "rb") as f:
                    st.download_button("⬇ MIDI", f, file_name="transcribed_output.mid")
        with cols[1]:
            if musicxml_ok and os.path.exists(musicxml_path):
                with open(musicxml_path, "rb") as f:
                    st.download_button(
                        "⬇ MusicXML", f, file_name="transcribed_output.musicxml"
                    )
        with cols[2]:
            st.caption(
                "PDF available when MuseScore is configured in a PDF‑enabled build."
            )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div></div>", unsafe_allow_html=True)

    # Cleanup
    for p in [raw_path, audio_path]:
        try:
            if os.path.exists(p):
                os.remove(p)
        except Exception:
            pass
else:
    # Empty state card
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">Get started</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-body">', unsafe_allow_html=True)
    st.write(
        "Upload an audio or video file to preview, analyze, and export your music."
    )
    st.markdown("</div></div>", unsafe_allow_html=True)
