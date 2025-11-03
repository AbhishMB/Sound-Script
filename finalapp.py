import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import streamlit as st
import crepe
import pretty_midi
from music21 import converter, midi

# ------------------------ STREAMLIT CONFIG ------------------------
st.set_page_config(page_title="🎵 Musical Transcriber", layout="wide")

# Sidebar navigation
st.sidebar.title("🎼 Navigation")
page = st.sidebar.radio("Go to", ["🎶 Transcription", "🎹 MIDI Sheet Generator"])

# ------------------------ PAGE 1: TRANSCRIPTION ------------------------
if page == "🎶 Transcription":
    st.title("🎶 Musical Note Transcriber (CREPE + Piano Roll)")
    st.markdown(
        """
        Upload a monophonic (single-note) audio file (voice, guitar, or piano) 🎤🎸🎹  
        The app detects musical notes and visualizes them as a piano roll.  
        You can also download the MIDI file for playback.
        """
    )

    uploaded_file = st.file_uploader("🎵 Upload your audio file", type=["wav", "mp3"])

    if uploaded_file is not None:
        # Save uploaded file temporarily
        file_ext = uploaded_file.name.split(".")[-1]
        temp_path = f"temp_input.{file_ext}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

        # Load and normalize audio
        audio, sr = librosa.load(temp_path, sr=16000, mono=True)
        audio = librosa.util.normalize(audio)
        st.audio(uploaded_file, format=f"audio/{file_ext}")

        st.info("🎧 Running CREPE pitch tracking...")

        time, frequency, confidence, activation = crepe.predict(
            audio, sr, viterbi=True, step_size=10, model_capacity="tiny"
        )

        conf_threshold = 0.5
        valid_idx = confidence > conf_threshold
        time = time[valid_idx]
        frequency = frequency[valid_idx]

        if len(frequency) == 0:
            st.error(
                "❌ No confident pitch detected. Try a clearer monophonic input (≥1s long)."
            )
        else:
            midi_notes = librosa.hz_to_midi(frequency)

            pm = pretty_midi.PrettyMIDI()
            instrument = pretty_midi.Instrument(program=0)

            prev_time = time[0]
            prev_note = midi_notes[0]
            duration = 0.1

            for t, n in zip(time, midi_notes):
                if abs(n - prev_note) > 0.5:
                    note = pretty_midi.Note(
                        velocity=100,
                        pitch=int(prev_note),
                        start=float(prev_time),
                        end=float(t),
                    )
                    instrument.notes.append(note)
                    prev_time, prev_note = t, n

            note = pretty_midi.Note(
                velocity=100,
                pitch=int(prev_note),
                start=float(prev_time),
                end=float(time[-1] + duration),
            )
            instrument.notes.append(note)
            pm.instruments.append(instrument)

            midi_path = "output.mid"
            pm.write(midi_path)

            st.success("✅ Transcription complete! Visualizing piano roll...")

            piano_roll = pm.get_piano_roll(fs=100)
            if piano_roll.size == 0 or np.all(piano_roll == 0):
                st.warning("⚠ No valid notes detected.")
            else:
                fig, ax = plt.subplots(figsize=(14, 8))
                note_min, note_max = (
                    int(np.min(midi_notes)) - 2,
                    int(np.max(midi_notes)) + 2,
                )
                roll_slice = piano_roll[note_min : note_max + 1, :]
                midi_ticks = np.arange(note_min, note_max + 1)
                note_labels = [pretty_midi.note_number_to_name(n) for n in midi_ticks]

                librosa.display.specshow(
                    roll_slice, x_axis="time", cmap="inferno", ax=ax
                )
                tick_idx = np.arange(0, len(midi_ticks), 3)
                ax.set_yticks(tick_idx)
                ax.set_yticklabels([note_labels[i] for i in tick_idx])
                ax.set_ylabel("Notes")
                ax.set_xlabel("Time (s)")
                ax.set_title("🎼 Piano Roll Visualization")
                st.pyplot(fig)

            with open(midi_path, "rb") as f:
                st.download_button(
                    "⬇ Download MIDI File", f, file_name="transcribed_output.mid"
                )

        os.remove(temp_path)
    else:
        st.info("⬆ Please upload a WAV or MP3 file to start.")


# ------------------------ PAGE 2: MIDI SHEET GENERATOR ------------------------
elif page == "🎹 MIDI Sheet Generator":
    st.title("🎹 MIDI to Sheet Music Generator")
    st.markdown(
        """
        Upload a MIDI file, and this will render it as a **musical sheet (notation).  
        Useful for visualizing the transcription as real music notes 🎵
        """
    )

    midi_file = st.file_uploader("🎼 Upload a MIDI file", type=["mid", "midi"])

    if midi_file is not None:
        temp_midi_path = "temp_uploaded.mid"
        with open(temp_midi_path, "wb") as f:
            f.write(midi_file.read())

        try:
            score = converter.parse(temp_midi_path)
            st.success("✅ MIDI file parsed successfully!")

            # Convert to musical notation (SVG)
            st.markdown("### 🎵 Musical Notation Preview:")
            score.show("musicxml.png")  # generate image from notation
            img_path = "temp_uploaded.png"

            if os.path.exists(img_path):
                st.image(img_path, caption="Musical Sheet", use_container_width=True)
                os.remove(img_path)
        except Exception as e:
            st.error(f"❌ Error rendering MIDI: {e}")

        os.remove(temp_midi_path)
