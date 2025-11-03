import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import streamlit as st
import crepe
import pretty_midi
import soundfile as sf

# Streamlit UI setup
st.title("🎶 Musical Note Transcriber (CREPE + Piano Roll)")
st.markdown(
    """
    Upload a *monophonic* (single-note) WAV file (voice, guitar, or piano) 🎤🎸🎹  
    The app detects musical notes and visualizes them as a *piano-roll*.  
    You can also download the *MIDI* file for playback.
    """
)

uploaded_file = st.file_uploader("🎵 Upload your WAV file", type=["wav"])

if uploaded_file is not None:
    # Save uploaded file temporarily
    temp_path = "temp_input.wav"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    # Load audio (force mono, normalize)
    audio, sr = librosa.load(temp_path, sr=16000, mono=True)
    audio = librosa.util.normalize(audio)

    st.audio(uploaded_file, format="audio/wav")

    st.info("🎧 Running CREPE pitch tracking... (this may take a few seconds)")

    # Run CREPE model for pitch tracking
    time, frequency, confidence, activation = crepe.predict(audio, sr, viterbi=True, step_size=10)

    # Filter by confidence threshold
    conf_threshold = 0.5
    valid_idx = confidence > conf_threshold
    time = time[valid_idx]
    frequency = frequency[valid_idx]

    if len(frequency) == 0:
        st.error("❌ No confident pitch detected. Try a clearer monophonic input (≥1s long).")
    else:
        # Convert to MIDI notes
        midi_notes = librosa.hz_to_midi(frequency)

        # Build PrettyMIDI object
        pm = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano

        # Create notes
        prev_time = time[0]
        prev_note = midi_notes[0]
        duration = 0.1

        for t, n in zip(time, midi_notes):
            if abs(n - prev_note) > 0.5:  # pitch change => new note
                note = pretty_midi.Note(
                    velocity=100,
                    pitch=int(prev_note),
                    start=float(prev_time),
                    end=float(t)
                )
                instrument.notes.append(note)
                prev_time = t
                prev_note = n

        # Append last note
        if len(time) > 0:
            note = pretty_midi.Note(
                velocity=100,
                pitch=int(prev_note),
                start=float(prev_time),
                end=float(time[-1] + duration)
            )
            instrument.notes.append(note)

        pm.instruments.append(instrument)

        # Save as MIDI file
        midi_path = "output.mid"
        pm.write(midi_path)

        # After pm.write(midi_path) in app.py
    from music21 import converter, environment

    musicxml_path = "output.musicxml"
    pdf_path = "output.pdf"

    # Configure MuseScore path once; adjust for your OS/install
    us = environment.UserSettings()
    us['musicxmlPath'] = '/usr/bin/musescore3'  # or 'musescore4', 'mscore', 'C:\\Program Files\\MuseScore 4\\bin\\MuseScore4.exe'

    # MIDI → MusicXML + PDF
    score = converter.parse(midi_path)
    score.write("musicxml", fp=musicxml_path)
    try:
        score.write("musicxml.pdf", fp=pdf_path)
        pdf_ok = True
    except Exception as e:
        pdf_ok = False

    # Streamlit download buttons
    with open(musicxml_path, "rb") as f:
        st.download_button("⬇ Download MusicXML", f, file_name="transcribed_output.musicxml")

    if pdf_ok and os.path.exists(pdf_path):
        with open(pdf_path, "rb") as f:
            st.download_button("⬇ Download PDF Score", f, file_name="transcribed_output.pdf")
    else:
        st.info("Install MuseScore and set musicxmlPath for PDF export, or use the MusicXML output.")  # optional message



        st.success("✅ Pitch tracking complete! Visualizing piano-roll...")

        # --- 🎹 Piano-roll visualization ---
        piano_roll = pm.get_piano_roll(fs=100)

        if piano_roll.size == 0 or np.all(piano_roll == 0):
            st.warning("⚠ No valid notes detected in piano-roll visualization.")
        else:
            fig, ax = plt.subplots(figsize=(14, 8))  # Increased size for clarity

            # Determine note range
            note_min = int(np.min(midi_notes)) - 2
            note_max = int(np.max(midi_notes)) + 2

            # Slice piano roll
            roll_slice = piano_roll[note_min:note_max+1, :]

            # Generate note labels in the correct chromatic order
            midi_ticks = np.arange(note_min, note_max + 1)
            note_labels = [pretty_midi.note_number_to_name(n) for n in midi_ticks]

            # Reorder labels to follow 12-note cycle starting from A
            chromatic_order = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']
            ordered_labels = sorted(note_labels, key=lambda x: chromatic_order.index(x[0]))

            # Display piano roll
            librosa.display.specshow(
                roll_slice,
                x_axis="time",
                y_axis=None,
                ax=ax,
                cmap="inferno"
            )

            # Tick labels — every 3rd to avoid clutter
            tick_idx = np.arange(0, len(midi_ticks), 3)
            ax.set_yticks(tick_idx)
            ax.set_yticklabels([note_labels[i] for i in tick_idx])

            ax.set_ylabel("Musical Notes (A–G# cycle)")
            ax.set_xlabel("Time (s)")
            ax.set_title("🎼 Piano-Roll Visualization (12-Note Chromatic Order)")
            plt.tight_layout()

            st.pyplot(fig)

        # --- Download MIDI File ---
        with open(midi_path, "rb") as f:
            st.download_button("⬇ Download MIDI File", f, file_name="transcribed_output.mid")

    # Clean up
    os.remove(temp_path)

    else:
        st.info("⬆ Please upload a WAV file to start.")