# # -------------------------------------------------------------------------

# import os
# import numpy as np
# import librosa
# import librosa.display
# import matplotlib.pyplot as plt
# import streamlit as st
# import crepe
# import pretty_midi
# import soundfile as sf

# # Optional: music21 is used for MIDI -> MusicXML (no external engraver needed for MusicXML)
# try:
#     from music21 import converter

#     MUSIC21_AVAILABLE = True
# except Exception:
#     MUSIC21_AVAILABLE = False

# st.title("🎶 Musical Note Transcriber (CREPE + Piano Roll)")
# st.markdown(
#     """
#     Upload a monophonic WAV file (voice, guitar, or piano).
#     The app transcribes to MIDI, renders a piano-roll, and exports MusicXML.
#     """
# )

# uploaded_file = st.file_uploader("🎵 Upload your WAV file", type=["wav"])

# if uploaded_file is not None:
#     # Save uploaded file temporarily
#     temp_path = "temp_input.wav"
#     with open(temp_path, "wb") as f:
#         f.write(uploaded_file.read())

#     # Load audio (force mono, normalize)
#     audio, sr = librosa.load(temp_path, sr=16000, mono=True)
#     audio = librosa.util.normalize(audio)

#     # Audio player (kept)
#     st.audio(uploaded_file, format="audio/wav")

#     # Waveform placeholder shown immediately (before heavy processing)
#     wave_ph = st.empty()

#     # Efficient waveform plot (downsample long audio for faster UI)
#     def plot_waveform(sig, rate, max_points=20000):
#         x = sig
#         if len(x) > max_points:
#             factor = int(np.ceil(len(x) / max_points))
#             x = x[: (len(x) // factor) * factor].reshape(-1, factor).mean(axis=1)
#             t = np.linspace(0, len(sig) / rate, num=len(x))
#         else:
#             t = np.arange(len(x)) / rate

#         fig, ax = plt.subplots(figsize=(14, 3))
#         ax.plot(t, x, color="steelblue", linewidth=0.8)
#         ax.set_xlim(0, t[-1] if len(t) else 0)
#         ax.set_ylabel("Amplitude")
#         ax.set_xlabel("Time (s)")
#         ax.set_title("Waveform")
#         fig.tight_layout()
#         return fig

#     wave_fig = plot_waveform(audio, sr)
#     wave_ph.pyplot(wave_fig)

#     # Optional: show a spinner and a progress bar while processing
#     st.info("🎧 Running CREPE pitch tracking...")
#     progress = st.progress(0)

#     with st.spinner("Transcribing..."):
#         # Run CREPE model for pitch tracking
#         time, frequency, confidence, activation = crepe.predict(
#             audio, sr, viterbi=True, step_size=10
#         )
#         progress.progress(30)

#         # Filter by confidence threshold
#         conf_threshold = 0.5
#         valid_idx = confidence > conf_threshold
#         time = time[valid_idx]
#         frequency = frequency[valid_idx]

#         if len(frequency) == 0:
#             st.error(
#                 "❌ No confident pitch detected. Try a clearer monophonic input (≥1s long)."
#             )
#         else:
#             # Convert to MIDI notes
#             midi_notes = librosa.hz_to_midi(frequency)

#             # Build PrettyMIDI object
#             pm = pretty_midi.PrettyMIDI()
#             instrument = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano

#             # Create notes by splitting on pitch changes
#             prev_time = float(time[0])
#             prev_note = float(midi_notes[0])
#             min_change = 0.5
#             default_dur = 0.1

#             for t, n in zip(time, midi_notes):
#                 t = float(t)
#                 n = float(n)
#                 if abs(n - prev_note) > min_change:
#                     note = pretty_midi.Note(
#                         velocity=100,
#                         pitch=int(round(prev_note)),
#                         start=prev_time,
#                         end=t,
#                     )
#                     instrument.notes.append(note)
#                     prev_time = t
#                     prev_note = n

#             # Append last note
#             last_end = float(time[-1]) + default_dur
#             note = pretty_midi.Note(
#                 velocity=100,
#                 pitch=int(round(prev_note)),
#                 start=prev_time,
#                 end=last_end,
#             )
#             instrument.notes.append(note)

#             pm.instruments.append(instrument)

#             # Save MIDI
#             midi_path = "output.mid"
#             pm.write(midi_path)
#             progress.progress(60)

#             # MusicXML export (music21 only, no external engraver required)
#             musicxml_path = "output.musicxml"
#             musicxml_ok = False
#             if MUSIC21_AVAILABLE:
#                 try:
#                     score = converter.parse(midi_path)
#                     score.write("musicxml", fp=musicxml_path)
#                     musicxml_ok = os.path.exists(musicxml_path)
#                 except Exception:
#                     musicxml_ok = False
#             else:
#                 st.info(
#                     "Install music21 to enable MusicXML export: pip install music21"
#                 )

#             st.success("✅ Pitch tracking complete! Visualizing piano-roll...")
#             progress.progress(80)

#             # Piano-roll visualization
#             piano_roll = pm.get_piano_roll(fs=100)

#             if piano_roll.size == 0 or np.all(piano_roll == 0):
#                 st.warning("⚠ No valid notes detected in piano-roll visualization.")
#             else:
#                 fig, ax = plt.subplots(figsize=(14, 8))
#                 # Determine note range
#                 note_min = int(np.floor(np.min(midi_notes))) - 2
#                 note_max = int(np.ceil(np.max(midi_notes))) + 2
#                 note_min = max(0, note_min)
#                 note_max = min(127, note_max)

#                 # Slice piano roll
#                 roll_slice = piano_roll[note_min : note_max + 1, :]

#                 # y-axis labels
#                 midi_ticks = np.arange(note_min, note_max + 1)
#                 note_labels = [pretty_midi.note_number_to_name(n) for n in midi_ticks]

#                 librosa.display.specshow(
#                     roll_slice,
#                     x_axis="time",
#                     y_axis=None,
#                     ax=ax,
#                     cmap="inferno",
#                 )

#                 # Tick labels — every 3rd to avoid clutter
#                 tick_idx = np.arange(0, len(midi_ticks), 3)
#                 ax.set_yticks(tick_idx)
#                 ax.set_yticklabels([note_labels[i] for i in tick_idx])

#                 ax.set_ylabel("Musical Notes")
#                 ax.set_xlabel("Time (s)")
#                 ax.set_title("🎼 Piano-Roll Visualization")
#                 plt.tight_layout()
#                 st.pyplot(fig)

#             progress.progress(100)

#             # Download buttons
#             if os.path.exists(midi_path):
#                 with open(midi_path, "rb") as f:
#                     st.download_button(
#                         "⬇ Download MIDI File", f, file_name="transcribed_output.mid"
#                     )

#             if musicxml_ok and os.path.exists(musicxml_path):
#                 with open(musicxml_path, "rb") as f:
#                     st.download_button(
#                         "⬇ Download MusicXML",
#                         f,
#                         file_name="transcribed_output.musicxml",
#                     )

#     # Clean up uploaded temp file
#     try:
#         os.remove(temp_path)
#     except Exception:
#         pass
# else:
#     st.info("⬆ Please upload a WAV file to start.")


# --------------------------------------------------------------
## iteration containing static waveform
# import os
# import base64
# import numpy as np
# import librosa
# import librosa.display
# import matplotlib.pyplot as plt
# import streamlit as st
# import streamlit.components.v1 as components
# import crepe
# import pretty_midi
# import soundfile as sf

# # Optional: music21 is used for MIDI -> MusicXML (no external engraver needed for MusicXML)
# try:
#     from music21 import converter

#     MUSIC21_AVAILABLE = True
# except Exception:
#     MUSIC21_AVAILABLE = False

# st.title("🎶 Musical Note Transcriber (CREPE + Piano Roll)")
# st.markdown(
#     """
#     Upload a monophonic WAV file (voice, guitar, or piano).
#     The app transcribes to MIDI, renders a piano-roll, and exports MusicXML.
#     """
# )

# uploaded_file = st.file_uploader("🎵 Upload your WAV file", type=["wav"])

# if uploaded_file is not None:
#     # Save uploaded file temporarily
#     temp_path = "temp_input.wav"
#     with open(temp_path, "wb") as f:
#         f.write(uploaded_file.read())

#     # Load audio (force mono, normalize)
#     audio, sr = librosa.load(temp_path, sr=16000, mono=True)
#     audio = librosa.util.normalize(audio)

#     # Audio player (kept for accessibility/fallback)
#     st.audio(uploaded_file, format="audio/wav")

#     # Animated, playback-synced waveform (Wavesurfer.js via Streamlit HTML component)
#     # Build a base64 data URL for the uploaded WAV so it plays client-side
#     with open(temp_path, "rb") as f:
#         b64_audio = base64.b64encode(f.read()).decode("utf-8")

#     wavesurfer_html = f"""
#     <html>
#       <head>
#         <meta charset="utf-8" />
#         <script src="https://unpkg.com/wavesurfer.js@7"></script>
#         <style>
#           .ws-container {{ margin: 8px 0; }}
#           #waveform {{ width: 100%; height: 96px; }}
#           .controls button {{
#             margin-right: 8px; padding: 6px 10px; border: 0; border-radius: 6px;
#             background: #1f77b4; color: white; cursor: pointer;
#           }}
#         </style>
#       </head>
#       <body>
#         <div class="ws-container">
#           <div id="waveform"></div>
#           <div class="controls">
#             <button id="play">Play/Pause</button>
#             <button id="stop">Stop</button>
#           </div>
#         </div>
#         <script>
#           const wavesurfer = WaveSurfer.create({{
#             container: '#waveform',
#             waveColor: '#a3b1c6',
#             progressColor: '#1f77b4',
#             cursorColor: '#ff4b4b',
#             height: 96,
#             responsive: true,
#             normalize: true,
#             minPxPerSec: 50,
#             interact: true,
#             dragToSeek: true,
#             backend: 'MediaElement'
#           }});

#           const src = "data:audio/wav;base64,{b64_audio}";
#           wavesurfer.load(src);

#           document.getElementById('play').onclick = () => {{
#             wavesurfer.isPlaying() ? wavesurfer.pause() : wavesurfer.play();
#           }};
#           document.getElementById('stop').onclick = () => {{
#             wavesurfer.stop();
#           }};
#         </script>
#       </body>
#     </html>
#     """

#     components.html(wavesurfer_html, height=180, scrolling=False)

#     st.info("🎧 Running CREPE pitch tracking...")

#     # Run CREPE model for pitch tracking
#     time, frequency, confidence, activation = crepe.predict(
#         audio, sr, viterbi=True, step_size=10
#     )

#     # Filter by confidence threshold
#     conf_threshold = 0.5
#     valid_idx = confidence > conf_threshold
#     time = time[valid_idx]
#     frequency = frequency[valid_idx]

#     if len(frequency) == 0:
#         st.error(
#             "❌ No confident pitch detected. Try a clearer monophonic input (≥1s long)."
#         )
#     else:
#         # Convert to MIDI notes
#         midi_notes = librosa.hz_to_midi(frequency)

#         # Build PrettyMIDI object
#         pm = pretty_midi.PrettyMIDI()
#         instrument = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano

#         # Create notes by splitting on pitch changes
#         prev_time = float(time[0])
#         prev_note = float(midi_notes[0])
#         min_change = 0.5
#         default_dur = 0.1

#         for t, n in zip(time, midi_notes):
#             t = float(t)
#             n = float(n)
#             if abs(n - prev_note) > min_change:
#                 note = pretty_midi.Note(
#                     velocity=100,
#                     pitch=int(round(prev_note)),
#                     start=prev_time,
#                     end=t,
#                 )
#                 instrument.notes.append(note)
#                 prev_time = t
#                 prev_note = n

#         # Append last note
#         last_end = float(time[-1]) + default_dur
#         note = pretty_midi.Note(
#             velocity=100,
#             pitch=int(round(prev_note)),
#             start=prev_time,
#             end=last_end,
#         )
#         instrument.notes.append(note)

#         pm.instruments.append(instrument)

#         # Save MIDI
#         midi_path = "output.mid"
#         pm.write(midi_path)

#         # MusicXML export (music21 only, no external engraver required)
#         musicxml_path = "output.musicxml"
#         musicxml_ok = False
#         if MUSIC21_AVAILABLE:
#             try:
#                 score = converter.parse(midi_path)
#                 score.write("musicxml", fp=musicxml_path)
#                 musicxml_ok = os.path.exists(musicxml_path)
#             except Exception:
#                 musicxml_ok = False
#         else:
#             st.info("Install music21 to enable MusicXML export: pip install music21")

#         st.success("✅ Pitch tracking complete! Visualizing piano-roll...")

#         # Piano-roll visualization
#         piano_roll = pm.get_piano_roll(fs=100)

#         if piano_roll.size == 0 or np.all(piano_roll == 0):
#             st.warning("⚠ No valid notes detected in piano-roll visualization.")
#         else:
#             fig, ax = plt.subplots(figsize=(14, 8))
#             # Determine note range
#             note_min = int(np.floor(np.min(midi_notes))) - 2
#             note_max = int(np.ceil(np.max(midi_notes))) + 2
#             note_min = max(0, note_min)
#             note_max = min(127, note_max)

#             # Slice piano roll
#             roll_slice = piano_roll[note_min : note_max + 1, :]

#             # y-axis labels
#             midi_ticks = np.arange(note_min, note_max + 1)
#             note_labels = [pretty_midi.note_number_to_name(n) for n in midi_ticks]

#             librosa.display.specshow(
#                 roll_slice,
#                 x_axis="time",
#                 y_axis=None,
#                 ax=ax,
#                 cmap="inferno",
#             )

#             # Tick labels — every 3rd to avoid clutter
#             tick_idx = np.arange(0, len(midi_ticks), 3)
#             ax.set_yticks(tick_idx)
#             ax.set_yticklabels([note_labels[i] for i in tick_idx])

#             ax.set_ylabel("Musical Notes")
#             ax.set_xlabel("Time (s)")
#             ax.set_title("🎼 Piano-Roll Visualization")
#             plt.tight_layout()
#             st.pyplot(fig)

#         # Download buttons
#         if os.path.exists(midi_path):
#             with open(midi_path, "rb") as f:
#                 st.download_button(
#                     "⬇ Download MIDI File", f, file_name="transcribed_output.mid"
#                 )

#         if musicxml_ok and os.path.exists(musicxml_path):
#             with open(musicxml_path, "rb") as f:
#                 st.download_button(
#                     "⬇ Download MusicXML", f, file_name="transcribed_output.musicxml"
#                 )

#     # Clean up uploaded temp file
#     try:
#         os.remove(temp_path)
#     except Exception:
#         pass
# else:
#     st.info("⬆ Please upload a WAV file to start.")


# --------------------------------------------------------------

import os
import base64
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components
import crepe
import pretty_midi
import soundfile as sf

# Optional: music21 is used for MIDI -> MusicXML (no external engraver needed for MusicXML)
try:
    from music21 import converter

    MUSIC21_AVAILABLE = True
except Exception:
    MUSIC21_AVAILABLE = False

st.title("🎶 Musical Note Transcriber (CREPE + Piano Roll)")
st.markdown(
    """
    Upload a monophonic WAV file (voice, guitar, or piano).
    The app transcribes to MIDI, renders a piano-roll, and exports MusicXML.
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

    # Build a base64 data URL for the uploaded WAV so it plays client-side
    with open(temp_path, "rb") as f:
        b64_audio = base64.b64encode(f.read()).decode("utf-8")

    # DJ-style animated waveform with bars + timeline + minimap (Wavesurfer.js)
    wavesurfer_html = f"""
    <html>
      <head>
        <meta charset="utf-8" />
        <!-- Core Wavesurfer -->
        <script src="https://unpkg.com/wavesurfer.js@7"></script>
        <!-- Plugins: Timeline & Minimap -->
        <script src="https://unpkg.com/wavesurfer.js@7/plugins/timeline.min.js"></script>
        <script src="https://unpkg.com/wavesurfer.js@7/plugins/minimap.min.js"></script>
        <style>
          body {{ margin: 0; }}
          .ws-wrap {{ margin: 8px 0; }}
          #waveform {{ width: 100%; height: 120px; }}
          #timeline {{ height: 24px; font: 12px/1 sans-serif; color: #6b7280; }}
          #minimap {{ height: 32px; margin-top: 6px; }}
          .controls button {{
            margin-right: 8px; padding: 6px 10px; border: 0; border-radius: 6px;
            background: #1f77b4; color: white; cursor: pointer;
          }}
        </style>
      </head>
      <body>
        <div class="ws-wrap">
          <div id="waveform"></div>
          <div id="timeline"></div>
          <div id="minimap"></div>
          <div class="controls">
            <button id="play">Play/Pause</button>
            <button id="stop">Stop</button>
            <button id="back5">⏪ -5s</button>
            <button id="fwd5">⏩ +5s</button>
          </div>
        </div>
        <script>
          // Gradient for the progress fill
          const canvas = document.createElement('canvas');
          const ctx = canvas.getContext('2d');
          const grad = ctx.createLinearGradient(0, 0, 0, 120);
          grad.addColorStop(0, '#60a5fa');   // top
          grad.addColorStop(1, '#1d4ed8');   // bottom

          const wavesurfer = WaveSurfer.create({{
            container: '#waveform',
            height: 120,
            waveColor: '#a3b1c6',
            progressColor: grad,
            cursorColor: '#ff4b4b',
            cursorWidth: 2,
            // DJ-style bars
            barWidth: 2,
            barGap: 1,
            barRadius: 2,
            barAlign: 'bottom',
            barHeight: 1.2,
            minPxPerSec: 80,
            normalize: true,
            autoCenter: true,
            interact: true,
            dragToSeek: true
          }});

          // Plugins
          const timeline = wavesurfer.registerPlugin(WaveSurfer.Timeline.create({{
            container: '#timeline',
            height: 24
          }}));

          const minimap = wavesurfer.registerPlugin(WaveSurfer.Minimap.create({{
            container: '#minimap',
            height: 32,
            waveColor: '#d1d5db',
            progressColor: '#6b7280'
          }}));

          // Load from data URL (client-side)
          const src = "data:audio/wav;base64,{b64_audio}";
          wavesurfer.load(src);

          // Transport controls
          const playBtn = document.getElementById('play');
          const stopBtn = document.getElementById('stop');
          const back5  = document.getElementById('back5');
          const fwd5   = document.getElementById('fwd5');

          playBtn.onclick = () => {{
            wavesurfer.isPlaying() ? wavesurfer.pause() : wavesurfer.play();
          }};
          stopBtn.onclick = () => wavesurfer.stop();
          back5.onclick  = () => wavesurfer.setTime(Math.max(0, wavesurfer.getCurrentTime() - 5));
          fwd5.onclick   = () => wavesurfer.setTime(Math.min(wavesurfer.getDuration(), wavesurfer.getCurrentTime() + 5));
        </script>
      </body>
    </html>
    """

    components.html(wavesurfer_html, height=260, scrolling=False)

    st.info("🎧 Running CREPE pitch tracking...")

    # Run CREPE model for pitch tracking
    time, frequency, confidence, activation = crepe.predict(
        audio, sr, viterbi=True, step_size=10
    )

    # Filter by confidence threshold
    conf_threshold = 0.5
    valid_idx = confidence > conf_threshold
    time = time[valid_idx]
    frequency = frequency[valid_idx]

    if len(frequency) == 0:
        st.error(
            "❌ No confident pitch detected. Try a clearer monophonic input (≥1s long)."
        )
    else:
        # Convert to MIDI notes
        midi_notes = librosa.hz_to_midi(frequency)

        # Build PrettyMIDI object
        pm = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano

        # Create notes by splitting on pitch changes
        prev_time = float(time[0])
        prev_note = float(midi_notes[0])
        min_change = 0.5
        default_dur = 0.1

        for t, n in zip(time, midi_notes):
            t = float(t)
            n = float(n)
            if abs(n - prev_note) > min_change:
                note = pretty_midi.Note(
                    velocity=100,
                    pitch=int(round(prev_note)),
                    start=prev_time,
                    end=t,
                )
                instrument.notes.append(note)
                prev_time = t
                prev_note = n

        # Append last note
        last_end = float(time[-1]) + default_dur
        note = pretty_midi.Note(
            velocity=100,
            pitch=int(round(prev_note)),
            start=prev_time,
            end=last_end,
        )
        instrument.notes.append(note)

        pm.instruments.append(instrument)

        # Save MIDI
        midi_path = "output.mid"
        pm.write(midi_path)

        # MusicXML export (music21 only, no external engraver required)
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

        # Piano-roll visualization
        piano_roll = pm.get_piano_roll(fs=100)

        if piano_roll.size == 0 or np.all(piano_roll == 0):
            st.warning("⚠ No valid notes detected in piano-roll visualization.")
        else:
            fig, ax = plt.subplots(figsize=(14, 8))
            # Determine note range
            note_min = int(np.floor(np.min(midi_notes))) - 2
            note_max = int(np.ceil(np.max(midi_notes))) + 2
            note_min = max(0, note_min)
            note_max = min(127, note_max)

            # Slice piano roll
            roll_slice = piano_roll[note_min : note_max + 1, :]

            # y-axis labels
            midi_ticks = np.arange(note_min, note_max + 1)
            note_labels = [pretty_midi.note_number_to_name(n) for n in midi_ticks]

            librosa.display.specshow(
                roll_slice,
                x_axis="time",
                y_axis=None,
                ax=ax,
                cmap="inferno",
            )

            # Tick labels — every 3rd to avoid clutter
            tick_idx = np.arange(0, len(midi_ticks), 3)
            ax.set_yticks(tick_idx)
            ax.set_yticklabels([note_labels[i] for i in tick_idx])

            ax.set_ylabel("Musical Notes")
            ax.set_xlabel("Time (s)")
            ax.set_title("🎼 Piano-Roll Visualization")
            plt.tight_layout()
            st.pyplot(fig)

        # Download buttons
        if os.path.exists(midi_path):
            with open(midi_path, "rb") as f:
                st.download_button(
                    "⬇ Download MIDI File", f, file_name="transcribed_output.mid"
                )

        if musicxml_ok and os.path.exists(musicxml_path):
            with open(musicxml_path, "rb") as f:
                st.download_button(
                    "⬇ Download MusicXML", f, file_name="transcribed_output.musicxml"
                )

    # Clean up uploaded temp file
    try:
        os.remove(temp_path)
    except Exception:
        pass
else:
    st.info("⬆ Please upload a WAV file to start.")
