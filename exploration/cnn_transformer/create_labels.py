"""
Convert MIDI files to piano roll format matching your dataset requirements.
Creates label files with shape (time_frames, 88) for piano transcription.
"""
import numpy as np
import pretty_midi
import librosa


def midi_to_piano_roll(midi_path, audio_path, hop_length=512, sr=22050):
    """
    Convert MIDI file to piano roll aligned with audio features.
    
    Args:
        midi_path: Path to MIDI file (.mid, .midi)
        audio_path: Path to corresponding audio file (for duration/alignment)
        hop_length: Hop length used in feature extraction (default: 512)
        sr: Sample rate (default: 22050)
    
    Returns:
        piano_roll: numpy array of shape (time_frames, 88), dtype=float32
            - Each row = one time frame
            - Each column = one piano key (A0=0, C8=87)
            - Values: 1.0 if key pressed, 0.0 otherwise
    """
    # Load MIDI file
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    
    # Load audio to get duration and calculate number of frames
    audio, _ = librosa.load(audio_path, sr=sr, mono=True)
    audio_duration = len(audio) / sr
    n_frames = int(np.ceil(audio_duration * sr / hop_length))
    
    # Initialize piano roll: (time_frames, 88)
    piano_roll = np.zeros((n_frames, 88), dtype=np.float32)
    
    # Process all instruments
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            # Convert MIDI note number to piano key index
            # Piano range: A0 (MIDI 21) to C8 (MIDI 108) = 88 keys
            midi_note = note.pitch
            
            # Check if note is within piano range
            if 21 <= midi_note <= 108:
                key_index = midi_note - 21  # Map to 0-87
                
                # Convert note times to frame indices
                start_frame = int(note.start * sr / hop_length)
                end_frame = int(note.end * sr / hop_length)
                
                # Ensure indices are within bounds
                start_frame = max(0, min(start_frame, n_frames - 1))
                end_frame = max(0, min(end_frame, n_frames))
                
                # Set piano roll values (note is active during this time)
                piano_roll[start_frame:end_frame, key_index] = 1.0
    
    return piano_roll


def create_labels_for_dataset(audio_dir, midi_dir, output_dir, hop_length=512, sr=22050):
    """
    Process a directory of audio-MIDI pairs into piano roll labels.
    
    Args:
        audio_dir: Directory containing WAV/MP3 files
        midi_dir: Directory containing corresponding MIDI files
        output_dir: Directory to save label .npy files
        hop_length: Hop length for feature extraction
        sr: Sample rate
    """
    from pathlib import Path
    
    audio_dir = Path(audio_dir)
    midi_dir = Path(midi_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all audio files
    audio_files = list(audio_dir.glob("*.wav")) + list(audio_dir.glob("*.mp3"))
    
    label_paths = []
    
    for audio_path in sorted(audio_files):
        # Find corresponding MIDI file
        midi_path = midi_dir / (audio_path.stem + ".mid")
        
        if not midi_path.exists():
            # Try alternative extensions
            for ext in [".midi", ".MID", ".MIDI"]:
                alt_path = midi_dir / (audio_path.stem + ext)
                if alt_path.exists():
                    midi_path = alt_path
                    break
            else:
                print(f"⚠️  Warning: No MIDI file found for {audio_path.name}")
                continue
        
        try:
            print(f"Processing {audio_path.name}...")
            piano_roll = midi_to_piano_roll(str(midi_path), str(audio_path), 
                                           hop_length=hop_length, sr=sr)
            
            # Save piano roll
            output_path = output_dir / (audio_path.stem + "_labels.npy")
            np.save(output_path, piano_roll)
            label_paths.append(str(output_path))
            
            print(f"  ✓ Saved: {output_path.name} (shape: {piano_roll.shape})")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    print(f"\n✓ Created {len(label_paths)} label files in {output_dir}")
    return label_paths


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 4:
        print("Usage: python create_labels.py <audio_dir> <midi_dir> <output_dir>")
        print("\nExample:")
        print("  python create_labels.py ./data/audio ./data/midi ./data/labels")
        print("\nThis creates piano roll label files (.npy) with shape (time_frames, 88)")
        sys.exit(1)
    
    audio_dir = sys.argv[1]
    midi_dir = sys.argv[2]
    output_dir = sys.argv[3]
    
    create_labels_for_dataset(audio_dir, midi_dir, output_dir)

