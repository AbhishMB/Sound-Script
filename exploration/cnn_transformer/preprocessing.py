import librosa
import numpy as np


def extract_audio_features(audio_path, sr=22050, n_mels=128, n_mfcc=20, hop_length=512):
    # Load audio as mono
    audio, sr = librosa.load(audio_path, sr=sr, mono=True)

    # 1. Mel Spectrogram (log scale)
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=n_mels, hop_length=hop_length
    )
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

    # 2. MFCCs from Mel Spectrogram
    mfcc = librosa.feature.mfcc(S=log_mel_spec, n_mfcc=n_mfcc)

    # 3. Chroma STFT
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr, hop_length=hop_length)

    # Normalize each feature
    def normalize(feature):
        return (feature - np.mean(feature)) / np.std(feature)

    log_mel_spec_norm = normalize(log_mel_spec)
    mfcc_norm = normalize(mfcc)
    chroma_norm = normalize(chroma)

    # Align frame lengths (lowest among features)
    min_frames = min(
        log_mel_spec_norm.shape[1], mfcc_norm.shape[1], chroma_norm.shape[1]
    )
    log_mel_spec_norm = log_mel_spec_norm[:, :min_frames]
    mfcc_norm = mfcc_norm[:, :min_frames]
    chroma_norm = chroma_norm[:, :min_frames]

    # Function to pad frequency bins to max_bins
    def pad_freq(feature, max_bins):
        pad_width = max_bins - feature.shape[0]
        if pad_width > 0:
            return np.pad(feature, ((0, pad_width), (0, 0)), mode="constant")
        else:
            return feature

    # Find max frequency bins among features
    max_freq_bins = max(
        log_mel_spec_norm.shape[0], mfcc_norm.shape[0], chroma_norm.shape[0]
    )

    # Pad frequency dimension
    log_mel_spec_norm = pad_freq(log_mel_spec_norm, max_freq_bins)
    mfcc_norm = pad_freq(mfcc_norm, max_freq_bins)
    chroma_norm = pad_freq(chroma_norm, max_freq_bins)

    # Stack features as channels for CNN input: shape = (channels, freq_bins, time_frames)
    stacked_features = np.stack([log_mel_spec_norm, mfcc_norm, chroma_norm], axis=0)

    return stacked_features


# Example usage
audio_path = "your_audio_file.wav"
features = extract_audio_features(audio_path)

print("Feature shape (channels, freq_bins, time_frames):", features.shape)
