import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import glob
import librosa
import pandas as pd
import matplotlib.pyplot as plt


def compute_log_mel_with_deltas(
    waveform: np.ndarray,
    sample_rate: int,
    n_mels: int = 128,
    hop_length: int = 512,
    n_fft: int = 2048,
) -> np.ndarray:
    """
    Compute a 3-channel input: [log-mel, delta(log-mel), delta-delta(log-mel)]
    Returns shape (3, n_mels=128, time_frames).
    """
    mel = librosa.feature.melspectrogram(
        y=waveform, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    delta_1 = librosa.feature.delta(log_mel)
    delta_2 = librosa.feature.delta(log_mel, order=2)
    stacked = np.stack([log_mel, delta_1, delta_2], axis=0)
    return stacked.astype(np.float32)


def pad_and_collate(batch):
    """
    Pad variable-length feature/label tensors in the time dimension and stack into a batch.
    """
    features, labels = zip(*batch)
    # Cap excessively long sequences to avoid OOM
    MAX_TIME_FRAMES = 8000
    max_time = min(max(f.shape[-1] for f in features), MAX_TIME_FRAMES)

    padded_features = []
    padded_labels = []

    for feat, lab in zip(features, labels):
        # If longer than cap, take the last max_time frames (or could random-crop)
        if feat.shape[-1] > max_time:
            start = feat.shape[-1] - max_time
            feat = feat[:, :, start:]
        if lab.shape[0] > max_time:
            start_l = lab.shape[0] - max_time
            lab = lab[start_l:, :]

        pad_frames_feat = max_time - feat.shape[-1]
        if pad_frames_feat > 0:
            feat = F.pad(feat, (0, pad_frames_feat, 0, 0, 0, 0))

        pad_frames_lab = max_time - lab.shape[0]
        if pad_frames_lab > 0:
            lab = F.pad(lab, (0, 0, 0, pad_frames_lab))

        padded_features.append(feat)
        padded_labels.append(lab)

    batch_features = torch.stack(padded_features, dim=0)
    batch_labels = torch.stack(padded_labels, dim=0)
    return batch_features, batch_labels


# Model Definition
class CNNTransformerAMT(nn.Module):
    def __init__(
        self,
        input_channels=3,
        nhead=8,
        num_layers=3,
        dim_feedforward=512,
        num_classes=88,
    ):
        super().__init__()
        # CNN Backbone
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        # Pool only along frequency, keep time unchanged
        self.pool = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # After two (2x1) pools along frequency only
        self.freq_bins_after_cnn = 128 // 4

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64 * self.freq_bins_after_cnn,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            activation="relu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.fc_out = nn.Linear(64 * self.freq_bins_after_cnn, num_classes)

        # Initialize weights for better training stability
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Xavier/Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x shape: (batch_size, channels=3, freq_bins=128, time_frames)

        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # (batch, 64, freq/4, time/4)

        batch_size, channels, freq, time = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(batch_size, time, channels * freq)

        x = self.transformer_encoder(x)

        x = self.fc_out(x)  # (batch, time, num_classes)
        # Apply sigmoid to constrain outputs to [0, 1] range matching normalized targets
        # This is appropriate for MSE loss with normalized continuous targets
        x = torch.sigmoid(x)

        return x


# Dataset Class
class MusicTranscriptionDataset(Dataset):
    def __init__(self, audio_paths, label_paths, sample_rate: int = 22050):
        # Align by basename (without extension)
        basename_to_audio = {
            os.path.splitext(os.path.basename(p))[0]: p for p in audio_paths
        }
        basename_to_label = {
            os.path.splitext(os.path.basename(p))[0]: p for p in label_paths
        }

        common_keys = sorted(
            set(basename_to_audio.keys()) & set(basename_to_label.keys())
        )
        self.items = [(basename_to_audio[k], basename_to_label[k]) for k in common_keys]
        self.sample_rate = sample_rate
        self.note_value_to_idx = self._build_note_value_mapping(
            [label_path for _, label_path in self.items]
        )

        if len(self.items) == 0:
            raise ValueError(
                "No matching audio (.wav) and label (.csv) files found. "
                "Ensure filenames match between train_data and train_labels."
            )

    def __len__(self):
        return len(self.items)

    def _build_note_value_mapping(self, label_paths):
        mapping = {}
        for path in label_paths:
            df = pd.read_csv(path, usecols=["note_value"])
            for value in df["note_value"].dropna().unique():
                if value not in mapping:
                    mapping[value] = len(mapping)
        return mapping

    def __getitem__(self, idx):
        audio_path, label_path = self.items[idx]

        # Load audio
        waveform, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)

        # Compute features (3, 128, T_spec)
        features = compute_log_mel_with_deltas(waveform, sr)  # float32

        # Load labels CSV with mixed data types
        df = pd.read_csv(label_path)
        numeric_cols = [
            "start_time",
            "end_time",
            "instrument",
            "note",
            "start_beat",
            "end_beat",
        ]
        numeric_data = df[numeric_cols].to_numpy(dtype=np.float32)
        note_value_encoded = (
            df["note_value"].map(self.note_value_to_idx).to_numpy(dtype=np.float32)
        )
        labels = np.column_stack([numeric_data, note_value_encoded])

        # Normalize label columns to [0, 1] to stabilize MSE scale
        # 0: start_time, 1: end_time -> scale by max end_time in file
        max_time_val = max(1.0, float(np.max(labels[:, 1])))
        labels[:, 0] = labels[:, 0] / max_time_val
        labels[:, 1] = labels[:, 1] / max_time_val

        # 2: instrument -> rough scale to [0,1] using 128 as upper bound
        labels[:, 2] = labels[:, 2] / 128.0

        # 3: note (MIDI) -> [0,1] by /127
        labels[:, 3] = labels[:, 3] / 127.0

        # 4: start_beat, 5: end_beat -> scale by max end_beat in file
        max_beat_val = max(1.0, float(np.max(labels[:, 5])))
        labels[:, 4] = labels[:, 4] / max_beat_val
        labels[:, 5] = labels[:, 5] / max_beat_val

        # 6: note_value (encoded category) -> [0,1] by / (num_classes-1)
        if len(self.note_value_to_idx) > 1:
            labels[:, 6] = labels[:, 6] / float(len(self.note_value_to_idx) - 1)
        else:
            labels[:, 6] = 0.0

        # Clamp all labels to [0, 1] range to ensure they're valid for BCEWithLogitsLoss
        # This handles edge cases like floating point precision issues or values slightly outside range
        labels = np.clip(labels, 0.0, 1.0)

        # Align time dimension: trim or pad labels to match spectrogram time frames
        time_frames = features.shape[-1]
        if labels.shape[0] < time_frames:
            pad_frames = time_frames - labels.shape[0]
            pad = np.zeros((pad_frames, labels.shape[1]), dtype=labels.dtype)
            labels = np.concatenate([labels, pad], axis=0)
        elif labels.shape[0] > time_frames:
            labels = labels[:time_frames, :]

        return torch.tensor(features, dtype=torch.float32), torch.tensor(
            labels, dtype=torch.float32
        )


# Prepare dataset and DataLoaders from folders:
# Expected structure:
#   ./train_data   -> *.wav audio files
#   ./train_labels -> *.csv label files (7 columns), same basenames
audio_dir = "./train_data"
label_dir = "./train_labels"
audio_file_list = sorted(glob.glob(os.path.join(audio_dir, "*.wav")))
label_file_list = sorted(glob.glob(os.path.join(label_dir, "*.csv")))

dataset = MusicTranscriptionDataset(audio_file_list, label_file_list)
dataset_size = len(dataset)
train_size = int(0.8 * dataset_size)
val_size = dataset_size - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_dataloader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    drop_last=True,
    collate_fn=pad_and_collate,
)
val_dataloader = DataLoader(
    val_dataset, batch_size=2, shuffle=False, collate_fn=pad_and_collate
)

# Training Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNTransformerAMT(num_classes=7)
model.to(device)

# Use MSELoss for regression of continuous [0, 1] normalized targets
# This is appropriate since we're predicting continuous values, not binary classifications
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)  # Increased LR
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=5, verbose=True  # More patience
)

# Mixed precision training to reduce memory
scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

best_val_loss = float("inf")
checkpoint_dir = "./checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

# Track loss history for visualization
train_losses = []
val_losses = []

num_epochs = 10

# Set up interactive plotting
plt.ion()
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Training and Validation Loss Curves")
ax.grid(True, alpha=0.3)
(train_line,) = ax.plot([], [], label="Train Loss", marker="o")
(val_line,) = ax.plot([], [], label="Validation Loss", marker="s")
ax.legend()
plt.tight_layout()
plt.show(block=False)

# Training & Validation Loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    epoch_grad_norms = []

    for inputs, targets in train_dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        # Fix autocast deprecation warning
        autocast_device = "cuda" if device.type == "cuda" else "cpu"
        with torch.amp.autocast(
            device_type=autocast_device, enabled=(device.type == "cuda")
        ):
            outputs = model(inputs)
            outputs_flat = outputs.view(-1, outputs.shape[-1])
            targets_flat = targets.view(-1, targets.shape[-1])
            loss = criterion(outputs_flat, targets_flat)
        scaler.scale(loss).backward()

        # Gradient clipping to prevent exploding gradients
        scaler.unscale_(optimizer)

        # Track gradient norm before clipping
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1.0 / 2)
        epoch_grad_norms.append(total_norm)

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_dataloader)
    avg_grad_norm = np.mean(epoch_grad_norms) if epoch_grad_norms else 0.0
    train_losses.append(avg_loss)

    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for val_inputs, val_targets in val_dataloader:
            val_inputs = val_inputs.to(device)
            val_targets = val_targets.to(device)

            with torch.amp.autocast(
                device_type=autocast_device, enabled=(device.type == "cuda")
            ):
                val_outputs = model(val_inputs)
            val_outputs_flat = val_outputs.view(-1, val_outputs.shape[-1])
            val_targets_flat = val_targets.view(-1, val_targets.shape[-1])

            val_loss = criterion(val_outputs_flat, val_targets_flat)
            running_val_loss += val_loss.item()

    avg_val_loss = running_val_loss / len(val_dataloader)
    val_losses.append(avg_val_loss)

    print(
        f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {avg_loss:.4f} Val Loss: {avg_val_loss:.4f} | Grad Norm: {avg_grad_norm:.6f} | LR: {optimizer.param_groups[0]['lr']:.6e}"
    )

    # Update plot after each epoch
    epochs = list(range(1, len(train_losses) + 1))
    train_line.set_xdata(epochs)
    train_line.set_ydata(train_losses)
    val_line.set_xdata(epochs)
    val_line.set_ydata(val_losses)
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw()
    fig.canvas.flush_events()

    scheduler.step(avg_val_loss)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        checkpoint_path = os.path.join(
            checkpoint_dir, f"best_model_epoch_{epoch+1}.pth"
        )
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch+1}")

# Save final plot
plt.savefig(
    os.path.join(checkpoint_dir, "loss_curves.png"), dpi=150, bbox_inches="tight"
)
print(f"\nLoss curves saved to {os.path.join(checkpoint_dir, 'loss_curves.png')}")
plt.ioff()  # Turn off interactive mode
plt.show()  # Keep plot window open
