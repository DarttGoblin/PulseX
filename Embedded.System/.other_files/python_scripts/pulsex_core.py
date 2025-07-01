# pulsex_core.py

import torch
import torchaudio
import torchvision.models as models
import os

# === Constants ===
PCF8591_ADDR = 0x48
CHANNEL = 0
SAMPLE_RATE = 1000
DURATION = 20
TOTAL_SAMPLES = SAMPLE_RATE * DURATION

# === Model Definitions ===
class BinaryResNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = models.resnet18(weights=None)
        self.cnn.conv1 = torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.cnn.fc = torch.nn.Identity()
        self.demographic = torch.nn.Linear(4, 32)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(512 + 32, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 2)
        )

    def forward(self, x, demo):
        cnn_feat = self.cnn(x)
        demo_feat = self.demographic(demo)
        combined = torch.cat([cnn_feat, demo_feat], dim=1)
        return self.classifier(combined)

class MultiClassResNet(torch.nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.cnn = models.resnet18(weights=None)
        self.cnn.conv1 = torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.cnn.fc = torch.nn.Identity()
        self.demographic = torch.nn.Linear(4, 32)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(512 + 32, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, num_classes)
        )

    def forward(self, x, demo):
        cnn_feat = self.cnn(x)
        demo_feat = self.demographic(demo)
        combined = torch.cat([cnn_feat, demo_feat], dim=1)
        return self.classifier(combined)

# === Preprocessing Function ===
def preprocess_audio(filepath):
    waveform, sr = torchaudio.load(filepath)

    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        waveform = resampler(waveform)

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )
    db_transform = torchaudio.transforms.AmplitudeToDB(stype='power')

    mel_spec = mel_transform(waveform)
    mel_spec = db_transform(mel_spec)
    mel_spec = mel_spec.expand(3, -1, -1)

    frame_duration = 512 / 16000
    target_duration = 20.0
    MAX_LEN = int(target_duration / frame_duration)

    if mel_spec.shape[2] < MAX_LEN:
        pad_width = MAX_LEN - mel_spec.shape[2]
        mel_spec = torch.nn.functional.pad(mel_spec, (0, pad_width))
    else:
        mel_spec = mel_spec[:, :, :MAX_LEN]

    return mel_spec.unsqueeze(0)

# === Utility: Generate short ID for files ===
def generate_next_id(folder="Audios"):
    existing = [f for f in os.listdir(folder) if f.startswith("heartbeat_") and f.endswith(".wav")]
    ids = [int(f.split("_")[1].split(".")[0]) for f in existing if f.split("_")[1].split(".")[0].isdigit()]
    return max(ids) + 1 if ids else 1
