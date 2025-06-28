import torch

import torchaudio
import torchvision.models as models
import pandas as pd

# Set backend for torchaudio (for Windows support)
torchaudio.set_audio_backend("soundfile")

# ==== Load trained model ====
class ResNetWithDemographics(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = models.resnet18(weights=None)  # No need to download, you load your own weights
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

# ==== Device ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== Load model ====
model = ResNetWithDemographics().to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

# ==== Preprocess audio ====
def preprocess_audio(filepath):
    waveform, sr = torchaudio.load(filepath)
    
    # Convert to mono if stereo
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

    frame_duration = 512 / 16000  # 0.032 sec
    target_duration = 20.0        # 20 seconds
    MAX_LEN = int(target_duration / frame_duration)  # 625 frames

    if mel_spec.shape[2] < MAX_LEN:
        pad_width = MAX_LEN - mel_spec.shape[2]
        mel_spec = torch.nn.functional.pad(mel_spec, (0, pad_width))
    else:
        mel_spec = mel_spec[:, :, :MAX_LEN]

    return mel_spec.unsqueeze(0)  # Add batch dimension

# ==== Example Demographics ====
example_demo = torch.tensor([[ 22/100.0,   # Age (scaled)
                               0.0,        # Gender (1=Male, 0=Female)
                               0.0,        # Smoker (1=Yes, 0=No)
                               1.0         # Lives (1=Urban, 0=Rural)
                            ]], dtype=torch.float)

# ==== Inference ====
wav_path = "asmae_recording_from_stetho.wav"  # <<-- Put your test wav file here (20 sec wav)

mel = preprocess_audio(wav_path).to(device)
demo = example_demo.to(device)

with torch.no_grad():
    logits = model(mel, demo)
    pred = torch.argmax(logits, dim=1).item()

# ==== Output ====
if pred == 1:
    print("✅ Prediction: NORMAL heartbeat sound")
else: 
    print("⚠️ Prediction: ABNORMAL heartbeat sound")
