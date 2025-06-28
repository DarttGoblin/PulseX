import torch
import torchaudio
import torchvision.models as models
import pandas as pd

# ==== Load trained model ====
class ResNetWithDemographicsMulti(torch.nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.cnn = models.resnet18(weights=None)
        self.cnn.conv1 = torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.cnn.fc = torch.nn.Identity()

        self.demographic = torch.nn.Linear(4, 32)

        # MATCHING TRAINING CLASSIFIER:
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

# ==== Device ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== Load model ====
NUM_CLASSES = 5  # <== Put your actual number of classes here
model = ResNetWithDemographicsMulti(num_classes=NUM_CLASSES).to(device)
model.load_state_dict(torch.load("best_multiclass_model.pth", map_location=device))
model.eval()

# ==== Preprocess audio ====
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

# ==== Example Demographics ====
example_demo = torch.tensor([[ 60/100.0,
                               1.0,
                               0.0,
                               1.0
                            ]], dtype=torch.float)

# ==== Inference ====
wav_path = "mr.wav"  # <<-- your test wav file (20 sec)

mel = preprocess_audio(wav_path).to(device)
demo = example_demo.to(device)

with torch.no_grad():
    logits = model(mel, demo)
    probs = torch.nn.functional.softmax(logits, dim=1)
    pred_class = torch.argmax(probs, dim=1).cpu().item()

# ==== Output ====
# You can customize these class names:
class_names = ["Aortic Regurgitation", "Aortic Stenosis", "Mitral Regurgitation", "Mitral Stenosis", "Multi Disease"]

print("\n✅ Multi-class prediction:")
print(f"  → Predicted Class: {class_names[pred_class]} (class_id={pred_class})")
