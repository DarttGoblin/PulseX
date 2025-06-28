import os
import torch
import torchaudio
import pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import WavLMModel, WavLMProcessor, AdamW
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

# ====== Setup ======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = WavLMProcessor.from_pretrained("microsoft/wavlm-base-plus")

# ====== Dataset ======
class HeartbeatDataset(Dataset):
    def __init__(self, df, processor):
        self.df = df.reset_index(drop=True)
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        waveform, sr = torchaudio.load(row["filepath"])
        inputs = self.processor(waveform.squeeze(), sampling_rate=sr, return_tensors="pt", padding=True)
        audio_tensor = inputs.input_values.squeeze(0)

        demo = torch.tensor([
            row["Age"], row["Gender"], row["Smoker"], row["Lives"]
        ], dtype=torch.float)

        label = torch.tensor(row["Healthy"], dtype=torch.long)
        return audio_tensor, demo, label

# ====== Collate Function for Padding ======
def collate_fn(batch):
    audios, demos, labels = zip(*batch)
    inputs = processor(list(audios), sampling_rate=16000, return_tensors="pt", padding=True)
    return inputs.input_values, torch.stack(demos), torch.tensor(labels)

# ====== Model ======
class AudioDemographicModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base-plus")
        self.audio_proj = nn.Linear(self.wavlm.config.hidden_size, 128)
        self.demo_proj = nn.Linear(4, 32)
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(160, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, audio, demo):
        with torch.no_grad():  # Freeze WavLM (unfreeze if fine-tuning)
            audio_feat = self.wavlm(audio).last_hidden_state.mean(dim=1)
        audio_emb = self.audio_proj(audio_feat)
        demo_emb = self.demo_proj(demo)
        combined = torch.cat([audio_emb, demo_emb], dim=1)
        return self.classifier(combined)

# ====== Data Preparation ======
def build_dataframe(audio_dir, metadata_csv):
    meta = pd.read_csv(metadata_csv)
    data = []

    for fname in os.listdir(audio_dir):
        if fname.endswith(".wav"):
            patient_num = fname.split("_")[3]  # '089' → patient_089
            patient_id = f"patient_{patient_num}"
            match = meta[meta["Patient_id"] == patient_id]
            if not match.empty:
                row = match.iloc[0].to_dict()
                row["filepath"] = os.path.join(audio_dir, fname)
                data.append(row)

    return pd.DataFrame(data)

# ====== Evaluation ======
def evaluate_model(model, val_loader):
    model.eval()
    preds, trues = [], []

    with torch.no_grad():
        for audio, demo, label in val_loader:
            audio, demo = audio.to(device), demo.to(device)
            logits = model(audio, demo)
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            true = label.numpy()
            preds.extend(pred)
            trues.extend(true)

    acc = accuracy_score(trues, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(trues, preds, average="binary", zero_division=0)
    print(f"\n📊 Validation — Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
    return acc

# ====== Run Pipeline ======
df = build_dataframe("dataset/audio", "demographic_data.csv")
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["Healthy"], random_state=42)

train_ds = HeartbeatDataset(train_df, processor)
val_ds = HeartbeatDataset(val_df, processor)
train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_ds, batch_size=8, collate_fn=collate_fn)

model = AudioDemographicModel().to(device)
optimizer = AdamW(model.parameters(), lr=1e-4)
loss_fn = nn.CrossEntropyLoss()

best_acc = 0
for epoch in range(10):
    model.train()
    total_loss = 0
    for audio, demo, label in train_loader:
        audio, demo, label = audio.to(device), demo.to(device), label.to(device)
        logits = model(audio, demo)
        loss = loss_fn(logits, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"\n🧪 Epoch {epoch+1} | Train Loss: {avg_loss:.4f}")
    acc = evaluate_model(model, val_loader)

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "best_model.pth")
        print("✅ Best model saved!")
