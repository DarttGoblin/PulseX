import smbus2
import time
import numpy as np
import scipy.io.wavfile as wav
from datetime import datetime
import os
import RPi.GPIO as GPIO
import torch
import torchaudio
import torchvision.models as models
import pandas as pd

# === CONFIGURATION ===
PCF8591_ADDR = 0x48
CHANNEL = 0
SAMPLE_RATE = 1000
DURATION = 20
TOTAL_SAMPLES = SAMPLE_RATE * DURATION

RED_LED_PIN = 26
BLUE_LED_PIN = 19
GREEN_LED_PIN = 13
YELLOW_LED_PIN = 6
BUTTON_PIN = 21

# === INIT ===
bus = smbus2.SMBus(1)

def read_adc(channel=0):
    assert 0 <= channel <= 3, "Invalid ADC channel!"
    bus.write_byte(PCF8591_ADDR, 0x40 | channel)
    bus.read_byte(PCF8591_ADDR)
    value = bus.read_byte(PCF8591_ADDR)
    return value

GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(GREEN_LED_PIN, GPIO.OUT)
GPIO.setup(YELLOW_LED_PIN, GPIO.OUT)
GPIO.setup(RED_LED_PIN, GPIO.OUT)
GPIO.setup(BLUE_LED_PIN, GPIO.OUT)

# Set all LEDs OFF initially
GPIO.output(GREEN_LED_PIN, GPIO.LOW)
GPIO.output(YELLOW_LED_PIN, GPIO.LOW)
GPIO.output(RED_LED_PIN, GPIO.LOW)
GPIO.output(BLUE_LED_PIN, GPIO.LOW)

# Create Audios folder if not exists
os.makedirs('Audios', exist_ok=True)

print("=== PulseX System Ready ===")

try:
    while True:
        # Clear old state at start of each cycle
        GPIO.output(GREEN_LED_PIN, GPIO.LOW)
        GPIO.output(YELLOW_LED_PIN, GPIO.LOW)
        GPIO.output(RED_LED_PIN, GPIO.LOW)
        GPIO.output(BLUE_LED_PIN, GPIO.LOW)

        # === WAIT FOR BUTTON PRESS ===
        print("🔘 Waiting for button press to start recording...")
        GPIO.wait_for_edge(BUTTON_PIN, GPIO.FALLING, bouncetime=200)
        print("🎙️ Button pressed! Starting recording...")
        GPIO.output(GREEN_LED_PIN, GPIO.HIGH)

        # Wait until button released
        while GPIO.input(BUTTON_PIN) == GPIO.LOW:
            time.sleep(0.1)

        # === RECORDING ===
        samples = []
        delay = 1.0 / SAMPLE_RATE
        start_time = time.time()

        for i in range(TOTAL_SAMPLES):
            value = read_adc(CHANNEL)
            normalized = (value / 127.5) - 1.0
            samples.append(normalized)

            target_time = start_time + (i + 1) * delay
            now = time.time()
            while now < target_time:
                now = time.time()

            if i % SAMPLE_RATE == 0:
                sec_passed = i // SAMPLE_RATE
                print(f"... {sec_passed}/{DURATION} seconds")

        print("✅ Recording finished.")

        samples_np = np.array(samples, dtype=np.float32)

        # === Generate short numeric ID ===
        existing = [f for f in os.listdir('Audios') if f.startswith("heartbeat_") and f.endswith(".wav")]
        ids = [int(f.split("_")[1].split(".")[0]) for f in existing if f.split("_")[1].split(".")[0].isdigit()]
        next_id = max(ids) + 1 if ids else 1

        wav_filename = f"Audios/heartbeat_{next_id}.wav"
        wav.write(wav_filename, SAMPLE_RATE, (samples_np * 32767).astype(np.int16))
        print(f"💾 WAV saved: {wav_filename}")

        txt_filename = f"Audios/heartbeat_{next_id}.txt"
        np.savetxt(txt_filename, samples_np, delimiter=',')
        print(f"📄 TXT saved: {txt_filename}")


        # === MODEL PART ===
        print("📦 Loading model...")

        class ResNetWithDemographics(torch.nn.Module):
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

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ResNetWithDemographics().to(device)
        model.load_state_dict(torch.load("Models/binary_label_model.pth", map_location=device))
        model.eval()

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

        example_demo = torch.tensor([[22/100.0, 1.0, 0.0, 1.0]], dtype=torch.float)

        print("🧐 Preprocessing audio...")
        mel = preprocess_audio(wav_filename).to(device)
        demo = example_demo.to(device)

        print("🤖 Running inference...")
        with torch.no_grad():
            logits = model(mel, demo)
            pred = torch.argmax(logits, dim=1).item()

        if pred == 1:
            print("✅ Prediction: NORMAL heartbeat sound")
            GPIO.output(BLUE_LED_PIN, GPIO.HIGH)
        else:
            print("⚠️ Prediction: ABNORMAL heartbeat sound")
            GPIO.output(RED_LED_PIN, GPIO.HIGH)

        print("✅ All tasks completed successfully.")
        print("🔄 Waiting for next recording... (Press button again)")

except KeyboardInterrupt:
    print("\n⚠️ Interrupted by user (Ctrl+C).")
    GPIO.output(YELLOW_LED_PIN, GPIO.HIGH)
    print("=== Program stopped ===")

except Exception as e:
    print(f"\n⚠️ ERROR: {e}")
    GPIO.output(YELLOW_LED_PIN, GPIO.HIGH)

finally:
    print("=== PulseX ended ===")
    GPIO.cleanup()