# pulsex_main.py

import smbus2
import time
import numpy as np
import scipy.io.wavfile as wav
from datetime import datetime
import RPi.GPIO as GPIO
import torch
import os

from pulsex_core import (
    PCF8591_ADDR, CHANNEL, SAMPLE_RATE, DURATION, TOTAL_SAMPLES,
    BinaryResNet, MultiClassResNet, preprocess_audio, generate_next_id
)

# === GPIO CONFIG ===
RED_LED_PIN = 26
BLUE_LED_PIN = 19
GREEN_LED_PIN = 13
YELLOW_LED_PIN = 6
WHITE_LED_PIN = 16 
STETHO_BUTTON_PIN = 21 
ECG_BUTTON_PIN = 12 

bus = smbus2.SMBus(1)

def read_adc(channel=0):
    assert 0 <= channel <= 3, "Invalid ADC channel!"
    bus.write_byte(PCF8591_ADDR, 0x40 | channel)
    bus.read_byte(PCF8591_ADDR)
    return bus.read_byte(PCF8591_ADDR)

GPIO.setmode(GPIO.BCM)

# Input pin (button) with pull-up resistor
GPIO.setup(STETHO_BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# Output pins (LEDs)
for pin in [WHITE_LED_PIN, YELLOW_LED_PIN, GREEN_LED_PIN, BLUE_LED_PIN, RED_LED_PIN]:
    GPIO.setup(pin, GPIO.OUT)

for led in [WHITE_LED_PIN, YELLOW_LED_PIN, GREEN_LED_PIN, BLUE_LED_PIN, RED_LED_PIN]:
    GPIO.output(led, GPIO.LOW)

os.makedirs("Audios", exist_ok=True)
print("=== PulseX System Ready ===")
GPIO.output(WHITE_LED_PIN, GPIO.HIGH)

# === Load models ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
binary_model = BinaryResNet().to(device)
binary_model.load_state_dict(torch.load("Models/binary_label_model.pth", map_location=device))
binary_model.eval()

multi_model = MultiClassResNet(num_classes=5).to(device)
multi_model.load_state_dict(torch.load("Models/multi_label_model.pth", map_location=device))
multi_model.eval()

try:
    while True:
        print("🔘 Waiting for button press to start recording...")
        GPIO.wait_for_edge(STETHO_BUTTON_PIN, GPIO.FALLING, bouncetime=200)

        # Reset LEDs
        for led in [GREEN_LED_PIN, YELLOW_LED_PIN, RED_LED_PIN, BLUE_LED_PIN]:
            GPIO.output(led, GPIO.LOW)

        print("🎙️ Button pressed! Starting recording...")
        GPIO.output(GREEN_LED_PIN, GPIO.HIGH)

        while GPIO.input(STETHO_BUTTON_PIN) == GPIO.LOW:
            time.sleep(0.1)

        samples = []
        delay = 1.0 / SAMPLE_RATE
        start_time = time.time()

        for i in range(TOTAL_SAMPLES):
            value = read_adc(CHANNEL)
            normalized = (value / 127.5) - 1.0
            samples.append(normalized)

            target_time = start_time + (i + 1) * delay
            while time.time() < target_time:
                pass

            if i % SAMPLE_RATE == 0:
                print(f"... {i // SAMPLE_RATE}/{DURATION} seconds")

        samples_np = np.array(samples, dtype=np.float32)
        next_id = generate_next_id()

        wav_filename = f"Audios/heartbeat_{next_id:03}.wav"
        wav.write(wav_filename, SAMPLE_RATE, (samples_np * 32767).astype(np.int16))
        print(f"💾 WAV saved: {wav_filename}")

        txt_filename = f"Audios/heartbeat_{next_id:03}.txt"
        np.savetxt(txt_filename, samples_np, delimiter=',')
        print(f"📄 TXT saved: {txt_filename}")

        GPIO.output(GREEN_LED_PIN, GPIO.LOW)

        # Inference
        print("🧐 Preprocessing audio...")
        mel = preprocess_audio(wav_filename).to(device)
        demo = torch.tensor([[22/100.0, 1.0, 0.0, 1.0]], dtype=torch.float).to(device)

        print("🤖 Running binary model...")
        with torch.no_grad():
            logits_bin = binary_model(mel, demo)
            pred_bin = torch.argmax(logits_bin, dim=1).item()

        if pred_bin == 1:
            print("✅ Prediction: NORMAL heartbeat sound")
            GPIO.output(BLUE_LED_PIN, GPIO.HIGH)
        else:
            print("⚠️ Prediction: ABNORMAL heartbeat sound")
            print("🤖 Running multi-class model...")
            with torch.no_grad():
                logits_multi = multi_model(mel, demo)
                probs = torch.nn.functional.softmax(logits_multi, dim=1)
                pred_class = torch.argmax(probs, dim=1).item()

            class_names = [
                "Aortic Regurgitation",
                "Aortic Stenosis",
                "Mitral Regurgitation",
                "Mitral Stenosis",
                "Multi Disease"
            ]
            print(f"💥 Detected Disease: {class_names[pred_class]} (class_id={pred_class})")
            GPIO.output(RED_LED_PIN, GPIO.HIGH)

        print("✅ Done. Press button to run again.")

except KeyboardInterrupt:
    print("\n⚠️ Stopped by user.")
    GPIO.output(YELLOW_LED_PIN, GPIO.HIGH)

except Exception as e:
    print(f"\n⚠️ ERROR: {e}")
    GPIO.output(YELLOW_LED_PIN, GPIO.HIGH)

finally:
    GPIO.cleanup()
    print("=== PulseX ended ===")
