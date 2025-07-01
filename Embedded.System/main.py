# pulsex_main.py

import smbus2
import time
import numpy as np
import scipy.io.wavfile as wav
from datetime import datetime
import RPi.GPIO as GPIO
import torch
import os
import subprocess
import threading

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
ORANGE_LED_PIN = 7 # not available yet

STETHO_BUTTON_PIN = 21
ECG_BUTTON_PIN = 12
SETTINGS_BUTTON1_PIN = 8 # not available yet
SETTINGS_BUTTON2_PIN = 25 # not available yet

LO_MINUS_PIN = 17
LO_PLUS_PIN = 27

bus = smbus2.SMBus(1)

def read_adc(channel=0):
    assert 0 <= channel <= 3, "Invalid ADC channel!"
    bus.write_byte(PCF8591_ADDR, 0x40 | channel)
    bus.read_byte(PCF8591_ADDR)
    return bus.read_byte(PCF8591_ADDR)

GPIO.setmode(GPIO.BCM)

GPIO.setup(STETHO_BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(ECG_BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(SETTINGS_BUTTON1_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(SETTINGS_BUTTON2_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

for pin in [WHITE_LED_PIN, YELLOW_LED_PIN, GREEN_LED_PIN, BLUE_LED_PIN, RED_LED_PIN, ORANGE_LED_PIN]:
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, GPIO.LOW)

os.makedirs("Audios", exist_ok=True)
print("=== PulseX System Ready ===")

def startup_sequence():
    sequence = [WHITE_LED_PIN, YELLOW_LED_PIN, GREEN_LED_PIN, BLUE_LED_PIN, RED_LED_PIN]
    for _ in range(5):
        # forward
        for pin in sequence:
            GPIO.output(pin, GPIO.HIGH)
            time.sleep(0.05)
            GPIO.output(pin, GPIO.LOW)
        # backward
        for pin in reversed(sequence):
            GPIO.output(pin, GPIO.HIGH)
            time.sleep(0.05)
            GPIO.output(pin, GPIO.LOW)
    # stop with only WHITE on
    GPIO.output(WHITE_LED_PIN, GPIO.HIGH)

startup_sequence()

# === Load models ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
binary_model = BinaryResNet().to(device)
binary_model.load_state_dict(torch.load("Models/binary_label_model.pth", map_location=device))
binary_model.eval()

multi_model = MultiClassResNet(num_classes=5).to(device)
multi_model.load_state_dict(torch.load("Models/multi_label_model.pth", map_location=device))
multi_model.eval()

def run_stethoscope():
    print("🎙️ Button pressed! Starting recording...")

    start_time = time.time()
    delay = 1.0 / SAMPLE_RATE

    samples = []
    led_on = True
    last_toggle = time.time()

    # Precompute sample thresholds
    solid_end = int(10 * SAMPLE_RATE)
    blink_500_end = int(13 * SAMPLE_RATE)
    blink_300_end = int(17 * SAMPLE_RATE)
    blink_100_end = int(19 * SAMPLE_RATE)
    blink_50_end = TOTAL_SAMPLES

    for i in range(TOTAL_SAMPLES):
        value = read_adc(CHANNEL)
        normalized = (value / 127.5) - 1.0
        samples.append(normalized)

        # Determine blinking pattern by sample count
        if i < solid_end:
            # solid on
            GPIO.output(GREEN_LED_PIN, GPIO.HIGH)
        elif i < blink_500_end:
            # blink every 500ms
            if time.time() - last_toggle >= 0.5:
                led_on = not led_on
                GPIO.output(GREEN_LED_PIN, GPIO.HIGH if led_on else GPIO.LOW)
                last_toggle = time.time()
        elif i < blink_300_end:
            # blink every 300ms
            if time.time() - last_toggle >= 0.3:
                led_on = not led_on
                GPIO.output(GREEN_LED_PIN, GPIO.HIGH if led_on else GPIO.LOW)
                last_toggle = time.time()
        elif i < blink_100_end:
            # blink every 100ms
            if time.time() - last_toggle >= 0.1:
                led_on = not led_on
                GPIO.output(GREEN_LED_PIN, GPIO.HIGH if led_on else GPIO.LOW)
                last_toggle = time.time()
        else:
            # blink every 50ms
            if time.time() - last_toggle >= 0.05:
                led_on = not led_on
                GPIO.output(GREEN_LED_PIN, GPIO.HIGH if led_on else GPIO.LOW)
                last_toggle = time.time()

        # keep sampling on time
        target_time = start_time + (i + 1) * delay
        while time.time() < target_time:
            pass

        # print progress by approximate "seconds"
        if i % SAMPLE_RATE == 0:
            print(f"... {i // SAMPLE_RATE}/{DURATION} seconds")

    GPIO.output(GREEN_LED_PIN, GPIO.LOW)

    # Save data
    samples_np = np.array(samples, dtype=np.float32)
    next_id = generate_next_id()
    wav_filename = f"Audios/heartbeat_{next_id:03}.wav"
    wav.write(wav_filename, SAMPLE_RATE, (samples_np * 32767).astype(np.int16))
    print(f"💾 WAV saved: {wav_filename}")
    txt_filename = f"Audios/heartbeat_{next_id:03}.txt"
    np.savetxt(txt_filename, samples_np, delimiter=',')
    print(f"📄 TXT saved: {txt_filename}")

    # Predictions
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

def run_ecg():
    print("🫀 ECG recording started...")
    GPIO.output(GREEN_LED_PIN, GPIO.HIGH)
    subprocess.run(["python3", "ecg_recorder.py"])
    GPIO.output(GREEN_LED_PIN, GPIO.LOW)
    print("🫀 ECG recording finished.")

try:
    printed_idle_msg = False  # NEW FLAG

    while True:
        if not printed_idle_msg:
            print("🔘 Waiting for button press (Stetho/ECG)...")
            printed_idle_msg = True

        if GPIO.input(STETHO_BUTTON_PIN) == GPIO.LOW:
            while GPIO.input(ECG_BUTTON_PIN) == GPIO.LOW:
                time.sleep(0.1)  # Wait until ECG button is released
            run_stethoscope()
            time.sleep(3)  # Keep prediction LED on for 3 sec
            for led in [GREEN_LED_PIN, YELLOW_LED_PIN, RED_LED_PIN, BLUE_LED_PIN]:
                GPIO.output(led, GPIO.LOW)
            printed_idle_msg = False  # allow printing next time

        elif GPIO.input(ECG_BUTTON_PIN) == GPIO.LOW:
            while GPIO.input(STETHO_BUTTON_PIN) == GPIO.LOW:
                time.sleep(0.1)  # Wait until stethoscope button is released
            run_ecg()
            time.sleep(3)  # if you want LED to stay for ECG too
            for led in [GREEN_LED_PIN, YELLOW_LED_PIN, RED_LED_PIN, BLUE_LED_PIN]:
                GPIO.output(led, GPIO.LOW)
            printed_idle_msg = False  # allow printing next time

        time.sleep(0.1)

except KeyboardInterrupt:
    print("\n⚠️ Stopped by user.")
    GPIO.output(YELLOW_LED_PIN, GPIO.HIGH)

except Exception as e:
    print(f"\n⚠️ ERROR: {e}")
    GPIO.output(YELLOW_LED_PIN, GPIO.HIGH)

finally:
    GPIO.cleanup()
    print("=== PulseX ended ===")