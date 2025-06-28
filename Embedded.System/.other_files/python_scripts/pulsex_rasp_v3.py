import smbus2
import time
import numpy as np
import scipy.io.wavfile as wav
from datetime import datetime

# === CONFIGURATION ===
PCF8591_ADDR = 0x48
CHANNEL = 0  # AIN0
SAMPLE_RATE = 1000  # <-- realistic value for I2C ADC !!
DURATION = 20  # seconds
TOTAL_SAMPLES = SAMPLE_RATE * DURATION

# === SETUP I2C ===
bus = smbus2.SMBus(1)

def read_adc(channel=0):
    assert 0 <= channel <= 3, "Invalid ADC channel!"
    bus.write_byte(PCF8591_ADDR, 0x40 | channel)
    bus.read_byte(PCF8591_ADDR)  # Dummy read
    value = bus.read_byte(PCF8591_ADDR)
    return value

# === RECORDING ===
print(f"🎙️ Recording started for {DURATION} seconds ({TOTAL_SAMPLES} samples)...")

samples = []
delay = 1.0 / SAMPLE_RATE  # Time between samples (sec)
start_time = time.time()

for i in range(TOTAL_SAMPLES):
    value = read_adc(CHANNEL)
    normalized = (value / 127.5) - 1.0
    samples.append(normalized)

    # Wait until correct time
    target_time = start_time + (i + 1) * delay
    now = time.time()
    while now < target_time:
        now = time.time()

    if i % SAMPLE_RATE == 0:
        sec_passed = i // SAMPLE_RATE
        print(f"... {sec_passed}/{DURATION} seconds")

print("✅ Recording finished.")

# === SAVE ===
samples_np = np.array(samples, dtype=np.float32)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
wav_filename = f"heartbeat_{timestamp}.wav"

wav.write(wav_filename, SAMPLE_RATE, (samples_np * 32767).astype(np.int16))
print(f"💾 WAV saved: {wav_filename}")

txt_filename = f"heartbeat_{timestamp}.txt"
np.savetxt(txt_filename, samples_np, delimiter=',')
print(f"📄 TXT saved: {txt_filename}")
