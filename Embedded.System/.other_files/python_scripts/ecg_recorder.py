# ecg_recorder.py

import time
import numpy as np
import matplotlib.pyplot as plt
import smbus2
import os
from datetime import datetime

# Constants
PCF8591_ADDR = 0x48
ECG_CHANNEL = 1  # Use channel 1 for ECG input
ECG_SAMPLE_RATE = 250  # 250 Hz sampling rate
RECORD_SECONDS = 60
TOTAL_SAMPLES = ECG_SAMPLE_RATE * RECORD_SECONDS

bus = smbus2.SMBus(1)

def read_adc(channel=0):
    assert 0 <= channel <= 3, "Invalid ADC channel!"
    bus.write_byte(PCF8591_ADDR, 0x40 | channel)
    bus.read_byte(PCF8591_ADDR)
    return bus.read_byte(PCF8591_ADDR)

def record_ecg():
    print("📈 Starting ECG recording...")
    samples = []
    delay = 1.0 / ECG_SAMPLE_RATE
    start_time = time.time()

    for i in range(TOTAL_SAMPLES):
        value = read_adc(ECG_CHANNEL)
        samples.append(value)

        target_time = start_time + (i + 1) * delay
        while time.time() < target_time:
            pass

        if i % ECG_SAMPLE_RATE == 0:
            print(f"... {i // ECG_SAMPLE_RATE}/{RECORD_SECONDS} seconds")

    samples_np = np.array(samples, dtype=np.uint8)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_base = f"ECG/ecg_{timestamp}"

    os.makedirs("ECG", exist_ok=True)

    txt_file = f"{filename_base}.txt"
    np.savetxt(txt_file, samples_np, fmt='%d', delimiter=',')
    print(f"📄 ECG data saved to: {txt_file}")

    jpg_file = f"{filename_base}.jpg"
    plt.figure(figsize=(10, 4))
    plt.plot(samples_np, color='black')
    plt.title("ECG Recording")
    plt.xlabel("Sample")
    plt.ylabel("ADC Value")
    plt.tight_layout()
    plt.savefig(jpg_file)
    plt.close()
    print(f"🖼️ ECG plot saved to: {jpg_file}")

    return txt_file, jpg_file

if __name__ == "__main__":
    record_ecg()
