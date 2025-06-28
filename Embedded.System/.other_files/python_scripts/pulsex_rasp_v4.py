import smbus2
import time
import numpy as np
import scipy.io.wavfile as wav
from datetime import datetime
import RPi.GPIO as GPIO

# === CONFIGURATION ===
PCF8591_ADDR = 0x48
CHANNEL = 0  # AIN0
SAMPLE_RATE = 1000  # realistic value for I2C ADC
DURATION = 20  # seconds
TOTAL_SAMPLES = SAMPLE_RATE * DURATION

BUTTON_PIN = 21    # GPIO pin for button
GREEN_LED_PIN = 13  # GPIO pin for green LED

# === SETUP I2C ===
bus = smbus2.SMBus(1)

def read_adc(channel=0):
    assert 0 <= channel <= 3, "Invalid ADC channel!"
    bus.write_byte(PCF8591_ADDR, 0x40 | channel)
    bus.read_byte(PCF8591_ADDR)  # Dummy read
    value = bus.read_byte(PCF8591_ADDR)
    return value

# === SETUP GPIO ===
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)  # Button with pull-up resistor
GPIO.setup(GREEN_LED_PIN, GPIO.OUT)
GPIO.output(GREEN_LED_PIN, GPIO.LOW)  # LED OFF initially

# === WAIT FOR BUTTON PRESS ===
print("🔘 Waiting for button press to start recording...")

# Wait for button press (falling edge, debounce 200ms)
GPIO.wait_for_edge(BUTTON_PIN, GPIO.FALLING, bouncetime=200)

print("🎙️ Button pressed! Starting recording...")
GPIO.output(GREEN_LED_PIN, GPIO.HIGH)  # Turn ON green LED

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

GPIO.output(GREEN_LED_PIN, GPIO.LOW)  # Turn OFF green LED

# === SAVE ===
samples_np = np.array(samples, dtype=np.float32)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
wav_filename = f"heartbeat_{timestamp}.wav"

wav.write(wav_filename, SAMPLE_RATE, (samples_np * 32767).astype(np.int16))
print(f"💾 WAV saved: {wav_filename}")

txt_filename = f"heartbeat_{timestamp}.txt"
np.savetxt(txt_filename, samples_np, delimiter=',')
print(f"📄 TXT saved: {txt_filename}")

# === CLEANUP ===
GPIO.cleanup()
