import smbus2
import time

PCF8591_ADDR = 0x48  # Detected address

bus = smbus2.SMBus(1)

def read_adc(channel=0):
    assert 0 <= channel <= 3, "Invalid ADC channel!"
    bus.write_byte(PCF8591_ADDR, 0x40 | channel)
    bus.read_byte(PCF8591_ADDR)  # Dummy read
    value = bus.read_byte(PCF8591_ADDR)
    return value

print("Reading PCF8591 AIN0...")
for i in range(10):
    val = read_adc(0)
    print(f"Sample {i+1}: {val}")
    time.sleep(0.1)
