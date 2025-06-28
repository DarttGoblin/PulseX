import smbus2

bus = smbus2.SMBus(1)
addr = 0x48
try:
    bus.write_byte(addr, 0x40)
    print("PCF8591 found.")
except Exception as e:
    print(f"I2C error: {e}")
