import matplotlib.pyplot as plt
import scipy.io.wavfile as wav

file_name = 'asmae_recording'
sample_rate, data = wav.read(f"{file_name}.wav")

if len(data.shape) == 2:
    data = data[:, 0]

seconds = 20
N = seconds * sample_rate

time_axis = [i / sample_rate for i in range(N)]

plt.figure(figsize=(12, 4))
plt.plot(time_axis, data[:N])
plt.title("Heartbeat Plot")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid()

# Save plot as image
plt.savefig(f"heartbeat_plot_{file_name}.png", dpi=300)

plt.show()