import os
from pydub import AudioSegment
from matplotlib import pyplot as plt

folder_path = "../classes/normal"
audios_lengths = {}
sum = 0

for filename in os.listdir(folder_path):
    if (filename.endswith(".wav")):
        audio_path = os.path.join(folder_path, filename)
        audio = AudioSegment.from_file(audio_path)
        audios_lengths[filename] = len(audio) / 1000 # seconds conversion
    
for length in audios_lengths:
    print(audios_lengths[length])
    sum += audios_lengths[length]

average = sum / len(audios_lengths)
print('average: ', average)

plt.figure(figsize=(20,20))
plt.bar(audios_lengths.keys(), audios_lengths.values(), color='blue')
plt.xlabel('Audios')
plt.ylabel('Durations')
plt.title('Audios lengths')
plt.show()