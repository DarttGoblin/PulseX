import librosa
import soundfile as sf
import os
import numpy as np

# Define paths
audio_folder = "../Dataset6/multi_disease"

# Ensure output directory exists
output_path = "../Dataset6/multi_disease_sliced"
os.makedirs(output_path, exist_ok=True)

# Get list of audio files in the folder
audio_files = [f for f in os.listdir(audio_folder) if f.endswith('.wav')]

# Slice each audio file in half
for file in audio_files:
    # Load audio
    audio, sr = librosa.load(os.path.join(audio_folder, file), sr=None)
    
    # Get the length of the audio (in samples)
    audio_length = len(audio)
    
    # Split the audio in half
    half_point = audio_length // 2
    first_half = audio[:half_point]
    second_half = audio[half_point:]
    
    # Extract filename without extension
    name_without_extension = os.path.splitext(file)[0]
    
    # Save the two halves with the desired naming convention
    sf.write(os.path.join(output_path, f"{name_without_extension}_first_half.wav"), first_half, sr)
    sf.write(os.path.join(output_path, f"{name_without_extension}_second_half.wav"), second_half, sr)

print("Audio slicing completed.")
