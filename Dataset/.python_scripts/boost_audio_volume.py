import os
from pydub import AudioSegment
from pydub.effects import normalize

def boost_audio_folder(input_folder, output_folder, boost_db=6):
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Process each audio file in the folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".wav"):  # Change this if your files have a different format
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"{boost_db}db_boosted_{filename}")

            # Load and boost audio
            audio = AudioSegment.from_file(input_path)
            boosted_audio = normalize(audio + boost_db)  # Apply boost and normalize to prevent clipping

            # Export boosted audio
            boosted_audio.export(output_path, format="wav")
            print(f"Processed: {filename} → {output_path}")

# Example usage: boost all .wav files in "input_folder" and save to "output_folder"
boost_audio_folder("raw_data/Dataset6/train/", "cleaned_data/Dataset6 (heart audio) (chosen)/train", boost_db=6)
