import os

directory = '../../Models/binary_classification_audio_model/binary_classification_model_data/abnormal'

for filename in os.listdir(directory):
    if 'half' in filename:
        file_path = os.path.join(directory, filename)
        os.remove(file_path)