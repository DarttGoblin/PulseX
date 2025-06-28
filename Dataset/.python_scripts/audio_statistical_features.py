import os
import numpy as np
import librosa
import scipy.stats
import pandas as pd

def extract_statistical_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    return {
        "file_name": os.path.basename(file_path),
        "mean": np.mean(y),
        "std_dev": np.std(y),
        "max": np.max(y),
        "min": np.min(y),
        "median": np.median(y),
        "rms": np.sqrt(np.mean(y**2)),
        "zero_crossing_rate": np.mean(librosa.feature.zero_crossing_rate(y)),
        "skewness": scipy.stats.skew(y),
        "kurtosis": scipy.stats.kurtosis(y)
    }

def process_folder(folder_path, output_csv="statistical_features.csv"):
    data = []
    for file in os.listdir(folder_path):
        if file.endswith(".wav"):
            file_path = os.path.join(folder_path, file)
            features = extract_statistical_features(file_path)
            data.append(features)
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"Saved features to {output_csv}")

# Example usage
# process_folder("../classes/aortic_regurgitation", "aortic_regurgitation_statistical_features.csv")
# process_folder("../classes/aortic_stenosis", "aortic_stenosis_statistical_features.csv")
# process_folder("../classes/mitral_regurgitation", "mitral_regurgitation_statistical_features.csv")
# process_folder("../classes/mitral_stenosis", "mitral_stenosis_statistical_features.csv")
# process_folder("../classes/multi_disease", "../statistical_features/multi_disease_statistical_features.csv")