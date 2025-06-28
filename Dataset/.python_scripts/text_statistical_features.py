import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis, entropy, mode

# Load the demographic CSV file
df = pd.read_csv('../demographic_data.csv')

# Define which columns are continuous and which are binary
continuous_cols = ['Age']
binary_cols = ['Gender', 'Smoker', 'Lives', 'Healthy']

# Feature extraction result container
features = {}

# Continuous features
for col in continuous_cols:
    data = df[col].dropna()
    features[col] = {
        'mean': data.mean(),
        'std': data.std(),
        'min': data.min(),
        'max': data.max(),
        'median': data.median(),
        'skewness': skew(data),
        'kurtosis': kurtosis(data),
        'entropy': entropy(np.histogram(data, bins=10, density=True)[0] + 1e-9)  # small offset to avoid log(0)
    }

# Binary features
for col in binary_cols:
    data = df[col].dropna()
    count_1 = data.sum()
    count_0 = len(data) - count_1
    mode_value, count = mode(data)
    
    # In case of multiple modes, we can pick the first one
    features[col] = {
        'count_1': int(count_1),
        'count_0': int(count_0),
        'proportion_1': count_1 / len(data),
        'mode': mode_value,  # Take the first mode if multiple
        'entropy': entropy([count_0, count_1])
    }

# Convert the features dict to a DataFrame and save it
features_df = pd.DataFrame(features).T
features_df.to_csv('demographic_stat_features.csv')
print("Feature extraction completed and saved.")