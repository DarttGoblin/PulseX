import pandas as pd
import numpy as np

df = pd.read_csv('../../Models/demographic_data_last_version.csv')
print(df['Healthy'].value_counts())