import pandas as pd

df = pd.read_csv('demographic_data.csv')
df = df[['Patient_id', 'Age', 'Gender', 'Smoker', 'Lives', 'Healthy']]
df.to_csv('demographic_data.csv', index=False)
