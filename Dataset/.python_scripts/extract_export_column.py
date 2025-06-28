import pandas as pd

existing = pd.read_csv('demographic_data.csv')
new_col = pd.read_csv('additional_metadata.csv')[['patient_id']].rename(columns={'patient_id': 'Patient_id'})
combined = pd.concat([existing, new_col], axis=1)
combined.to_csv('demographic_data.csv', index=False)
