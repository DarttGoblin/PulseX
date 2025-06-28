import pandas as pd

df = pd.read_csv('../demographic_data_survey.csv')
df.drop('Timestamp', axis=1, inplace=True)
df.drop('Username', axis=1, inplace=True)
df.to_csv('../demographic_data_survey.csv', index=False)