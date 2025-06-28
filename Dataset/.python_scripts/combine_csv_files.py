import pandas as pd

df1 = pd.read_csv('../demographic_data.csv')
df2 = pd.read_csv('../demographic_data_survey.csv')
df_combined = pd.concat([df1, df2], ignore_index=True)
df_combined.to_csv('../combined.csv', index=False)