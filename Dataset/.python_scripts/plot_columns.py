import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('metadata.csv')
columns = ['Age', 'Gender', 'Smoker', 'Lives', 'Healthy']
colors = ['red', 'green', 'blue', 'orange', 'black']

for col, color in zip(columns, colors):
    plt.figure()
    if df[col].nunique() <= 2:
        df[col].value_counts().sort_index().plot(kind='bar', color=color)
        plt.xticks(ticks=[0, 1], labels=['0', '1'])
    else:
        df[col].plot(kind='hist', color=color, bins=20)
    plt.title(col)
    plt.savefig(f'{col}_plot.png')
    plt.close()
