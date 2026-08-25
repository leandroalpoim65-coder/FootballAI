import pandas as pd

df = pd.read_csv("data/processed/dados_preparados.csv")

print(df.columns.tolist())