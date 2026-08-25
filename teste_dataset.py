import pandas as pd

df = pd.read_csv("data/processed/training_data.csv")

print("\nCOLUNAS:")
print(df.columns.tolist())

print("\nPRIMEIRAS LINHAS:")
print(df.head())