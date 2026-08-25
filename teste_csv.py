import pandas as pd

df = pd.read_csv(r"data/historical/England/E0 (1).csv")

for coluna in df.columns:
    if "2.5" in coluna or ">2.5" in coluna or "<2.5" in coluna:
        print(coluna)