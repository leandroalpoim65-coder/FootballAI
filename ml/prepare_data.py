import sqlite3
import pandas as pd

conn = sqlite3.connect("database/football.db")

df = pd.read_sql_query("""
SELECT
    id,
    date,
    league,
    home_team,
    away_team,

    home_goals,
    away_goals,

    over25,
    btts,

    odd_home,
    odd_draw,
    odd_away,

    odd_over25,
    odd_under25

FROM matches
""", conn)

conn.close()

# Garante que as odds são numéricas
odds = [
    "odd_home",
    "odd_draw",
    "odd_away",
    "odd_over25",
    "odd_under25"
]

for col in odds:
    df[col] = pd.to_numeric(df[col], errors="coerce")

print(df.head())

df.to_csv("data/processed/training_data.csv", index=False)

print(f"\n✅ Dataset criado com {len(df)} jogos.")