import sqlite3
import pandas as pd

conn = sqlite3.connect("database/football.db")

df = pd.read_sql_query("SELECT * FROM matches", conn)

conn.close()

equipas = sorted(set(df["home_team"]).union(df["away_team"]))

dados = []

for equipa in equipas:

    casa = df[df.home_team == equipa]
    fora = df[df.away_team == equipa]

    jogos = len(casa) + len(fora)

    gm = casa.home_goals.sum() + fora.away_goals.sum()

    gs = casa.away_goals.sum() + fora.home_goals.sum()

    over = (casa.over25.sum() + fora.over25.sum()) / jogos * 100

    btts = (casa.btts.sum() + fora.btts.sum()) / jogos * 100

    dados.append([
        equipa,
        jogos,
        gm,
        gs,
        round(over,1),
        round(btts,1)
    ])

tabela = pd.DataFrame(
    dados,
    columns=[
        "Equipa",
        "Jogos",
        "GM",
        "GS",
        "Over25",
        "BTTS"
    ]
)

tabela = tabela.sort_values("GM", ascending=False)

print(tabela.head(20))