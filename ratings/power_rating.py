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

    if jogos == 0:
        continue

    gm = casa.home_goals.sum() + fora.away_goals.sum()
    gs = casa.away_goals.sum() + fora.home_goals.sum()

    ataque = gm / jogos
    defesa = gs / jogos

    over = (casa.over25.sum() + fora.over25.sum()) / jogos

    rating = (ataque * 50) - (defesa * 20) + (over * 30)

    dados.append({
        "Equipa": equipa,
        "Ataque": round(ataque, 2),
        "Defesa": round(defesa, 2),
        "Over25": round(over * 100, 1),
        "PowerRating": round(rating, 2)
    })

tabela = pd.DataFrame(dados)

tabela = tabela.sort_values("PowerRating", ascending=False)

print(tabela.head(20))