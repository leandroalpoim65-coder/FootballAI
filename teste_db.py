import sqlite3
import random

# Script de teste rápido
conn = sqlite3.connect("database/football.db")
cursor = conn.cursor()

# Vamos inserir 5 jogos "fingidos" que o robô já analisou e já sabemos o resultado
dados_teste = [
    ('2026-07-10', 'Porto vs Benfica', 'BTTS', 'SIM', 0.8, '2-1', 1),
    ('2026-07-11', 'Sporting vs Braga', 'BTTS', 'SIM', 0.7, '1-0', 0),
    ('2026-07-12', 'Real Madrid vs Barça', 'BTTS', 'SIM', 0.9, '3-1', 1),
    ('2026-07-12', 'Man City vs Liverpool', 'BTTS', 'SIM', 0.6, '2-2', 1)
]

for d in dados_teste:
    cursor.execute("INSERT INTO historico_previsoes (data_jogo, partida, mercado, previsao, probabilidade, resultado_real, acertou) VALUES (?, ?, ?, ?, ?, ?, ?)", d)

conn.commit()
conn.close()
print("Dados de teste inseridos!")