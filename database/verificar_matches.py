import sqlite3

conn = sqlite3.connect("database/football.db")
cursor = conn.cursor()

cursor.execute("SELECT COUNT(*) FROM matches")
print("Total de jogos:", cursor.fetchone()[0])

cursor.execute("""
SELECT id, date, league, home_team, away_team
FROM matches
LIMIT 5
""")

for linha in cursor.fetchall():
    print(linha)