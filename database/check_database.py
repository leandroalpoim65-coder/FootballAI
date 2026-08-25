import sqlite3

conn = sqlite3.connect("database/football.db")
cursor = conn.cursor()

cursor.execute("SELECT COUNT(*) FROM matches")
total = cursor.fetchone()[0]

print(f"Total de jogos: {total}")

cursor.execute("""
SELECT
AVG(home_goals + away_goals)
FROM matches
""")

media = cursor.fetchone()[0]

print(f"Média de golos: {media:.2f}")

cursor.execute("""
SELECT
AVG(over25) * 100
FROM matches
""")

over = cursor.fetchone()[0]

print(f"Over 2.5: {over:.1f}%")

cursor.execute("""
SELECT
AVG(btts) * 100
FROM matches
""")

btts = cursor.fetchone()[0]

print(f"BTTS: {btts:.1f}%")

conn.close()