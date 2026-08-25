import os
import sqlite3

db = "database/football.db"

print("Base de dados:", os.path.abspath(db))

conn = sqlite3.connect(db)

cursor = conn.cursor()

cursor.execute("SELECT COUNT(*) FROM matches")

print("Jogos:", cursor.fetchone()[0])

conn.close()