import sqlite3

conn = sqlite3.connect("database/football.db")
cursor = conn.cursor()

cursor.execute("PRAGMA table_info(matches)")

for coluna in cursor.fetchall():
    print(coluna)

conn.close()