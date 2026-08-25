import sqlite3
import os

os.makedirs("database", exist_ok=True)

conn = sqlite3.connect("database/football.db")
cursor = conn.cursor()

cursor.execute("""
DROP TABLE IF EXISTS matches
""")

cursor.execute("""
CREATE TABLE matches (

    id INTEGER PRIMARY KEY AUTOINCREMENT,

    date TEXT,
    season TEXT,
    league TEXT,
    division TEXT,

    home_team TEXT,
    away_team TEXT,

    home_goals INTEGER,
    away_goals INTEGER,

    result TEXT,

    over25 INTEGER,
    btts INTEGER,

    odd_home REAL,
    odd_draw REAL,
    odd_away REAL,

    odd_over25 REAL,
    odd_under25 REAL,

    referee TEXT,

    home_shots INTEGER,
    away_shots INTEGER,

    home_shots_target INTEGER,
    away_shots_target INTEGER,

    home_corners INTEGER,
    away_corners INTEGER,

    home_yellow INTEGER,
    away_yellow INTEGER,

    home_red INTEGER,
    away_red INTEGER
)
""")

conn.commit()
conn.close()