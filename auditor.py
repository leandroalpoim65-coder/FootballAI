from fuzzywuzzy import fuzz
import sqlite3
import pandas as pd

def encontrar_equipa_similar(nome_db, lista_sites):
    # Procura na lista do site a equipa que mais se parece com a que tens na DB
    melhor_nome = None
    max_score = 0
    for nome_site in lista_sites:
        score = fuzz.ratio(nome_db.lower(), nome_site.lower())
        if score > max_score:
            max_score = score
            melhor_nome = nome_site
    
    # Retorna o nome se a semelhança for superior a 80%
    return melhor_nome if max_score > 80 else None

# Exemplo de como usar no teu auditor:
# Se o nome no teu SQL é "Real Madrid" e no site vem "Real Madrid CF"
# score = fuzz.ratio("real madrid", "real madrid cf") 
# Isto dará um score alto (> 90), e o robô saberá que são a mesma equipa!