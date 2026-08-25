import sqlite3
import os

# Garante que a pasta 'database' existe para evitar erros
if not os.path.exists('database'):
    os.makedirs('database')

def criar_tabela():
    # Conecta ao ficheiro football.db (cria automaticamente se não existir)
    conn = sqlite3.connect("database/football.db")
    cursor = conn.cursor()
    
    # Executa o comando SQL para criar a tabela
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS historico_previsoes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            data_jogo DATE,
            partida TEXT,
            mercado TEXT,
            previsao TEXT,
            probabilidade REAL,
            resultado_real TEXT,
            acertou BOOLEAN
        )
    """)
    
    conn.commit()
    conn.close()
    print("✅ Tabela 'historico_previsoes' criada com sucesso em 'database/football.db'!")

if __name__ == "__main__":
    criar_tabela()