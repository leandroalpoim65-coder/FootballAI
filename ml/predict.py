import sqlite3
import pandas as pd
import joblib

# 1. Carregar o modelo guardado
try:
    modelo = joblib.load('ml/modelo_btts.pkl')
    features = joblib.load('ml/features_btts.pkl')
except FileNotFoundError:
    print("Erro: Treina o modelo primeiro correndo 'train_model.py'!")
    exit()

# 2. Função para ir buscar as estatísticas recentes das duas equipas à BD
def obter_stats_equipa(team_name):
    conn = sqlite3.connect("database/football.db")
    
    # Procurar os últimos 5 jogos onde esta equipa jogou (seja casa ou fora)
    query = f"""
    SELECT date, home_team, away_team, home_goals, away_goals 
    FROM matches 
    WHERE home_team = '{team_name}' OR away_team = '{team_name}'
    ORDER BY date DESC 
    LIMIT 5
    """
    df_recent = pd.read_sql_query(query, conn)
    conn.close()
    
    if len(df_recent) == 0:
        return None
        
    # Calcular golos marcados, sofridos e pontos nos últimos 5 jogos
    goals_scored = []
    goals_conceded = []
    points = []
    
    for _, row in df_recent.iterrows():
        if row['home_team'] == team_name:
            goals_scored.append(row['home_goals'])
            goals_conceded.append(row['away_goals'])
            pts = 3 if row['home_goals'] > row['away_goals'] else (1 if row['home_goals'] == row['away_goals'] else 0)
        else:
            goals_scored.append(row['away_goals'])
            goals_conceded.append(row['home_goals'])
            pts = 3 if row['away_goals'] > row['home_goals'] else (1 if row['home_goals'] == row['away_goals'] else 0)
        points.append(pts)
            
    return {
        'avg_scored': sum(goals_scored) / len(goals_scored),
        'avg_conceded': sum(goals_conceded) / len(goals_conceded),
        'power_rating': sum(points) # Soma dos pontos dos últimos 5 jogos
    }

# 3. INTERFACE DE UTILIZADOR NO TERMINAL
print("=========================================")
print("      PREVISOR DE AMBAS MARCAM (BTTS)     ")
print("=========================================\n")

casa = input("Nome da Equipa da Casa: ").strip()
fora = input("Nome da Equipa de Fora: ").strip()

stats_casa = obter_stats_equipa(casa)
stats_fora = obter_stats_equipa(fora)

if not stats_casa:
    print(f"Erro: Não encontrámos dados históricos para a equipa '{casa}'. Verifica se o nome está correto na BD.")
elif not stats_fora:
    print(f"Erro: Não encontrámos dados históricos para a equipa '{fora}'. Verifica se o nome está correto na BD.")
else:
    # Construir os dados do jogo exatamente como o modelo espera
    dados_jogo = {
        'home_avg_scored_5': stats_casa['avg_scored'],
        'home_avg_conceded_5': stats_casa['avg_conceded'],
        'away_avg_scored_5': stats_fora['avg_scored'],
        'away_avg_conceded_5': stats_fora['avg_conceded'],
        'total_expected_scored': stats_casa['avg_scored'] + stats_fora['avg_scored'],
        'total_expected_conceded': stats_casa['avg_conceded'] + stats_fora['avg_conceded'],
        'home_power_rating': stats_casa['power_rating'],
        'away_power_rating': stats_fora['power_rating'],
        'power_rating_diff': stats_casa['power_rating'] - stats_fora['power_rating']
    }
    
    # Criar DataFrame de uma linha para o modelo ler
    df_jogo = pd.DataFrame([dados_jogo])[features]
    
    # Prever a probabilidade
    probabilidade = modelo.predict_proba(df_jogo)[0, 1] * 100
    
    print("\n---------------- RESULTADO ----------------")
    print(f"Jogo: {casa} vs {fora}")
    print(f"Probabilidade calculada para AMBAS MARCAM: {probabilidade:.2f}%")
    
    if probabilidade >= 65:
        print("RECOMENDAÇÃO: 🟢 APOSTAR EM AMBAS MARCAM (SIM)")
    elif probabilidade <= 35:
        print("RECOMENDAÇÃO: 🔴 APOSTAR EM AMBAS MARCAM (NÃO)")
    else:
        print("RECOMENDAÇÃO: ⚪ SEM VALOR (Não apostar neste jogo)")
    print("-------------------------------------------\n")