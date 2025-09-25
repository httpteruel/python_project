import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

NOME_ARQUIVO_CSV = 'resultados_classificacao_real.csv'

try:
    df_resultados = pd.read_csv(NOME_ARQUIVO_CSV)
except FileNotFoundError:
    print(f"Erro: O arquivo '{NOME_ARQUIVO_CSV}' não foi encontrado.")
    print("Execute primeiro o script 'gerar_csv.py' para criar o arquivo de resultados.")
    exit()

classificadores = df_resultados['nome_classificador'].unique()

def criar_boxplot(df, metrica_alvo):
    df_metrica = df[df['metrica'] == metrica_alvo]

    dados_boxplot = [df_metrica[df_metrica['nome_classificador'] == c].iloc[0, 3:].values.astype(float) 
                     for c in classificadores]

    plt.figure(figsize=(10, 6))
    plt.boxplot(dados_boxplot, labels=classificadores, patch_artist=True, medianprops=dict(color='red'))
    
    plt.title(f'Distribuição de Resultados REAIS ({metrica_alvo}) por Classificador', fontsize=16)
    plt.xlabel('Classificador', fontsize=12)
    plt.ylabel(metrica_alvo, fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    min_val = min(min(d) for d in dados_boxplot)
    plt.ylim(max(0, min_val - 0.02), 1.0)
    
    plt.show()

criar_boxplot(df_resultados, 'Acuracia')

criar_boxplot(df_resultados, 'F1-Score')    