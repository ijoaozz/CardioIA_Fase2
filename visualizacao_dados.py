import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def visualizar_dados():
    print("Carregando dados para gerar os gráficos...")
    # Lê o arquivo CSV
    from utils import carregar_dataset
    df = carregar_dataset()

    # Configurando o estilo visual dos gráficos
    sns.set_theme(style="whitegrid")

    # --- Gráfico 1: Quantidade de Pacientes (Doentes vs Saudáveis) ---
    plt.figure(figsize=(6, 4))
    sns.countplot(x='doenca_cardiaca', data=df, palette='viridis')
    plt.title('Distribuição: 0 = Saudável | 1 = Doença Cardíaca', fontsize=12)
    plt.xlabel('Diagnóstico')
    plt.ylabel('Número de Pacientes')
    plt.show() # Feche a janela deste gráfico para abrir o próximo

    # --- Gráfico 2: Diagnóstico por Sexo ---
    plt.figure(figsize=(6, 4))
    sns.countplot(x='doenca_cardiaca', hue='sexo', data=df, palette='Set2')
    plt.title('Doença Cardíaca por Sexo (0 = Mulher | 1 = Homem)', fontsize=12)
    plt.xlabel('Diagnóstico')
    plt.ylabel('Número de Pacientes')
    plt.show() # Feche a janela deste gráfico para abrir o próximo

    # --- Gráfico 3: Mapa de Calor de Correlação ---
    # Mostra matematicamente quais fatores estão mais ligados à doença
    plt.figure(figsize=(12, 8))
    correlacao = df.corr()
    sns.heatmap(correlacao, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Mapa de Calor: Correlação entre os Fatores de Risco', fontsize=14)
    plt.show()

# Executa a função
if __name__ == "__main__":
    visualizar_dados()