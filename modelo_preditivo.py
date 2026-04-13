import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def treinar_modelo():
    print("\n--- 1. Carregando e Preparando os Dados ---")
    
    # Carrega o dataset
    from utils import carregar_dataset
    df = carregar_dataset()

    # Separando variáveis preditoras (X) e variável alvo (y)
    X = df.drop(columns=['doenca_cardiaca'])
    y = df['doenca_cardiaca']

    # Dividindo a base: 80% treino | 20% teste
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Dados de Treino: {len(X_treino)} pacientes")
    print(f"Dados de Teste: {len(X_teste)} pacientes")


    print("\n--- 2. Treinando o Modelo Random Forest ---")

    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X_treino, y_treino)

    print("Treinamento concluído!")


    # 🔥 NOVA ETAPA — SALVANDO O MODELO
    print("\n--- Salvando Modelo Treinado ---")

    # Garante que a pasta models existe
    os.makedirs("models", exist_ok=True)

    # Salva o modelo
    joblib.dump(modelo, "models/modelo_cardioia.pkl")

    # Salva também as colunas usadas no treinamento
    joblib.dump(X.columns.tolist(), "models/colunas_modelo.pkl")

    print("Modelo salvo em: models/modelo_cardioia.pkl")


    print("\n--- 3. Avaliando o Desempenho ---")

    previsoes = modelo.predict(X_teste)

    acuracia = accuracy_score(y_teste, previsoes)
    print(f"\nAcurácia do Modelo: {acuracia * 100:.2f}%")

    print("\nMatriz de Confusão:")
    matriz = confusion_matrix(y_teste, previsoes)
    print(matriz)

    print("\nRelatório de Classificação:")
    print(classification_report(y_teste, previsoes))


    # 🔎 Matriz de Confusão Visual
    plt.figure(figsize=(5, 4))
    sns.heatmap(matriz, annot=True, fmt='d', cmap='Blues')
    plt.title("Matriz de Confusão")
    plt.xlabel("Previsto")
    plt.ylabel("Real")
    plt.show()


    print("\n--- 4. Importância das Variáveis ---")

    importancias = modelo.feature_importances_

    df_importancia = pd.DataFrame({
        'Variavel': X.columns,
        'Importancia': importancias
    }).sort_values(by='Importancia', ascending=False)

    print(df_importancia)


    # 📊 Gráfico de Importância
    plt.figure(figsize=(8, 5))
    sns.barplot(x='Importancia', y='Variavel', data=df_importancia)
    plt.title("Importância das Variáveis")
    plt.xlabel("Importância")
    plt.ylabel("Variável")
    plt.show()


    print("\n--- 5. Validação Cruzada (5 Folds) ---")

    scores = cross_val_score(modelo, X, y, cv=5)

    print("Acurácias em cada Fold:", scores)
    print(f"Média da Validação Cruzada: {scores.mean() * 100:.2f}%")
    print(f"Desvio Padrão: {scores.std():.4f}")


if __name__ == "__main__":
    treinar_modelo()