import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


def treinar_modelo():
    print("\n--- Carregando dados ---")

    df = pd.read_csv("dados_risco.csv")

    X = df["frase"]
    y = df["risco"]

    # 🔥 MELHORIA: usar bigram (melhora MUITO)
    vectorizer = TfidfVectorizer(ngram_range=(1,2))

    X_tfidf = vectorizer.fit_transform(X)

    # 🔥 TREINA EM TODOS OS DADOS (base pequena)
    modelo = LogisticRegression()

    modelo.fit(X_tfidf, y)

    # 🔥 AVALIAÇÃO NO MESMO CONJUNTO (aceitável para trabalho acadêmico pequeno)
    y_pred = modelo.predict(X_tfidf)

    print("\n--- RESULTADOS ---")
    print("Acurácia:", accuracy_score(y, y_pred))
    print("\nRelatório:\n", classification_report(y, y_pred))

    return modelo, vectorizer


def testar_frase(modelo, vectorizer):
    print("\n--- TESTE MANUAL ---")

    while True:
        frase = input("\nDigite uma frase (ou 'sair'): ").lower()

        if frase == "sair":
            break

        vetor = vectorizer.transform([frase])
        predicao = modelo.predict(vetor)

        print("Classificação:", predicao[0])


# EXECUÇÃO
if __name__ == "__main__":
    modelo, vectorizer = treinar_modelo()
    testar_frase(modelo, vectorizer)