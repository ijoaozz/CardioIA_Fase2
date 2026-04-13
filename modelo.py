"""
modelo.py — Módulo de Análise de Viés e Governança de Dados do CardioIA
Implementa funções para auditar equidade, balanceamento e padrões de vocabulário
do classificador de risco clínico.

Versão: 2.0 (Fase 2 — refatoração completa)
"""

import logging
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("CardioIA.modelo")


# ---------------------------------------------------------------------------
# FUNÇÃO 1 — DISTRIBUIÇÃO DE CLASSES
# ---------------------------------------------------------------------------

def analisar_distribuicao_classes(df: pd.DataFrame, coluna_classe: str = "risco") -> pd.DataFrame:
    """
    Analisa o balanceamento das classes no dataset.

    Datasets desbalanceados podem fazer com que o modelo aprenda a classificar
    a classe majoritária com alta acurácia, enquanto falha sistematicamente
    na classe minoritária — que, em contexto clínico, é frequentemente a mais
    crítica (alto risco).

    Args:
        df: DataFrame contendo os dados de treinamento.
        coluna_classe: Nome da coluna com os rótulos de classe.

    Returns:
        DataFrame com contagem, proporção e avaliação de balanceamento por classe.
    """
    if coluna_classe not in df.columns:
        raise ValueError(f"Coluna '{coluna_classe}' não encontrada no DataFrame.")

    contagem = df[coluna_classe].value_counts()
    proporcao = df[coluna_classe].value_counts(normalize=True).round(4)
    total = len(df)

    resultado = pd.DataFrame({
        "classe": contagem.index,
        "contagem": contagem.values,
        "proporcao": proporcao.values,
        "percentual": (proporcao.values * 100).round(2)
    })

    # Avaliação do nível de balanceamento
    razao = contagem.min() / contagem.max()
    if razao >= 0.8:
        status = "Balanceado (razão ≥ 80%)"
    elif razao >= 0.5:
        status = "Moderadamente desbalanceado (razão entre 50–80%)"
    else:
        status = "Desbalanceado — risco de viés de classe (razão < 50%)"

    logger.info("Balanceamento: %s | Razão: %.2f", status, razao)

    print("=" * 50)
    print("DISTRIBUIÇÃO DE CLASSES")
    print("=" * 50)
    print(resultado.to_string(index=False))
    print(f"\nTotal de amostras: {total}")
    print(f"Razão min/max: {razao:.2%}")
    print(f"Status: {status}")
    print("=" * 50)

    return resultado


# ---------------------------------------------------------------------------
# FUNÇÃO 2 — ANÁLISE DE TOKENS POR CLASSE
# ---------------------------------------------------------------------------

def analisar_tokens_por_classe(
    df: pd.DataFrame,
    coluna_texto: str = "frase",
    coluna_classe: str = "risco",
    top_n: int = 10
) -> pd.DataFrame:
    """
    Identifica as palavras mais associadas a cada classe usando TF-IDF.

    Palavras com coeficientes elevados indicam forte associação com a classe.
    Se termos demograficamente neutros (como "idosa", "mulher") aparecerem
    com alto coeficiente, isso é evidência de viés demográfico.

    Args:
        df: DataFrame com texto e rótulos.
        coluna_texto: Coluna contendo os relatos de texto.
        coluna_classe: Coluna com os rótulos de classe.
        top_n: Quantidade de termos mais relevantes a exibir por classe.

    Returns:
        DataFrame com os termos mais discriminativos por classe.
    """
    X = df[coluna_texto]
    y = df[coluna_classe]

    # Vetorização para análise de tokens
    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=300, sublinear_tf=True)
    X_tfidf = vec.fit_transform(X)

    # Modelo para extrair coeficientes
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    clf.fit(X_tfidf, y)

    nomes = vec.get_feature_names_out()
    coefs = clf.coef_[0]  # coeficientes para a classe positiva (última em ordem alfabética)
    classes = clf.classes_

    # Classe com coeficiente mais alto = associada ao índice maior
    # Classe com coeficiente mais baixo = associada ao índice menor
    classe_positiva = classes[-1]
    classe_negativa = classes[0]

    idx_positivo = np.argsort(coefs)[-top_n:][::-1]
    idx_negativo = np.argsort(coefs)[:top_n]

    dados = []
    for idx in idx_positivo:
        dados.append({"termo": nomes[idx], "coeficiente": round(coefs[idx], 4), "classe_associada": classe_positiva})
    for idx in idx_negativo:
        dados.append({"termo": nomes[idx], "coeficiente": round(coefs[idx], 4), "classe_associada": classe_negativa})

    resultado = pd.DataFrame(dados)

    print("=" * 60)
    print(f"TOP {top_n} TERMOS MAIS DISCRIMINATIVOS POR CLASSE")
    print("=" * 60)
    for classe in [classe_positiva, classe_negativa]:
        subset = resultado[resultado["classe_associada"] == classe]
        print(f"\nAssociados a: [{classe.upper()}]")
        print(subset[["termo", "coeficiente"]].to_string(index=False))
    print("=" * 60)

    logger.info("Análise de tokens concluída: %d termos analisados.", len(nomes))
    return resultado


# ---------------------------------------------------------------------------
# FUNÇÃO 3 — SIMULAÇÃO DE VIÉS DEMOGRÁFICO
# ---------------------------------------------------------------------------

def simular_vies_demografico(
    modelo,
    vetorizador,
    sintoma_base: str = "dor no peito e cansaco extremo"
) -> pd.DataFrame:
    """
    Testa se o modelo atribui classificações diferentes para frases
    clinicamente equivalentes, diferenciadas apenas por marcadores
    demográficos (gênero e faixa etária).

    Em um sistema equânime, a probabilidade de alto risco para
    'um homem com dor no peito' deve ser similar à de
    'uma mulher idosa com dor no peito' — pois o sintoma é idêntico.
    Divergências indicam viés não clínico aprendido.

    Args:
        modelo: Classificador treinado com método predict_proba.
        vetorizador: TfidfVectorizer já ajustado.
        sintoma_base: Sintomas comuns a todas as frases de teste.

    Returns:
        DataFrame com a classificação e probabilidade por perfil demográfico.
    """
    perfis = [
        ("Homem jovem (20-30)",  f"um homem jovem com {sintoma_base}"),
        ("Mulher jovem (20-30)", f"uma mulher jovem com {sintoma_base}"),
        ("Homem adulto (40-50)", f"um homem de meia idade com {sintoma_base}"),
        ("Mulher adulta (40-50)", f"uma mulher de meia idade com {sintoma_base}"),
        ("Homem idoso (70+)",    f"um homem idoso com {sintoma_base}"),
        ("Mulher idosa (70+)",   f"uma mulher idosa com {sintoma_base}"),
    ]

    classes = modelo.classes_
    resultados = []

    for perfil, frase in perfis:
        vetor = vetorizador.transform([frase.lower()])
        pred = modelo.predict(vetor)[0]
        probs = modelo.predict_proba(vetor)[0]

        # Probabilidade de alto risco
        idx_alto = list(classes).index("alto risco") if "alto risco" in classes else 0
        p_alto = probs[idx_alto]

        resultados.append({
            "Perfil demográfico": perfil,
            "Frase de teste": frase,
            "Classificação": pred,
            "P(alto risco)": f"{p_alto:.2%}"
        })

    df_resultado = pd.DataFrame(resultados)

    print("=" * 70)
    print("SIMULAÇÃO DE VIÉS DEMOGRÁFICO")
    print(f"Sintoma base: '{sintoma_base}'")
    print("=" * 70)
    print(df_resultado[["Perfil demográfico", "Classificação", "P(alto risco)"]].to_string(index=False))
    print()

    # Análise de dispersão das probabilidades
    probs_numericas = [
        float(r["P(alto risco)"].strip("%")) / 100
        for r in resultados
    ]
    dispersao = max(probs_numericas) - min(probs_numericas)

    if dispersao > 0.15:
        alerta = "ALERTA: Dispersão > 15% — viés demográfico provável."
    elif dispersao > 0.05:
        alerta = "ATENÇÃO: Dispersão entre 5–15% — monitorar com mais dados."
    else:
        alerta = "OK: Dispersão ≤ 5% — modelo relativamente equânime para este sintoma."

    print(f"Dispersão máxima de P(alto risco): {dispersao:.2%}")
    print(f"Avaliação: {alerta}")
    print("=" * 70)

    logger.info("Simulação de viés demográfico concluída. Dispersão: %.2f%%", dispersao * 100)
    return df_resultado


# ---------------------------------------------------------------------------
# EXECUÇÃO STANDALONE — para testes diretos
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    # Carrega o dataset
    caminho_csv = "dados_risco.csv"
    if not os.path.exists(caminho_csv):
        print(f"Arquivo '{caminho_csv}' não encontrado. Execute a partir da raiz do projeto.")
    else:
        df = pd.read_csv(caminho_csv)

        # 1. Distribuição de classes
        analisar_distribuicao_classes(df)

        # 2. Tokens por classe
        analisar_tokens_por_classe(df, top_n=8)

        # 3. Viés demográfico — requer modelo treinado
        vec = TfidfVectorizer(ngram_range=(1, 2), max_features=200, sublinear_tf=True)
        X_tfidf = vec.fit_transform(df["frase"])
        clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        clf.fit(X_tfidf, df["risco"])

        simular_vies_demografico(clf, vec)
