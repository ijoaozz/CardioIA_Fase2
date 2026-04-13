import pandas as pd

# Configuração do Pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)



# 1. ANÁLISE DO DATASET

def analisar_dados():
    print("\n--- 1. Carregando o dataset heart.csv ---")
    try:
        from utils import carregar_dataset
        df = carregar_dataset()
    except FileNotFoundError:
        print("❌ Erro: Arquivo 'heart.csv' não encontrado.")
        return

    print("\n--- Primeiras 5 linhas ---")
    print(df.head())

    print("\n--- Dados Faltantes ---")
    print(df.isnull().sum())

    print("\n--- Estatísticas ---")
    print(df.describe())

    if 'doenca_cardiaca' in df.columns:
        print("\n--- Distribuição ---")
        print("0 = Saudável | 1 = Doença Cardíaca")
        print(df['doenca_cardiaca'].value_counts())
    else:
        print("⚠️ Coluna 'doenca_cardiaca' não encontrada.")



# 2. LEITURA DO TXT

def ler_frases_txt(caminho):
    try:
        with open(caminho, "r", encoding="utf-8") as arquivo:
            frases = arquivo.readlines()
        return [f.strip() for f in frases if f.strip() != ""]
    except FileNotFoundError:
        print("❌ Arquivo 'sintomas.txt' não encontrado.")
        return []


def _formatar_doencas(resultado):
    """
    Converte a lista de dicts retornada por prever_doenca() em texto legível.

    prever_doenca() retorna top 3 por padrão:
      [{'doenca': 'Infarto', 'corroboracoes': 9}, ...]

    O score inclui:
      +1 por sintoma individual que bate com o mapa
      +2 bônus quando dois sintomas de uma mesma linha do CSV batem juntos
    """
    if not resultado:
        return None
    # Score máximo possível serve de referência para exibir a barra de confiança
    score_max = resultado[0]["corroboracoes"] if resultado else 1
    linhas = []
    for i, item in enumerate(resultado, start=1):
        doenca = item["doenca"]
        score  = item["corroboracoes"]
        # Barra visual de confiança relativa ao top-1
        proporcao = score / score_max
        if proporcao >= 0.75:
            nivel = "███ Alta"
        elif proporcao >= 0.40:
            nivel = "██░ Média"
        else:
            nivel = "█░░ Baixa"
        linhas.append(f"  {i}. {doenca:<28} score: {score:>2}  [{nivel}]")
    return "\n".join(linhas)



# 3. PROCESSAMENTO AUTOMÁTICO

def processar_arquivo():
    print("\n📄 PROCESSANDO ARQUIVO DE SINTOMAS...\n")

    from nlp import extrair_sintomas, carregar_mapa, prever_doenca

    mapa = carregar_mapa()
    frases = ler_frases_txt("sintomas.txt")

    if not frases:
        print("⚠️ Nenhuma frase encontrada.")
        return

    for i, frase in enumerate(frases, start=1):
        print(f"\n🧾 Frase {i}: {frase}")

        sintomas = extrair_sintomas(frase)

        if not sintomas:
            print("❌ Nenhum sintoma reconhecido.")
            continue

        print("✅ Sintomas encontrados:", sintomas)

        # prever_doenca retorna lista de dicts com 'doenca' e 'corroboracoes'
        resultado = prever_doenca(sintomas, mapa)
        texto = _formatar_doencas(resultado)

        if texto:
            print("🩺 Possíveis doenças (por corroboração):")
            print(texto)
        else:
            print("❌ Nenhuma doença associada.")



# 4. SISTEMA INTERATIVO

def sistema_interativo():
    print("\n==============================")
    print("🧠 SISTEMA INTERATIVO DE SINTOMAS")
    print("Digite uma frase com sintomas")
    print("Digite 'sair' para encerrar")
    print("==============================")

    from nlp import extrair_sintomas, carregar_mapa, prever_doenca

    mapa = carregar_mapa()

    while True:
        frase = input("\n👉 Digite o sintoma: ").strip().lower()

        if frase == "sair":
            print("\n👋 Encerrando sistema...")
            break

        if frase == "":
            print("⚠️ Digite algo válido.")
            continue

        sintomas = extrair_sintomas(frase)

        if not sintomas:
            print("❌ Nenhum sintoma reconhecido.")
            continue

        print("✅ Sintomas encontrados:", sintomas)

        resultado = prever_doenca(sintomas, mapa)
        texto = _formatar_doencas(resultado)

        if texto:
            print("🩺 Possíveis doenças (por corroboração):")
            print(texto)
        else:
            print("❌ Nenhuma doença associada.")



# EXECUÇÃO PRINCIPAL

if __name__ == "__main__":
    analisar_dados()
    processar_arquivo()
    sistema_interativo()
