"""
nlp.py — Módulo de Processamento de Linguagem Natural para o CardioIA
Responsável pela extração de sintomas a partir de frases em linguagem natural
e pela predição de doenças associadas com base em mapa de conhecimento clínico.

Versão: 2.0 (Fase 2 — refatoração completa)
"""

import csv
import re
import logging
import unicodedata
from collections import Counter
from pathlib import Path

# Configuração do sistema de logging para rastreabilidade
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("CardioIA.nlp")


# ---------------------------------------------------------------------------
# MAPA INTERNO DE VARIAÇÕES DE SINTOMAS
# Mapeia termos canônicos (chave) para todas as variações conhecidas (lista).
# Utilizado como vocabulário clínico de fallback quando o CSV não cobre a frase.
# ---------------------------------------------------------------------------

MAPA_VARIACOES: dict[str, list[str]] = {
    "dor no peito": [
        r"dor no peito",
        r"aperto no peito",
        r"pressao no peito",
        r"aperto.{0,25}peito",      # captura "aperto forte no meio do peito"
        r"dor.{0,25}peito",         # captura "dor intensa no peito"
        r"dor toracica",
        r"aperto no meio",
    ],
    "falta de ar": [
        r"falta de ar",
        r"dificuldade para respirar",
        r"dificuldade em respirar",
        r"sem folego",               # "sem fôlego"
        r"sem ar",
        r"sufoco",
        r"ar some",                  # "sinto que o ar some"
        r"ar sumindo",               # "sinto o ar sumindo"
        r"pra respirar",             # "preciso parar pra respirar"
        r"nao consigo respirar",
        r"falta o ar",               # "falta o ar quando subo"
        r"deixa sem folego",         # "me deixa sem fôlego"
        r"dificuldade de respirar",
    ],
    "cansaco": [
        r"cansaco",
        r"fadiga",
        r"cansaco extremo",
        r"muito cansado",
        r"exaustao",
        r"sem energia",
        r"fraqueza",
    ],
    "tontura": [
        r"tontura",
        r"tonteira",
        r"cabeca girando",
        r"desequilibrio",
    ],
    "palpitacao": [
        r"palpitacao",
        r"palpitacoes",
        r"bater forte do coracao",
        r"coracao disparado",
        r"trancos",                  # "trancos e palpitações"
        r"coracao batendo forte",
        r"coracao na garganta",      # expressão coloquial comum
        r"sinto o coracao",
    ],
    "batimento acelerado": [
        r"coracao acelerado",        # "coração acelerado" (solicitado)
        r"batimento acelerado",
        r"taquicardia",
        r"coracao disparado",
        r"coracao muito acelerado",
        r"batimento muito rapido",
        r"coracao.*acelerado",       # "coração fica acelerado"
        r"acelera sozinho",          # "às vezes acelera sozinho"
        r"coracao dispara",          # "o coração dispara do nada"
        r"bpm alto",
        r"pulso acelerado",
    ],
    "batimento irregular": [
        r"batimento irregular",
        r"batimentos irregulares",
        r"coracao irregular",
        r"arritmia",
        r"batendo.*irregular",       # captura "coração está batendo de forma irregular"
        r"forma irregular",
    ],
    "inchaco nas pernas": [
        r"inchaco nas pernas",
        r"pernas inchadas",
        r"edema",
        r"inchaco nas duas pernas",  # captura "inchaço nas duas pernas"
        r"pernas pesadas e inchadas",
        r"pernas.*inchadas",
        r"inchaco.*pernas",
    ],
    "suor frio": [
        r"suor frio",
        r"suando frio",
        r"transpiracao fria",
    ],
    "dor no braco esquerdo": [
        r"dor no braco esquerdo",    # "braço esquerdo" (solicitado)
        r"dor no braco",             # "dor no braço" (solicitado)
        r"formigamento no braco esquerdo",
        r"dormencia no braco esquerdo",
        r"irradiando.*braco",        # "dor irradiando pro braço esquerdo"
        r"comecou no braco esquerdo",
        r"braco esquerdo.*dor",
        r"dor.*braco esquerdo",
        r"formigamento no braco",    # "formigamento no braço"
        r"dormencia no braco",       # "dormência no braço"
    ],
    "nausea": [
        r"nausea",
        r"enjoo",
        r"vontade de vomitar",
        r"mal estar gastrico",
    ],
    "desmaio": [
        r"desmaio",
        r"desmaiei",
        r"perda de consciencia",
        r"quase desmaiou",
    ],
    "pressao alta": [
        r"pressao alta",
        r"hipertensao",
        r"pressao elevada",
    ],
    "pressao baixa": [
        r"pressao baixa",
        r"hipotensao",
        r"pressao caiu",
    ],
    "tosse persistente": [
        r"tosse persistente",
        r"tosse constante",
        r"tosse que nao passa",
        r"tosse cronica",
    ],
    "dor de cabeca": [
        r"dor de cabeca",
        r"cefaleia",
        r"cabeca doendo",
        r"enxaqueca",
    ],
    "visao turva": [
        r"visao turva",
        r"visao embacada",
        r"dificuldade para enxergar",
        r"turvacao visual",
    ],
}


def _normalizar(texto: str) -> str:
    """
    Remove acentos e converte para minúsculas para matching robusto.
    Exemplo: 'Palpitação' → 'palpitacao'

    Args:
        texto: String original com possíveis acentos e maiúsculas.

    Returns:
        String normalizada sem acentos em letras minúsculas.
    """
    nfkd = unicodedata.normalize("NFKD", texto)
    sem_acentos = "".join(c for c in nfkd if not unicodedata.combining(c))
    return sem_acentos.lower()


def extrair_sintomas(frase: str) -> list[str]:
    """
    Extrai sintomas reconhecidos de uma frase em linguagem natural.

    Aplica normalização Unicode antes do matching para garantir robustez
    frente a variações de acentuação e capitalização. Utiliza re.search
    para permitir que o padrão ocorra em qualquer posição da frase.

    Args:
        frase: Relato do paciente em linguagem natural.

    Returns:
        Lista de termos canônicos dos sintomas identificados (sem duplicatas).

    Exemplo:
        >>> extrair_sintomas("Estou com falta de ar e palpitações")
        ['falta de ar', 'palpitação']
    """
    if not frase or not isinstance(frase, str):
        logger.warning("Frase inválida recebida — retornando lista vazia.")
        return []

    frase_normalizada = _normalizar(frase)
    logger.debug("Frase normalizada: %s", frase_normalizada)

    sintomas_encontrados: list[str] = []

    for sintoma_canonico, padroes in MAPA_VARIACOES.items():
        for padrao in padroes:
            padrao_normalizado = _normalizar(padrao)
            if re.search(padrao_normalizado, frase_normalizada):
                sintomas_encontrados.append(sintoma_canonico)
                logger.debug("Sintoma detectado: '%s' via padrão '%s'", sintoma_canonico, padrao)
                break  # evita duplicata do mesmo sintoma canônico

    if not sintomas_encontrados:
        logger.info("Nenhum sintoma reconhecido na frase: '%s'", frase[:60])

    return sintomas_encontrados


def carregar_mapa(caminho: str = "mapa_sintomas.csv") -> dict:
    """
    Carrega o mapa de conhecimento clínico a partir de um arquivo CSV.

    O arquivo deve conter as colunas: sintoma_1, sintoma_2, doenca_associada.
    Cada linha representa uma combinação de sintomas associada a uma doença.

    Args:
        caminho: Caminho para o arquivo CSV do mapa de sintomas.

    Returns:
        Dicionário onde cada chave é um sintoma normalizado e o valor
        é a lista de doenças associadas a ele.

    Raises:
        FileNotFoundError: Se o arquivo CSV não for encontrado.
        KeyError: Se o CSV não contiver as colunas esperadas.
    """
    mapa: dict[str, list[str]] = {}

    try:
        with open(caminho, newline="", encoding="utf-8") as arquivo:
            leitor = csv.DictReader(arquivo)

            # Valida presença das colunas obrigatórias
            if leitor.fieldnames is None:
                raise ValueError("Arquivo CSV vazio ou sem cabeçalho.")

            colunas_esperadas = {"sintoma_1", "sintoma_2", "doenca_associada"}
            colunas_presentes = set(leitor.fieldnames)

            if not colunas_esperadas.issubset(colunas_presentes):
                raise KeyError(
                    f"Colunas ausentes no CSV: {colunas_esperadas - colunas_presentes}"
                )

            for linha in leitor:
                doenca = linha["doenca_associada"].strip()
                for campo in ("sintoma_1", "sintoma_2"):
                    sintoma = _normalizar(linha[campo].strip())
                    if sintoma:
                        mapa.setdefault(sintoma, []).append(doenca)

        logger.info("Mapa carregado: %d entradas de sintomas em '%s'", len(mapa), caminho)

    except FileNotFoundError:
        logger.error("Arquivo não encontrado: %s", caminho)
        raise
    except KeyError as e:
        logger.error("Estrutura inválida no CSV: %s", e)
        raise

    return mapa


def prever_doenca(sintomas: list[str], mapa: dict, top_n: int = 3) -> list[dict]:
    """
    Prediz doenças associadas aos sintomas com pontuação ponderada e filtro top-N.

    Lógica de pontuação:
      - Cada sintoma que bate sozinho com uma linha do CSV contribui +1.
      - Quando os DOIS sintomas de uma linha batem simultaneamente, a doença
        recebe +2 bônus adicional (peso duplo por combinação confirmada).
        Isso garante que "dor no peito + suor frio → Infarto" pontue muito
        acima de "dor no peito" isolado.

    Args:
        sintomas: Lista de sintomas canônicos extraídos da frase.
        mapa: Dicionário carregado por carregar_mapa().
        top_n: Quantidade máxima de doenças retornadas (padrão: 3).

    Returns:
        Lista de até top_n dicionários com 'doenca' e 'corroboracoes',
        ordenados do maior para o menor score.

    Exemplo:
        >>> prever_doenca(['dor no peito', 'suor frio'], mapa)
        [{'doenca': 'Infarto', 'corroboracoes': 9}, ...]
    """
    if not sintomas:
        logger.info("Nenhum sintoma fornecido para predição.")
        return []

    # Normaliza todos os sintomas extraídos uma única vez
    sintomas_norm = [_normalizar(s) for s in sintomas]

    contagem: Counter = Counter()

    # --- Pontuação individual: +1 por sintoma que bate ---
    for s_norm in sintomas_norm:
        for chave_mapa, doencas in mapa.items():
            if s_norm in chave_mapa or chave_mapa in s_norm:
                for doenca in doencas:
                    contagem[doenca] += 1

    # --- Peso duplo: +2 quando AMBOS sintomas de uma linha CSV batem ---
    # Percorre o CSV original para verificar pares completos.
    # O mapa já foi construído a partir do CSV, mas para verificar pares
    # precisamos correlacionar sintoma_1 e sintoma_2 da mesma linha.
    # Reconstruímos o raciocínio: se a doença recebeu ≥2 pontos de sintomas
    # diferentes que são conhecidos como par clínico, aplica o bônus.
    # Implementação: percorremos o mapa por pares implícitos (duas chaves
    # distintas que compartilham a mesma doença e ambas batem com os sintomas).
    doencas_por_sintoma: dict[str, set] = {}
    for chave_mapa, doencas in mapa.items():
        # Verifica se este sintoma do mapa bate com algum sintoma extraído
        bateu = any(s_norm in chave_mapa or chave_mapa in s_norm
                    for s_norm in sintomas_norm)
        if bateu:
            for doenca in doencas:
                doencas_por_sintoma.setdefault(doenca, set()).add(chave_mapa)

    # Aplica +2 para cada doença corroborada por ≥2 chaves distintas do mapa
    for doenca, chaves_que_bateram in doencas_por_sintoma.items():
        if len(chaves_que_bateram) >= 2:
            contagem[doenca] += 2  # bônus de combinação confirmada
            logger.debug(
                "Bônus de par aplicado a '%s': chaves %s", doenca, chaves_que_bateram
            )

    if not contagem:
        logger.info("Nenhuma doença associada encontrada para os sintomas: %s", sintomas)
        return []

    # Filtro top_n: retorna apenas as top_n doenças com maior score
    resultado = [
        {"doenca": doenca, "corroboracoes": n}
        for doenca, n in contagem.most_common(top_n)
    ]

    logger.info(
        "Predição concluída: top %d de %d candidatas para sintomas %s",
        len(resultado), len(contagem), sintomas
    )

    return resultado
