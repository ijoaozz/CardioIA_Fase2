import os
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Baixa os pacotes de idiomas e separadores do NLTK (só faz o download na primeira vez)
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

arquivos_txt = [
    "docs/Fatores Associados às Doenças Cardiovasculares.txt",
    "docs/Promoção da Saúde às Doenças Cardiovasculares.txt"
]

def limpar_texto(lista_txts):
    # Carrega a lista de palavras comuns em português que queremos ignorar
    stop_words_pt = set(stopwords.words('portuguese'))
    
    for arquivo in lista_txts:
        if not os.path.exists(arquivo):
            print(f"Aviso: '{arquivo}' não encontrado.")
            continue
            
        # Lê o arquivo de texto
        with open(arquivo, 'r', encoding='utf-8') as f:
            texto = f.read()
            
        # 1. Coloca tudo em letras minúsculas
        texto = texto.lower()
        
        # 2. Remove todas as pontuações (vírgulas, pontos, parênteses, etc)
        texto_sem_pontuacao = texto.translate(str.maketrans('', '', string.punctuation))
        
        # 3. Tokenização: separa o texto palavra por palavra
        palavras = word_tokenize(texto_sem_pontuacao, language='portuguese')
        
        # 4. Remove as stopwords e números soltos
        palavras_limpas = [palavra for palavra in palavras if palavra not in stop_words_pt and not palavra.isnumeric()]
        
        # 5. Junta as palavras limpas de volta em um texto único
        texto_final = " ".join(palavras_limpas)
        
        # 6. Salva o resultado em um novo arquivo
        nome_novo_arquivo = arquivo.replace(".txt", "_limpo.txt")
        with open(nome_novo_arquivo, 'w', encoding='utf-8') as f:
            f.write(texto_final)
            
        print(f"Sucesso: O texto foi limpo e salvo como '{nome_novo_arquivo}'")

# Roda a função
limpar_texto(arquivos_txt)