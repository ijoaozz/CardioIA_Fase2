import os
import pandas as pd
from collections import Counter

def contar_palavras():
    # Procura os arquivos limpos na pasta atual
    arquivos_limpos = [arquivo for arquivo in os.listdir() if arquivo.endswith('_limpo.txt')]
    
    if not arquivos_limpos:
        print("Nenhum arquivo limpo encontrado.")
        return

    for arquivo in arquivos_limpos:
        # Lê o conteúdo de cada arquivo
        with open(arquivo, 'r', encoding='utf-8') as f:
            texto = f.read()
            
        # Divide o texto em uma lista de palavras individuais
        palavras = texto.split()
        
        # Conta matematicamente quantas vezes cada palavra aparece
        contagem = Counter(palavras)
        
        # Pega apenas as 10 palavras mais comuns
        top_10 = contagem.most_common(10)
        
        # Transforma o resultado em uma tabela (DataFrame) usando o Pandas
        df = pd.DataFrame(top_10, columns=['Palavra', 'Frequência'])
        
        # Imprime o resultado formatado no terminal
        print(f"\n--- Top 10 palavras mais citadas em: {arquivo} ---")
        print(df.to_string(index=False)) # Oculta o índice numérico para ficar mais limpo

# Executa a função
contar_palavras()