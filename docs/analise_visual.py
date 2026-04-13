import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud

def gerar_nuvem_palavras():
    # O Python vai procurar automaticamente todos os arquivos que terminam com '_limpo.txt'
    arquivos_limpos = [arquivo for arquivo in os.listdir() if arquivo.endswith('_limpo.txt')]
    
    if len(arquivos_limpos) == 0:
        print("Nenhum arquivo '_limpo.txt' foi encontrado na pasta atual.")
        return

    for arquivo in arquivos_limpos:
        # Lê o texto limpo
        with open(arquivo, 'r', encoding='utf-8') as f:
            texto = f.read()
            
        # Gera a nuvem de palavras
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(texto)
        
        # Cria a figura para exibir
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off") # Remove as bordas com números
        plt.title(f"Nuvem de Palavras: {arquivo}", fontsize=12)
        plt.show() # Mostra a imagem na tela

# Executa a função
gerar_nuvem_palavras()