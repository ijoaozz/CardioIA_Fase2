import os
from pypdf import PdfReader

# Lista com os nomes exatos dos seus arquivos PDF
arquivos_pdf = [
    "Fatores Associados às Doenças Cardiovasculares.pdf",
    "Promoção da Saúde às Doenças Cardiovasculares.pdf"
]

def converter_pdf_para_txt(lista_pdfs):
    for arquivo in lista_pdfs:
        # Define o nome do arquivo de saída trocando a extensão
        nome_txt = arquivo.replace(".pdf", ".txt")
        
        # Verifica se o arquivo PDF realmente existe na pasta
        if not os.path.exists(arquivo):
            print(f"Aviso: O arquivo '{arquivo}' não foi encontrado na pasta atual.")
            continue
            
        try:
            # Inicia o leitor de PDF
            leitor = PdfReader(arquivo)
            texto_completo = ""
            
            # Extrai o texto de página por página
            for pagina in leitor.pages:
                texto_extraido = pagina.extract_text()
                if texto_extraido:
                    texto_completo += texto_extraido + "\n"
            
            # Salva o texto em um arquivo .txt usando codificação UTF-8 (importante para o português)
            with open(nome_txt, "w", encoding="utf-8") as arquivo_txt:
                arquivo_txt.write(texto_completo)
                
            print(f"Sucesso: '{arquivo}' foi convertido e salvo como '{nome_txt}'")
            
        except Exception as erro:
            print(f"Ocorreu um erro ao converter '{arquivo}': {erro}")

# Executa a função
converter_pdf_para_txt(arquivos_pdf)