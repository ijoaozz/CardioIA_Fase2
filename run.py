from main import executar_analise
from modelo_preditivo import treinar_modelo
from docs.limpeza_nlp import processar_textos

if __name__ == "__main__":
    executar_analise()
    treinar_modelo()
    processar_textos()