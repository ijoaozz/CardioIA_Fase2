import joblib
import pandas as pd

from utils import carregar_dataset

# Carregar modelo salvo
modelo = joblib.load("models/modelo_cardioia.pkl")
colunas = joblib.load("models/colunas_modelo.pkl")

print("Modelo carregado com sucesso!")

# Carregar dados novamente
df = carregar_dataset()

X = df[colunas]

# Fazer uma previsão com o primeiro paciente
previsao = modelo.predict(X.iloc[[0]])

print("Previsão para o primeiro paciente:", previsao)