import pandas as pd

def carregar_dataset():
    df = pd.read_csv("dataset/heart.csv")

    df = df.rename(columns={
        "age": "idade",
        "sex": "sexo",
        "cp": "tipo_dor_peito",
        "trestbps": "pressao_repouso",
        "chol": "colesterol",
        "fbs": "glicose_jejum",
        "restecg": "eletrocardiograma",
        "thalach": "freq_cardiaca_max",
        "exang": "angina_exercicio",
        "oldpeak": "depressao_st",
        "slope": "inclinacao_st",
        "ca": "num_vasos_principais",
        "thal": "talassemia",
        "target": "doenca_cardiaca"
    })

    return df