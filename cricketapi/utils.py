import pickle
import os

# Cargar el modelo desde un archivo .pkl
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELO_LG = os.path.join(BASE_DIR, "modelo_lg.pkl")
MODELO_UMAP = os.path.join(BASE_DIR, "umap_model.pkl")

with open(MODELO_LG, "rb") as archivo:
    modelo_lg = pickle.load(archivo)

with open(MODELO_UMAP, "rb") as archivo:
    modelo_umap = pickle.load(archivo)

def umap_transform(array):
    umap_inf = modelo_umap.transform([array])
    return umap_inf

def predecir(valores):
    prediccion = modelo_lg.predict([valores])
    return prediccion[0]

def predecir_proba(valores):
    prediccion = modelo_lg.predict_proba([valores])
    return prediccion[0]