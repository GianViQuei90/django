from rest_framework.response import Response
from rest_framework.decorators import api_view
from django.views.decorators.csrf import csrf_exempt
from .utils import predecir, umap_transform, predecir_proba
import json
from langdetect import detect
import os
# from openai import OpenAI
from openai import AzureOpenAI
import numpy as np
from dotenv import load_dotenv

load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"), 
    api_version="2023-05-15",
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    )

def get_embedding(text, model="text-embedding-3-large"):
   text = text.replace("\n", " ")
   embedding = client.embeddings.create(input = [text], model=model).data[0].embedding
   return np.array(embedding, dtype=np.float32)

# @csrf_exempt
@api_view(["POST"])
def api_prediccion(request):
    if request.method == "POST":
        try:
            # Parsear el JSON recibido
            datos = json.loads(request.body)
            texto = datos.get("text")  # Esperamos una lista [x1, x2, x3]

            if not texto or not isinstance(texto, str):
                return Response({"error": "Debes enviar el argumento 'text' como string"}, status=400)

            idioma = detect(texto)
            print("Idioma detectado:", idioma)

            if idioma == 'es':
                language_es = True
            else:
                language_es = False

            embedding = get_embedding(texto, model='text-embedding-3-large')
            embedding_list = embedding.tolist()
            umap_inf = umap_transform(embedding_list)
            print(umap_inf)
            umap1 = umap_inf[0,0]
            print(umap1)
            umap2 = umap_inf[0,1]
            print(umap2)
            # Obtener la predicción
            valores = [umap1, umap2, language_es]
            prediccion = predecir(valores)
            proba = predecir_proba(valores)
            print(proba)
            return Response(
                {
                    "language_es": language_es,
                    "first_10_embedding": embedding_list[:10],
                    "prediccion_no_fraud": bool(prediccion),
                    "prob": proba.tolist() 

                }, 
                 status=200)

        except Exception as e:
            return Response({"error": str(e)}, status=500)

    return Response({"error": "Método no permitido. Usa POST."}, status=405)
