# Pega esto como la PRIMERA celda de tu notebook en Google Colab.
# Requisito previo: haber agregado la secret "GROQ_API_KEY" con el ícono
# de llave (panel izquierdo de Colab) y activado "Notebook access".
#
# Esto NUNCA imprime ni muestra la key en pantalla -- ideal para tu
# demostración en vivo frente al jurado.

import os
from google.colab import userdata

os.environ["GROQ_API_KEY"] = userdata.get("GROQ_API_KEY")
print("GROQ_API_KEY cargada desde Colab Secrets (no se muestra en pantalla).")

# A partir de aquí, instala la librería y corre el asistente normalmente:
# !pip install -q groq
# !python 05_genai_assistant.py outputs/verified_results.json outputs/genai_log.json
