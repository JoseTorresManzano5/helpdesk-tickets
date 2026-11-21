"""Genera notebooks/colab_pipeline.ipynb como JSON crudo (sin depender de
la librería nbformat, que no está instalada en este sandbox). El formato
.ipynb es JSON válido según el esquema nbformat v4; Colab y Jupyter lo
abren igual que si se hubiera creado con nbformat."""
import json

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}

def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src.splitlines(keepends=True)}

cells = [
    md("# Proyecto Final · Riesgo de resolución lenta de tickets (Helpdesk)\n"
       "Diplomado IA Generativa y ML — pipeline reproducible completo: "
       "datos → limpieza → EDA → modelo → asistente de IA generativa.\n\n"
       "Dataset: Verenich, I. (2016). *Helpdesk* [Data set]. Mendeley Data. "
       "DOI 10.17632/39bp3vv62t.1 (Licencia MIT)."),

    md("## 1. Clonar el repositorio\n"
       "Reemplaza la URL por la de tu propio repo de GitHub (ver README "
       "para cómo subirlo)."),
    code('!git clone https://github.com/TU_USUARIO/TU_REPOSITORIO.git\n'
         '%cd TU_REPOSITORIO'),

    md("## 2. Instalar dependencias"),
    code("!pip install -q -r requirements.txt"),

    md("## 3. Construir la tabla de casos (agregación evento -> ticket)"),
    code("!python scripts/01_build_case_table.py data/helpdesk_raw.csv outputs/case_table.csv"),

    md("## 4. Análisis exploratorio (EDA)"),
    code("!python scripts/02_eda.py outputs/case_table.csv outputs"),
    code("from IPython.display import Image, display\n"
         "for f in ['fig1_duration_distribution.png', 'fig2_rework_vs_duration.png',\n"
         "          'fig3_slowrate_by_weekday.png']:\n"
         "    display(Image(filename=f'outputs/{f}'))"),

    md("## 5. Modelo: baseline + regresión logística + random forest\n"
       "Split cronológico (train 2010-2011, test 2012), target definido "
       "SOLO con el periodo de entrenamiento (sin fuga de información)."),
    code("!python scripts/03_model.py outputs/case_table.csv outputs"),
    code("import pandas as pd\n"
         "pd.read_csv('outputs/model_comparison.csv')"),

    md("## 6. Consolidar resultados verificados (fuente de verdad para el LLM)"),
    code("!python scripts/04_build_results_context.py outputs"),

    md("## 7. Asistente de IA generativa (Groq)\n"
       "**Antes de correr esta celda:** agrega tu API key gratuita de "
       "https://console.groq.com como *Secret* de Colab (ícono de llave 🔑 "
       "en el panel izquierdo), nombrándola `GROQ_API_KEY`, y activa "
       "\"Notebook access\". Así nunca queda escrita en texto plano aquí."),
    code('import os\n'
         'from google.colab import userdata\n\n'
         'os.environ["GROQ_API_KEY"] = userdata.get("GROQ_API_KEY")\n'
         'print("GROQ_API_KEY cargada desde Colab Secrets (no se muestra en pantalla).")'),
    code("!pip install -q groq\n"
         "!python scripts/05_genai_assistant.py outputs/verified_results.json outputs/genai_log.json"),

    md("## 8. Producto funcional (dashboard HTML autocontenido)\n"
       "Útil como respaldo offline de la app de Streamlit."),
    code("!python scripts/06_build_dashboard.py outputs"),
    code('from IPython.display import HTML\n'
         "HTML(filename='outputs/dashboard.html')"),

    md("## 9. Siguiente paso: GitHub + Streamlit Community Cloud\n"
       "Ver `README.md` en la raíz del repo para los pasos de despliegue "
       "en https://share.streamlit.io/"),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        "colab": {"provenance": []},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open("notebooks/colab_pipeline.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("Notebook generado en notebooks/colab_pipeline.ipynb")
