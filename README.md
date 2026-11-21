# Riesgo de resolución lenta de tickets — Mesa de ayuda (Helpdesk)

Proyecto Final · Diplomado IA Generativa y ML

**Pregunta analítica:** ¿se puede anticipar, justo después del 2º evento
registrado de un ticket, si su resolución será anormalmente lenta?

**Dataset:** Verenich, I. (2016). *Helpdesk* [Data set]. Mendeley Data.
DOI [10.17632/39bp3vv62t.1](https://doi.org/10.17632/39bp3vv62t.1).
Licencia MIT. 3.804 casos, 13.710 eventos, 2010-01 a 2012-11.

---

## Estructura del repositorio

```
.
├── app.py                      # App de Streamlit (producto final desplegable)
├── requirements.txt
├── data/
│   └── helpdesk_raw.csv        # Dataset original, sin modificar
├── scripts/
│   ├── 01_build_case_table.py  # Agregación evento -> ticket (limpieza)
│   ├── 02_eda.py                # Análisis exploratorio + 3 hallazgos
│   ├── 03_model.py              # Baseline + regresión logística + random forest
│   ├── 04_build_results_context.py  # Consolida resultados verificados (fuente de verdad del LLM)
│   ├── 05_genai_assistant.py    # Asistente de IA generativa (Groq) con control de cifras
│   └── 06_build_dashboard.py    # Dashboard HTML autocontenido (respaldo offline)
├── notebooks/
│   └── colab_pipeline.ipynb     # Mismo pipeline, para correr completo en Google Colab
├── outputs/                     # Se genera al correr los scripts (ya viene con una corrida de ejemplo)
└── .streamlit/
    └── secrets.toml.example     # Plantilla -- NUNCA pongas tu key real aquí
```

## Metodología (resumen para la ficha técnica)

- **Unidad de observación:** el ticket (caso), no el evento individual.
- **Variable objetivo:** `slow_resolution` = 1 si la duración de resolución
  supera el percentil 75 **calculado solo con el periodo de entrenamiento**
  (2010-2011), aplicado igual a 2012 — evita fuga de información del futuro
  hacia la definición del target.
- **Punto de predicción:** justo después del 2º evento del ticket (no en la
  apertura cruda) — con solo hora/día/mes de apertura el modelo no supera el
  azar (AUC < 0,5); este reencuadre, estándar en la literatura de process
  mining para este mismo dataset, sí recupera señal real.
- **Validación:** split cronológico (train 2010-2011 → test 2012), nunca
  aleatorio, para respetar el orden temporal.
- **Limitación documentada:** se excluye el 4,7% de tickets que cierran en
  solo 2 eventos (ahí el 2º evento coincide con el cierre, lo que sería
  fuga de información).
- **Resultado:** regresión logística, AUC = 0,644 fuera de muestra, por
  encima de ambos baselines (clase mayoritaria y regla "si es viernes").

---

## 1) Reproducir en Google Colab

1. Abre `notebooks/colab_pipeline.ipynb` en [Google Colab](https://colab.research.google.com/).
2. En la primera celda, reemplaza la URL del `git clone` por la de tu propio
   repositorio (ver paso 2 abajo para crearlo).
3. Antes de la celda del asistente de IA generativa: agrega tu API key
   gratuita de [console.groq.com](https://console.groq.com) como **Secret**
   de Colab (ícono de llave 🔑, panel izquierdo) con el nombre
   `GROQ_API_KEY`, y activa "Notebook access". **Nunca la escribas en una
   celda como texto plano** — así no se ve en pantalla durante tu defensa.
4. Ejecuta las celdas en orden.

## 2) Subir a GitHub

```bash
cd TU_CARPETA_DEL_PROYECTO
git init
git add .
git commit -m "Proyecto final: riesgo de resolución lenta de tickets (Helpdesk)"
git branch -M main
git remote add origin https://github.com/TU_USUARIO/TU_REPOSITORIO.git
git push -u origin main
```

**Importante:** `.gitignore` ya excluye `.streamlit/secrets.toml` — revisa
con `git status` antes del primer commit que ese archivo (si llegaste a
crearlo localmente) no aparezca listado para subir. Nunca subas tu API key.

## 3) Desplegar en Streamlit Community Cloud

1. Entra a [share.streamlit.io](https://share.streamlit.io/) con tu cuenta
   de GitHub.
2. "New app" → selecciona tu repositorio → rama `main` → archivo principal
   `app.py`.
3. Antes de desplegar (o después, en "Settings → Secrets"), agrega:
   ```toml
   GROQ_API_KEY = "tu_api_key_de_groq"
   ```
   Esta es la forma segura de pasar la key: vive en la configuración de la
   app en Streamlit Cloud, nunca en tu repositorio ni en el código.
4. "Deploy". La app queda en una URL pública tipo
   `https://TU-APP.streamlit.app` — esa es la que usas en vivo para la
   demostración.

La app de Streamlit lee los resultados ya calculados en `outputs/` (no
reentrena el modelo en cada visita — más rápido y más reproducible). Si
cambias el dataset o el modelo, vuelve a correr los scripts 01-04 (local o
en Colab) y sube de nuevo `outputs/` al repo antes de redeploy.

---

## Reproducción 100% local (alternativa a Colab)

```bash
pip install -r requirements.txt
python scripts/01_build_case_table.py data/helpdesk_raw.csv outputs/case_table.csv
python scripts/02_eda.py outputs/case_table.csv outputs
python scripts/03_model.py outputs/case_table.csv outputs
python scripts/04_build_results_context.py outputs
# opcional, requiere GROQ_API_KEY en el entorno:
python scripts/05_genai_assistant.py outputs/verified_results.json outputs/genai_log.json
python scripts/06_build_dashboard.py outputs   # respaldo offline en outputs/dashboard.html
streamlit run app.py
```

## Licencia y cita

Dataset bajo licencia MIT. Cítalo en tu ficha técnica como:

> Verenich, I. (2016). Helpdesk (Version 1) [Data set]. Mendeley Data.
> https://doi.org/10.17632/39bp3vv62t.1
