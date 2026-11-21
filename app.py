"""
Producto funcional (sección 4.5) como app de Streamlit, para desplegar en
https://share.streamlit.io/

Lee los resultados YA CALCULADOS por los scripts 01-04 (no recalcula el
modelo en cada visita: más rápido y más reproducible). El asistente de IA
generativa sí corre en vivo contra Groq, usando la API key desde
st.secrets (nunca hardcodeada, nunca impresa en pantalla).

Reproducir desde cero:
    python scripts/01_build_case_table.py data/helpdesk_raw.csv outputs/case_table.csv
    python scripts/02_eda.py outputs/case_table.csv outputs
    python scripts/03_model.py outputs/case_table.csv outputs
    python scripts/04_build_results_context.py outputs
    streamlit run app.py
"""
import json
import os

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Riesgo de resolución lenta de tickets · Helpdesk",
    page_icon="🎫",
    layout="wide",
)

OUT_DIR = "outputs"


@st.cache_data
def load_results():
    with open(f"{OUT_DIR}/verified_results.json", encoding="utf-8") as f:
        verified = json.load(f)
    model_cmp = pd.read_csv(f"{OUT_DIR}/model_comparison.csv")
    feat_imp = pd.read_csv(f"{OUT_DIR}/feature_importance.csv")
    genai_log = None
    genai_path = f"{OUT_DIR}/genai_log.json"
    if os.path.exists(genai_path):
        with open(genai_path, encoding="utf-8") as f:
            genai_log = json.load(f)
    return verified, model_cmp, feat_imp, genai_log


try:
    R, model_cmp, feat_imp, genai_log = load_results()
except FileNotFoundError:
    st.error(
        "Faltan los archivos de resultados en `outputs/`. Corre primero los "
        "scripts 01 a 04 (ver instrucciones al inicio de este archivo o el "
        "README) para generar `verified_results.json`, `model_comparison.csv` "
        "y `feature_importance.csv`."
    )
    st.stop()

# ---------------------------------------------------------------- header
st.title("¿Qué tickets de mesa de ayuda se van a demorar?")
st.caption(
    f"Dataset Helpdesk (Verenich, 2016 · Mendeley Data · Licencia MIT) — "
    f"{R['dataset']['n_casos']} casos, {R['dataset']['n_eventos']} eventos, "
    f"{R['dataset']['periodo']}"
)

# ---------------------------------------------------------------- KPIs
c1, c2, c3, c4 = st.columns(4)
c1.metric("Duración mediana", f"{R['eda']['duracion_mediana_horas']:.0f} h")
c2.metric(
    "Tickets con reproceso",
    f"{R['eda']['tasa_reproceso']:.1f}%",
    help=f"Tardan {R['eda']['factor_reproceso_sobre_duracion']:.1f}x más (mediana) que los que no.",
)
c3.metric(
    "AUC del modelo",
    f"{R['modelo']['auc_regresion_logistica']:.2f}",
    help=f"{R['modelo']['mejor_modelo']}, fuera de muestra (2012).",
)
c4.metric(
    "Señal más fuerte",
    f"{R['modelo']['importancia_variable_principal_pct']:.0f}%",
    help=R["modelo"]["variable_mas_importante"],
)

st.divider()

# ---------------------------------------------------------------- EDA
st.subheader("Hallazgos del análisis exploratorio")
col_a, col_b, col_c = st.columns(3)
with col_a:
    st.image(f"{OUT_DIR}/fig1_duration_distribution.png", use_container_width=True)
with col_b:
    st.image(f"{OUT_DIR}/fig2_rework_vs_duration.png", use_container_width=True)
with col_c:
    st.image(f"{OUT_DIR}/fig3_slowrate_by_weekday.png", use_container_width=True)
st.caption(
    "Sábado y domingo se excluyen del gráfico por día (56 de "
    f"{R['dataset']['n_casos']} casos — muestra insuficiente, domingo tiene 1 solo caso)."
)

st.divider()

# ---------------------------------------------------------------- Modelo
st.subheader("Modelo: ¿vale la pena frente a reglas simples?")
st.caption(
    "Predicción justo después del 2º evento del ticket · entrenado 2010-2011, "
    "probado fuera de muestra en 2012 · excluye el 4,7% de casos donde el 2º "
    "evento coincide con el cierre (evita fuga de información)."
)
col_d, col_e = st.columns([3, 2])
with col_d:
    st.dataframe(model_cmp.set_index("modelo"), use_container_width=True)
with col_e:
    st.image(f"{OUT_DIR}/fig4_roc_curve.png", use_container_width=True)

st.markdown("**¿Qué variable pesa más?**")
st.image(f"{OUT_DIR}/fig5_feature_importance.png", use_container_width=False)

st.divider()

# ---------------------------------------------------------------- GenAI
st.subheader("Asistente de IA generativa")
st.caption(
    "Responde SOLO con los resultados verificados de este proyecto. Cada "
    "respuesta se contrasta automáticamente contra las cifras reales."
)

if genai_log:
    st.markdown("**Preguntas ya evaluadas:**")
    for item in genai_log["interacciones"]:
        ok = item["control_de_cifras"]["todas_las_cifras_verificadas"]
        with st.expander(item["pregunta"]):
            st.write(item["respuesta"])
            if ok:
                st.success("Todas las cifras citadas coinciden con los resultados verificados.")
            else:
                st.warning(
                    "Cifras no verificables: "
                    + ", ".join(item["control_de_cifras"]["numeros_no_verificables"])
                )
else:
    st.info(
        "Aún no se generó `outputs/genai_log.json` (evidencia precomputada). "
        "Puedes igual probar el asistente en vivo abajo si configuraste "
        "`GROQ_API_KEY` en Secrets."
    )

st.markdown("**Pregúntale algo nuevo (en vivo):**")
user_question = st.text_input(
    "Pregunta de negocio sobre este proyecto",
    placeholder="ej. ¿Conviene priorizar tickets abiertos los viernes?",
)

if user_question:
    groq_key = st.secrets.get("GROQ_API_KEY") if hasattr(st, "secrets") else None
    if not groq_key:
        st.warning(
            "No hay `GROQ_API_KEY` configurada en Secrets de esta app. "
            "En Streamlit Community Cloud: Settings → Secrets → agrega "
            '`GROQ_API_KEY = "tu_key"`. Localmente: crea '
            "`.streamlit/secrets.toml` (está en .gitignore, nunca se sube)."
        )
    else:
        from groq import Groq

        system_prompt = (
            "Eres un asistente de análisis de datos para un proyecto académico "
            "de gestión de incidentes de mesa de ayuda. SOLO puedes afirmar "
            "números que estén EXACTAMENTE en este bloque de resultados "
            "verificados (calculados con código, no por ti). Si te preguntan "
            "algo que no está aquí, dilo explícitamente, nunca inventes ni "
            "estimes.\n\nRESULTADOS VERIFICADOS:\n"
            + json.dumps(R, indent=2, ensure_ascii=False)
        )
        with st.spinner("Consultando al asistente..."):
            try:
                client = Groq(api_key=groq_key)
                resp = client.chat.completions.create(
                    model="openai/gpt-oss-120b",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_question},
                    ],
                    temperature=0.1,
                    max_tokens=400,
                )
                st.write(resp.choices[0].message.content)
            except Exception as e:
                st.error(f"No se pudo consultar a Groq: {e}")

st.divider()
st.caption(
    "Proyecto Final · Diplomado IA Generativa y ML · Fuente: Verenich, I. (2016). "
    "Helpdesk [Data set]. Mendeley Data. DOI 10.17632/39bp3vv62t.1 (Licencia MIT)."
)
