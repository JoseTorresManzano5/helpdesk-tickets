"""
Producto funcional (sección 4.5) como app de Streamlit, para desplegar en
https://share.streamlit.io/

Lee los resultados YA CALCULADOS por los scripts 01-04 (no recalcula el
modelo en cada visita: más rápido y más reproducible). Los gráficos son
interactivos (Plotly: hover, zoom) construidos a partir de esos mismos
resultados -- no se inventa ningún número aquí. El asistente de IA
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
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Riesgo de resolución lenta de tickets · Helpdesk",
    page_icon="🎫",
    layout="wide",
)

OUT_DIR = "outputs"

# paleta validada (dataviz skill, references/palette.md)
C = {
    "blue": "#2a78d6", "orange": "#eb6834",
    "good": "#0ca30c", "critical": "#d03b3b",
    "text2": "#52514e", "grid": "#e1e0d9", "muted": "#898781",
}

PLOTLY_LAYOUT = dict(
    font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", color="#0b0b0b"),
    plot_bgcolor="#fcfcfb", paper_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=10, r=10, t=40, b=10),
    xaxis=dict(gridcolor=C["grid"], zeroline=False),
    yaxis=dict(gridcolor=C["grid"], zeroline=False),
)


@st.cache_data
def load_results():
    with open(f"{OUT_DIR}/verified_results.json", encoding="utf-8") as f:
        verified = json.load(f)
    model_cmp = pd.read_csv(f"{OUT_DIR}/model_comparison.csv")
    feat_imp = pd.read_csv(f"{OUT_DIR}/feature_importance.csv")
    case_df = pd.read_csv(f"{OUT_DIR}/case_table.csv", parse_dates=["open_timestamp"])
    roc_df = None
    roc_path = f"{OUT_DIR}/roc_curve.csv"
    if os.path.exists(roc_path):
        roc_df = pd.read_csv(roc_path)
    genai_log = None
    genai_path = f"{OUT_DIR}/genai_log.json"
    if os.path.exists(genai_path):
        with open(genai_path, encoding="utf-8") as f:
            genai_log = json.load(f)
    return verified, model_cmp, feat_imp, case_df, roc_df, genai_log


try:
    R, model_cmp, feat_imp, case_df, roc_df, genai_log = load_results()
except FileNotFoundError:
    st.error(
        "Faltan los archivos de resultados en `outputs/`. Corre primero los "
        "scripts 01 a 04 (ver instrucciones al inicio de este archivo o el "
        "README) para generar `verified_results.json`, `model_comparison.csv`, "
        "`feature_importance.csv` y `case_table.csv`."
    )
    st.stop()

# ---------------------------------------------------------------- header
st.title("🎫 ¿Qué tickets de mesa de ayuda se van a demorar?")
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
    delta=f"{R['eda']['factor_reproceso_sobre_duracion']:.1f}x más lento",
    delta_color="inverse",
    help="Mediana de duración con reproceso vs. sin reproceso.",
)
c3.metric(
    "AUC del modelo",
    f"{R['modelo']['auc_regresion_logistica']:.2f}",
    help=f"{R['modelo']['mejor_modelo']}, fuera de muestra (2012). Azar = 0.50.",
)
c4.metric(
    "Señal más fuerte",
    f"{R['modelo']['importancia_variable_principal_pct']:.0f}%",
    help=R["modelo"]["variable_mas_importante"],
)

tab_eda, tab_modelo, tab_genai = st.tabs(
    ["📊 Exploración de datos", "🤖 Modelo predictivo", "💬 Asistente de IA"]
)

# ================================================================== EDA
with tab_eda:
    st.subheader("Distribución del tiempo de resolución")
    dur_clip = case_df["duration_hours"].clip(upper=case_df["duration_hours"].quantile(0.98))
    fig1 = px.histogram(
        x=dur_clip, nbins=40,
        labels={"x": "Duración de resolución (horas)"},
        color_discrete_sequence=[C["blue"]],
    )
    fig1.add_vline(
        x=R["eda"]["duracion_p75_horas"], line_dash="dash", line_color=C["critical"],
        annotation_text=f"umbral p75 ({R['eda']['duracion_p75_horas']:.0f} h)",
        annotation_font_color=C["critical"],
    )
    fig1.update_layout(**PLOTLY_LAYOUT, yaxis_title="Nº de tickets", bargap=0.05, height=360)
    st.plotly_chart(fig1, use_container_width=True)
    st.caption(
        "La mayoría de los tickets se resuelve rápido, pero una cola larga se "
        "demora semanas — por eso se usa un umbral (p75) en vez de la hora exacta."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("El reproceso casi cuadruplica la demora")
        box_df = case_df.assign(
            grupo=case_df["had_rework"].map({0: "Sin reproceso", 1: "Con reproceso"})
        )
        fig2 = px.box(
            box_df, x="grupo", y="duration_hours", color="grupo",
            color_discrete_map={"Sin reproceso": C["good"], "Con reproceso": C["critical"]},
            points=False,
        )
        fig2.update_layout(**PLOTLY_LAYOUT, showlegend=False, xaxis_title="",
                            yaxis_title="Duración (horas)", height=340)
        st.plotly_chart(fig2, use_container_width=True)

    with col_b:
        st.subheader("Riesgo por día de apertura (lun–vie)")
        wk = case_df[case_df["open_weekday"] < 5].copy()
        dias = {0: "Lun", 1: "Mar", 2: "Mié", 3: "Jue", 4: "Vie"}
        train_p75 = case_df.loc[
            pd.to_datetime(case_df["open_timestamp"]).dt.year < 2012, "duration_hours"
        ].quantile(0.75)
        wk["slow"] = wk["duration_hours"] > train_p75
        rate = (wk.groupby("open_weekday")["slow"].mean() * 100).reindex(range(5))
        rate.index = [dias[i] for i in rate.index]
        colors = [C["critical"] if v == rate.max() else C["blue"] for v in rate.values]
        fig3 = go.Figure(go.Bar(x=rate.index, y=rate.values, marker_color=colors,
                                 text=[f"{v:.0f}%" for v in rate.values], textposition="outside"))
        fig3.update_layout(**PLOTLY_LAYOUT, yaxis_title="% tickets lentos", height=340)
        st.plotly_chart(fig3, use_container_width=True)
        st.caption(
            f"Sáb/dom excluidos (56 de {R['dataset']['n_casos']} casos — "
            "muestra insuficiente, domingo tiene 1 solo caso)."
        )

# =============================================================== MODELO
with tab_modelo:
    st.subheader("Modelo: ¿vale la pena frente a reglas simples?")
    st.caption(
        "Predicción justo después del 2º evento del ticket · entrenado 2010-2011, "
        "probado fuera de muestra en 2012 · excluye el 4,7% de casos donde el 2º "
        "evento coincide con el cierre (evita fuga de información)."
    )

    def highlight_best(row):
        is_model = row["modelo"] in ("Regresión logística", "Random Forest")
        return ["background-color: #e8f3ff" if is_model else "" for _ in row]

    st.dataframe(
        model_cmp.set_index("modelo").style.apply(highlight_best, axis=1).format("{:.3f}", na_rep="—"),
        use_container_width=True,
    )

    col_c, col_d = st.columns([3, 2])
    with col_c:
        st.markdown("**Curva ROC (fuera de muestra, 2012)**")
        if roc_df is not None:
            fig4 = go.Figure()
            for name, color in [("Regresión logística", C["blue"]), ("Random Forest", C["orange"])]:
                sub = roc_df[roc_df["modelo"] == name]
                auc_val = sub["auc"].iloc[0]
                fig4.add_trace(go.Scatter(
                    x=sub["fpr"], y=sub["tpr"], mode="lines", name=f"{name} (AUC={auc_val:.2f})",
                    line=dict(color=color, width=2),
                    hovertemplate="FPR=%{x:.2f}<br>TPR=%{y:.2f}<extra>" + name + "</extra>",
                ))
            fig4.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Azar (AUC=0.50)",
                                       line=dict(color=C["muted"], width=1, dash="dash")))
            fig4.update_layout(**PLOTLY_LAYOUT, xaxis_title="Falsos positivos",
                                yaxis_title="Verdaderos positivos", height=380,
                                legend=dict(x=0.55, y=0.08))
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.image(f"{OUT_DIR}/fig4_roc_curve.png", use_container_width=True)
            st.caption("(Versión estática — corre scripts/03_model.py de nuevo para la interactiva.)")

    with col_d:
        st.markdown("**¿Qué variable pesa más?**")
        top = feat_imp.head(8).sort_values("importance")
        fig5 = go.Figure(go.Bar(
            x=top["importance"] * 100, y=top["feature"], orientation="h",
            marker_color=C["blue"],
            text=[f"{v:.1f}%" for v in top["importance"] * 100], textposition="outside",
        ))
        fig5.update_layout(**PLOTLY_LAYOUT, xaxis_title="Importancia (%)", height=380)
        st.plotly_chart(fig5, use_container_width=True)

# ================================================================ GENAI
with tab_genai:
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
                "estimes. Responde en un párrafo corto de 3 a 5 frases, SIN "
                "tablas ni listas largas -- cita como máximo 2 o 3 cifras "
                "concretas dentro de la prosa, tal cual aparecen abajo. Termina "
                "siempre con una conclusión completa, nunca a la mitad de una "
                "frase.\n\nRESULTADOS VERIFICADOS:\n"
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
                        max_tokens=600,
                    )
                    st.write(resp.choices[0].message.content)
                except Exception as e:
                    st.error(f"No se pudo consultar a Groq: {e}")

st.divider()
st.caption(
    "Proyecto Final · Diplomado IA Generativa y ML · Fuente: Verenich, I. (2016). "
    "Helpdesk [Data set]. Mendeley Data. DOI 10.17632/39bp3vv62t.1 (Licencia MIT)."
)
