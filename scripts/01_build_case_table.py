"""
Agrega el log de eventos (CaseID, ActivityID, CompleteTimestamp) a nivel de
ticket (caso). Separa explícitamente:
  - features conocidas al ABRIR el ticket (sin fuga de información)
  - variables objetivo / descriptivas que solo se conocen al CERRAR el ticket
Dataset: Helpdesk (Verenich, 2016, Mendeley Data, DOI 10.17632/39bp3vv62t.1)
Licencia: MIT. 3,804 casos / 13,710 eventos documentados en la fuente oficial.
"""
import sys
import pandas as pd

RAW_PATH = sys.argv[1] if len(sys.argv) > 1 else "data/helpdesk_raw.csv"
OUT_PATH = sys.argv[2] if len(sys.argv) > 2 else "outputs/case_table.csv"

df = pd.read_csv(RAW_PATH, parse_dates=["CompleteTimestamp"])

# --- chequeos de calidad (para el entregable de limpieza) ---
n_rows_raw = len(df)
n_dupes = df.duplicated().sum()
n_missing = df.isna().sum().sum()
n_cases_raw = df["CaseID"].nunique()

df = df.sort_values(["CaseID", "CompleteTimestamp"]).reset_index(drop=True)

# --- agregación por caso ---
grp = df.groupby("CaseID")

first_event = grp.first()
last_ts = grp["CompleteTimestamp"].max()
first_ts = grp["CompleteTimestamp"].min()
n_events = grp.size()
n_distinct_activities = grp["ActivityID"].nunique()

# señal de reproceso: una actividad se repite dentro del mismo caso
def has_repeated_activity(s):
    return int(s.duplicated().any())

repeated_activity = grp["ActivityID"].apply(has_repeated_activity)

# --- segundo evento del caso (punto de predicción alternativo: "ticket ya
#     triado", no solo "ticket recién abierto") ---
def nth_row(g, n):
    return g.iloc[n] if len(g) > n else pd.Series({"ActivityID": -1, "CompleteTimestamp": pd.NaT})

second = grp.apply(lambda g: nth_row(g, 1), include_groups=False)

cases = pd.DataFrame({
    "CaseID": first_event.index,
    "first_activity": first_event["ActivityID"].values,
    "open_timestamp": first_ts.values,
    "close_timestamp": last_ts.values,
    "n_events": n_events.values,
    "n_distinct_activities": n_distinct_activities.values,
    "had_rework": repeated_activity.values,
    "second_activity": second["ActivityID"].values,
    "second_timestamp": second["CompleteTimestamp"].values,
}).reset_index(drop=True)

cases["gap_to_second_event_h"] = (
    (cases["second_timestamp"] - cases["open_timestamp"]).dt.total_seconds() / 3600
)
# Para casos con exactamente 2 eventos, el "segundo evento" coincide con el
# cierre del caso -> usar su timestamp como punto de predicción sería fuga
# de información (el gap seria igual a la duracion total). Se marcan para
# excluirlos del modelo que use el punto de prediccion "segundo evento".
cases["second_event_is_closing_event"] = (cases["n_events"] == 2)

cases["duration_hours"] = (
    (cases["close_timestamp"] - cases["open_timestamp"]).dt.total_seconds() / 3600
)

# --- features disponibles SOLO al abrir el ticket (para evitar fuga) ---
cases["open_hour"] = cases["open_timestamp"].dt.hour
cases["open_weekday"] = cases["open_timestamp"].dt.weekday  # 0=lunes
cases["open_month"] = cases["open_timestamp"].dt.month
cases["open_is_weekend"] = cases["open_weekday"].isin([5, 6]).astype(int)

# NOTA IMPORTANTE: la binarización de "slow_resolution" (umbral p75) se hace
# en el script 03 usando SOLO el periodo de entrenamiento (2010-2011), nunca
# con el dataset completo. Calcularla aqui con todos los datos filtraria
# informacion del periodo de prueba (2012) hacia la definicion del target
# -> fuga de informacion. Por eso este script solo deja "duration_hours"
# (continua) lista; el umbral se fija mas adelante, solo con train.

cases.to_csv(OUT_PATH, index=False)

print("=== Chequeos de calidad (datos crudos) ===")
print(f"Filas (eventos) crudas: {n_rows_raw}")
print(f"Casos únicos: {n_cases_raw}")
print(f"Filas duplicadas exactas: {n_dupes}")
print(f"Valores faltantes totales: {n_missing}")
print()
print("=== Tabla de casos (agregada) ===")
print(f"Casos en tabla final: {len(cases)}")
print(f"Duración (horas) -> min={cases.duration_hours.min():.1f} "
      f"p50={cases.duration_hours.median():.1f} "
      f"p75={cases.duration_hours.quantile(0.75):.1f} max={cases.duration_hours.max():.1f}")
print(f"% casos con rework (actividad repetida): {cases.had_rework.mean()*100:.1f}%")
print(f"Guardado en: {OUT_PATH}")
