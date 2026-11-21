"""
EDA sobre la tabla de casos (outputs/case_table.csv).
Genera estadísticas y gráficos para sustentar al menos 3 hallazgos (sección 4.2).
"""
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

IN_PATH = sys.argv[1] if len(sys.argv) > 1 else "outputs/case_table.csv"
OUT_DIR = sys.argv[2] if len(sys.argv) > 2 else "outputs"

sns.set_theme(style="whitegrid", font_scale=1.05)
PALETTE = {"ok": "#2E86AB", "slow": "#E94F37", "neutral": "#6C757D"}

df = pd.read_csv(IN_PATH, parse_dates=["open_timestamp", "close_timestamp"])

# Umbral de "resolución lenta" definido SOLO con el periodo 2010-2011
# (igual que en el modelo, script 03) -> evita fuga de información del
# periodo de prueba (2012) hacia la definición del target.
TRAIN_P75 = df.loc[df["open_timestamp"].dt.year < 2012, "duration_hours"].quantile(0.75)
df["slow_resolution"] = (df["duration_hours"] > TRAIN_P75).astype(int)
print(f"Umbral 'resolución lenta' (p75 de 2010-2011): {TRAIN_P75:.1f} horas\n")

print("=== Hallazgo 1: distribución de la duración de resolución ===")
print(df["duration_hours"].describe())
skew = df["duration_hours"].skew()
print(f"Asimetría (skew): {skew:.2f} (positivo = cola larga hacia la derecha)")

fig, ax = plt.subplots(figsize=(7, 4.2))
sns.histplot(df["duration_hours"], bins=60, color=PALETTE["neutral"], ax=ax)
ax.axvline(df["duration_hours"].quantile(0.75), color=PALETTE["slow"], ls="--",
           label="Umbral p75 (resolución lenta)")
ax.set_xlabel("Duración de resolución (horas)")
ax.set_ylabel("Nº de tickets")
ax.set_title("Hallazgo 1 · La mayoría de los tickets se resuelve rápido,\npero una cola larga se demora semanas")
ax.legend()
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/fig1_duration_distribution.png", dpi=150)
plt.close(fig)

print("\n=== Hallazgo 2: reproceso (rework) vs duración ===")
g = df.groupby("had_rework")["duration_hours"].median()
print(g)
lift = g[1] / g[0]
print(f"Los tickets con reproceso tardan {lift:.1f}x más (mediana) que los que no")

fig, ax = plt.subplots(figsize=(6, 4.2))
sns.boxplot(data=df, x="had_rework", y="duration_hours",
            palette=[PALETTE["ok"], PALETTE["slow"]], ax=ax, showfliers=False)
ax.set_xticklabels(["Sin reproceso", "Con reproceso"])
ax.set_xlabel("")
ax.set_ylabel("Duración de resolución (horas)")
ax.set_title("Hallazgo 2 · El reproceso casi duplica el tiempo\nde resolución del ticket")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/fig2_rework_vs_duration.png", dpi=150)
plt.close(fig)

print("\n=== Hallazgo 3: tasa de resolución lenta por día de apertura ===")
n_weekend = int((df["open_weekday"] >= 5).sum())
print(f"Aviso de muestra: sábado+domingo solo tienen {n_weekend} de {len(df)} casos "
      f"({n_weekend/len(df)*100:.1f}%) -> se EXCLUYEN de la comparación por tamaño "
      f"insuficiente (ej. domingo tiene 1 solo caso, cualquier tasa ahí es ruido, no señal).")

dias = ["Lun", "Mar", "Mié", "Jue", "Vie"]
weekday_df = df[df["open_weekday"] < 5]
rate_by_day = weekday_df.groupby("open_weekday")["slow_resolution"].mean()
rate_by_day.index = [dias[i] for i in rate_by_day.index]
print(rate_by_day)
spread = (rate_by_day.max() - rate_by_day.min()) * 100
print(f"Diferencia entre el día más alto y más bajo: {spread:.1f} puntos porcentuales "
      "(señal moderada, no concluyente por sí sola)")

fig, ax = plt.subplots(figsize=(7, 4.2))
colors = [PALETTE["slow"] if v > rate_by_day.mean() else PALETTE["ok"] for v in rate_by_day.values]
ax.bar(rate_by_day.index, rate_by_day.values * 100, color=colors)
ax.axhline(rate_by_day.mean() * 100, color=PALETTE["neutral"], ls="--", lw=1,
           label="Promedio (solo días hábiles)")
ax.set_ylabel("% de tickets con resolución lenta")
ax.set_title("Hallazgo 3 · El día de apertura (lun-vie) se asocia\ncon una variación moderada en el riesgo de demora")
ax.legend()
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/fig3_slowrate_by_weekday.png", dpi=150)
plt.close(fig)

print("\n=== Hallazgo 4 (contexto, no predictor): complejidad del caso vs duración ===")
corr = df["n_distinct_activities"].corr(df["duration_hours"])
print(f"Correlación nº de actividades distintas vs duración: {corr:.2f}")
print("(Esta variable solo se conoce al cerrar el caso -> NO se usa como predictor,")
print(" se reporta solo como hallazgo descriptivo del proceso)")

summary = {
    "duration_median_h": float(df.duration_hours.median()),
    "duration_p75_h": float(df.duration_hours.quantile(0.75)),
    "rework_rate": float(df.had_rework.mean()),
    "rework_duration_lift_x": float(lift),
    "slowrate_best_day": rate_by_day.idxmin(),
    "slowrate_worst_day": rate_by_day.idxmax(),
    "slowrate_weekday_spread_pp": float(spread),
    "n_weekend_cases_excluded": n_weekend,
    "complexity_duration_corr": float(corr),
}
pd.Series(summary).to_json(f"{OUT_DIR}/eda_summary.json", indent=2)
print(f"\nResumen guardado en {OUT_DIR}/eda_summary.json")
print(f"Gráficos guardados en {OUT_DIR}/fig1..fig3_*.png")
