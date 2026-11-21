"""
Modelo: predecir slow_resolution (ticket por encima del percentil 75 de duración)
usando SOLO variables conocidas al abrir el ticket (sin fuga de información).

Split cronológico (no aleatorio): entrena con 2010-2011, valida con 2012.
Esto respeta la sección 8 del documento ("en series temporales, respetar el
orden cronológico").

Incluye: baseline trivial + baseline por regla simple + modelo real,
comparados con las mismas métricas.
"""
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve,
)

sns.set_theme(style="whitegrid", font_scale=1.05)
PALETTE = {"ok": "#2E86AB", "slow": "#E94F37", "neutral": "#6C757D"}
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

IN_PATH = sys.argv[1] if len(sys.argv) > 1 else "outputs/case_table.csv"
OUT_DIR = sys.argv[2] if len(sys.argv) > 2 else "outputs"

df = pd.read_csv(IN_PATH, parse_dates=["open_timestamp", "close_timestamp"])

# --- punto de predicción: "tras el 2do evento" (ticket ya triado), no la
# apertura cruda. Se excluyen los casos donde el 2do evento coincide con el
# cierre (n_events==2): ahí el gap_to_second_event_h seria identico a la
# duracion total -> fuga de informacion para ese subconjunto (~4.7% de los
# casos). Esta exclusion se documenta como limitacion de alcance.
n_before = len(df)
df = df[(~df["second_event_is_closing_event"]) & (df["n_events"] >= 2)].copy()
print(f"Casos excluidos por fuga potencial (2do evento = cierre): "
      f"{n_before - len(df)} de {n_before} ({(n_before-len(df))/n_before*100:.1f}%)")

FEATURES = ["open_hour", "open_weekday", "open_month", "first_activity",
            "second_activity", "gap_to_second_event_h"]
CATEGORICAL = ["open_weekday", "open_month", "first_activity", "second_activity"]
TARGET = "slow_resolution"

# --- split cronológico: train = 2010-2011, test = 2012 ---
df["open_year"] = df["open_timestamp"].dt.year
train_df = df[df["open_year"] < 2012].copy()
test_df = df[df["open_year"] == 2012].copy()

# --- target definido SOLO con train (evita fuga del futuro hacia la etiqueta) ---
TRAIN_P75_HOURS = train_df["duration_hours"].quantile(0.75)
train_df[TARGET] = (train_df["duration_hours"] > TRAIN_P75_HOURS).astype(int)
test_df[TARGET] = (test_df["duration_hours"] > TRAIN_P75_HOURS).astype(int)
print(f"Umbral 'resolución lenta' fijado con train (2010-2011): {TRAIN_P75_HOURS:.1f} horas")
print(f"Train: {len(train_df)} casos (2010-2011) | Test: {len(test_df)} casos (2012)")
print(f"Tasa base de 'slow_resolution' en train: {train_df[TARGET].mean():.3f}")
print(f"Tasa base de 'slow_resolution' en test:  {test_df[TARGET].mean():.3f}")

X_train, y_train = train_df[FEATURES], train_df[TARGET]
X_test, y_test = test_df[FEATURES], test_df[TARGET]

def eval_model(name, y_true, y_pred, y_prob=None):
    row = {
        "modelo": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_prob is not None:
        row["roc_auc"] = roc_auc_score(y_true, y_prob)
    return row

results = []

# --- Baseline 1: siempre predice la clase mayoritaria (no-lenta) ---
pred_majority = np.zeros(len(y_test), dtype=int)
results.append(eval_model("Baseline: clase mayoritaria", y_test, pred_majority))

# --- Baseline 2: regla simple de negocio (viernes = día de mayor riesgo del EDA) ---
pred_rule = (X_test["open_weekday"] == 4).astype(int)  # 4 = viernes
results.append(eval_model("Baseline: regla 'si es viernes'", y_test, pred_rule))

# --- Modelo real 1: Regresión logística ---
preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
], remainder="passthrough")

logreg = Pipeline([
    ("prep", preprocess),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
])
logreg.fit(X_train, y_train)
prob_lr = logreg.predict_proba(X_test)[:, 1]
pred_lr = (prob_lr >= 0.5).astype(int)
results.append(eval_model("Regresión logística", y_test, pred_lr, prob_lr))

# --- Modelo real 2: Random Forest (referencia no lineal) ---
rf = Pipeline([
    ("prep", preprocess),
    ("clf", RandomForestClassifier(
        n_estimators=300, max_depth=5, min_samples_leaf=20,
        class_weight="balanced", random_state=42
    )),
])
rf.fit(X_train, y_train)
prob_rf = rf.predict_proba(X_test)[:, 1]
pred_rf = (prob_rf >= 0.5).astype(int)
results.append(eval_model("Random Forest", y_test, pred_rf, prob_rf))

results_df = pd.DataFrame(results).round(3)
print("\n=== Comparación de modelos (test = 2012, fuera de muestra temporal) ===")
print(results_df.to_string(index=False))

cm = confusion_matrix(y_test, pred_rf)
print("\nMatriz de confusión (Random Forest):")
print(pd.DataFrame(cm, index=["real: rápido", "real: lento"],
                    columns=["pred: rápido", "pred: lento"]))

# importancia de variables (Random Forest)
numeric_passthrough = [f for f in FEATURES if f not in CATEGORICAL]
feat_names = (
    list(rf.named_steps["prep"].named_transformers_["cat"].get_feature_names_out(CATEGORICAL))
    + numeric_passthrough
)
importances = rf.named_steps["clf"].feature_importances_
imp_df = pd.DataFrame({"feature": feat_names, "importance": importances})
imp_df = imp_df.sort_values("importance", ascending=False).head(10)
print("\nTop variables más importantes (Random Forest):")
print(imp_df.to_string(index=False))

# --- gráfico 1: curva ROC (regresión logística vs random forest) ---
fig, ax = plt.subplots(figsize=(6.5, 5.2))
for name, prob, color in [("Regresión logística", prob_lr, PALETTE["ok"]),
                           ("Random Forest", prob_rf, PALETTE["slow"])]:
    fpr, tpr, _ = roc_curve(y_test, prob)
    auc_val = roc_auc_score(y_test, prob)
    ax.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.2f})", color=color, lw=2)
ax.plot([0, 1], [0, 1], "--", color=PALETTE["neutral"], lw=1, label="Azar (AUC=0.50)")
ax.set_xlabel("Falsos positivos")
ax.set_ylabel("Verdaderos positivos")
ax.set_title("Curva ROC · predicción tras el 2º evento\n(train 2010-2011 → test 2012, fuera de muestra)")
ax.legend(loc="lower right", fontsize=9)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/fig4_roc_curve.png", dpi=150)
plt.close(fig)

# --- gráfico 2: importancia de variables ---
fig, ax = plt.subplots(figsize=(6.5, 4.5))
top10 = imp_df.head(8).iloc[::-1]
ax.barh(top10["feature"], top10["importance"], color=PALETTE["ok"])
ax.set_xlabel("Importancia (Random Forest)")
ax.set_title("¿Qué pesa más para anticipar un ticket lento?")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/fig5_feature_importance.png", dpi=150)
plt.close(fig)

results_df.to_csv(f"{OUT_DIR}/model_comparison.csv", index=False)
imp_df.to_csv(f"{OUT_DIR}/feature_importance.csv", index=False)
with open(f"{OUT_DIR}/model_summary.json", "w") as f:
    json.dump({
        "train_size": len(train_df),
        "test_size": len(test_df),
        "base_rate_train": float(train_df[TARGET].mean()),
        "base_rate_test": float(test_df[TARGET].mean()),
        "best_model": "Regresión logística" if roc_auc_score(y_test, prob_lr) >= roc_auc_score(y_test, prob_rf) else "Random Forest",
        "best_roc_auc": float(max(roc_auc_score(y_test, prob_lr), roc_auc_score(y_test, prob_rf))),
        "logreg_roc_auc": float(roc_auc_score(y_test, prob_lr)),
        "rf_roc_auc": float(roc_auc_score(y_test, prob_rf)),
        "top_features": imp_df.to_dict(orient="records"),
    }, f, indent=2, ensure_ascii=False)

print(f"\nGuardado: {OUT_DIR}/model_comparison.csv, feature_importance.csv, model_summary.json")
