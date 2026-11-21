"""
Consolida TODOS los resultados ya calculados (EDA + modelo) en un único
JSON "verified_results.json". Este archivo es la ÚNICA fuente de verdad
que se le entrega al LLM: así el asistente de IA generativa no puede
inventar cifras, solo citar las que ya fueron calculadas con código.
"""
import json
import sys

OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "outputs"

with open(f"{OUT_DIR}/eda_summary.json") as f:
    eda = json.load(f)
with open(f"{OUT_DIR}/model_summary.json") as f:
    model = json.load(f)

verified = {
    "dataset": {
        "nombre": "Helpdesk (Verenich, 2016, Mendeley Data, DOI 10.17632/39bp3vv62t.1)",
        "licencia": "MIT",
        "n_casos": 3804,
        "n_eventos": 13710,
        "periodo": "2010-01-13 a 2012-11-06",
    },
    "eda": {
        "duracion_mediana_horas": round(eda["duration_median_h"], 1),
        "duracion_p75_horas": round(eda["duration_p75_h"], 1),
        "tasa_reproceso": round(eda["rework_rate"] * 100, 1),
        "factor_reproceso_sobre_duracion": round(eda["rework_duration_lift_x"], 1),
        "dia_mejor_tasa_lenta": eda["slowrate_best_day"],
        "dia_peor_tasa_lenta": eda["slowrate_worst_day"],
        "spread_dia_semana_pp": round(eda["slowrate_weekday_spread_pp"], 1),
        "casos_fin_de_semana_excluidos_del_analisis_diario": eda["n_weekend_cases_excluded"],
    },
    "modelo": {
        "pregunta": "¿Se resolverá este ticket más lento que el 75% de los casos "
                    "del periodo de entrenamiento (umbral fijo en horas), prediciendo "
                    "justo después de registrado el 2º evento del caso?",
        "punto_de_prediccion": "tras el 2º evento del ticket (no en la apertura)",
        "entrenamiento": "2010-2011 (chronological split)",
        "prueba_fuera_de_muestra": "2012",
        "n_entrenamiento": model["train_size"],
        "n_prueba": model["test_size"],
        "tasa_base_prueba": round(model["base_rate_test"] * 100, 1),
        "mejor_modelo": model["best_model"],
        "auc_regresion_logistica": round(model["logreg_roc_auc"], 3),
        "auc_random_forest": round(model["rf_roc_auc"], 3),
        "variable_mas_importante": model["top_features"][0]["feature"],
        "importancia_variable_principal_pct": round(model["top_features"][0]["importance"] * 100, 1),
        "limitacion_cobertura": "excluye 4.7% de tickets que cierran en 2 eventos "
                                 "(el 2do evento coincidiría con el cierre -> fuga de información)",
    },
}

with open(f"{OUT_DIR}/verified_results.json", "w") as f:
    json.dump(verified, f, indent=2, ensure_ascii=False)

print(json.dumps(verified, indent=2, ensure_ascii=False))
print(f"\nGuardado: {OUT_DIR}/verified_results.json")
