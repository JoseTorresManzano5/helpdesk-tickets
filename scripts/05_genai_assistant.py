"""
Asistente de IA generativa (sección 4.4): responde preguntas empresariales
usando SOLO los resultados verificados en verified_results.json (producidos
por código ejecutado, no por el LLM). Incluye un control automático que
compara los números citados por el LLM contra los números reales, y marca
cualquier cifra que no coincida ("afirmación no verificable").

Requiere:
  pip install groq
  export GROQ_API_KEY="tu_api_key_gratuita_de https://console.groq.com"

Uso:
  python 05_genai_assistant.py outputs/verified_results.json outputs/genai_log.json
"""
import json
import re
import sys
import os
from datetime import datetime, timezone

RESULTS_PATH = sys.argv[1] if len(sys.argv) > 1 else "outputs/verified_results.json"
LOG_PATH = sys.argv[2] if len(sys.argv) > 2 else "outputs/genai_log.json"
MODEL = "openai/gpt-oss-120b"  # modelo gratuito disponible en Groq

with open(RESULTS_PATH, encoding="utf-8") as f:
    verified = json.load(f)

SYSTEM_PROMPT = f"""Eres un asistente de análisis de datos para un proyecto académico
de gestión de incidentes de mesa de ayuda (dataset Helpdesk, Mendeley Data).

REGLA ABSOLUTA: solo puedes afirmar números que estén EXACTAMENTE en este
bloque de resultados verificados (calculados con código, no por ti). Si te
preguntan algo que no está aquí, di explícitamente "ese dato no fue calculado
en este proyecto" — nunca estimes, redondees de forma distinta ni inventes.

RESULTADOS VERIFICADOS:
{json.dumps(verified, indent=2, ensure_ascii=False)}

Responde en español, de forma breve y orientada a decisiones de negocio.
Cuando cites una cifra, repítela tal cual aparece arriba.
"""

BUSINESS_QUESTIONS = [
    "¿Vale la pena construir un modelo para anticipar tickets lentos, o el "
    "proceso actual (sin modelo) ya es suficiente?",
    "¿Cuál es la señal más importante para saber si un ticket se va a demorar, "
    "y en qué momento del proceso está disponible?",
    "¿El día de la semana en que se abre el ticket es un buen predictor por sí "
    "solo? Justifica con los números.",
]


def call_groq(messages):
    """Llama a la API de Groq (compatible con OpenAI). Requiere internet y
    GROQ_API_KEY configurada en el entorno donde se ejecute este script."""
    from groq import Groq  # pip install groq
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    resp = client.chat.completions.create(
        model=MODEL, messages=messages, temperature=0.1, max_tokens=400,
    )
    return resp.choices[0].message.content


def extract_numbers(text):
    return set(re.findall(r"\d+(?:[.,]\d+)?", text))


def flatten_verified_numbers(d, acc=None):
    """Junta todos los números que SÍ son válidos (los del JSON verificado)."""
    if acc is None:
        acc = set()
    for v in d.values():
        if isinstance(v, dict):
            flatten_verified_numbers(v, acc)
        elif isinstance(v, (int, float)):
            acc.add(str(v))
            acc.add(str(round(v)))
    return acc


def check_claims(answer_text, verified_dict):
    """Control de afirmaciones incorrectas (sección 4.4): marca cualquier
    número citado por el LLM que no aparezca en los resultados verificados."""
    valid_numbers = flatten_verified_numbers(verified_dict)
    cited = extract_numbers(answer_text)
    unverifiable = [n for n in cited if n not in valid_numbers and len(n.replace(",", "").replace(".", "")) > 1]
    return {
        "numeros_citados": sorted(cited),
        "numeros_no_verificables": sorted(unverifiable),
        "todas_las_cifras_verificadas": len(unverifiable) == 0,
    }


def main():
    log = {
        "modelo_llm": MODEL,
        "generado_en_utc": datetime.now(timezone.utc).isoformat(),
        "fuente_de_verdad": RESULTS_PATH,
        "interacciones": [],
    }

    if "GROQ_API_KEY" not in os.environ:
        print("AVISO: no hay GROQ_API_KEY configurada en este entorno "
              "(y este sandbox no tiene acceso a internet para llamar a la "
              "API de Groq). Este script está listo para ejecutarse en Google "
              "Colab o tu máquina: solo exporta tu API key gratuita y corre "
              "de nuevo. No se simulará ninguna respuesta falsa.")
        return

    for q in BUSINESS_QUESTIONS:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q},
        ]
        answer = call_groq(messages)
        control = check_claims(answer, verified)
        log["interacciones"].append({
            "pregunta": q, "respuesta": answer, "control_de_cifras": control,
        })
        print(f"\nQ: {q}\nA: {answer}\nControl: {control}\n{'-'*60}")

    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    print(f"\nEvidencia de funcionamiento guardada en: {LOG_PATH}")


if __name__ == "__main__":
    main()
