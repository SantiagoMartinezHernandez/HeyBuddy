import os
from google import genai

GEMINI_MODEL = "models/gemini-2.0-flash"


def get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")

    return genai.Client(api_key=api_key)


def build_coaching_prompt(metrics: dict) -> str:
    """
    Build a short coaching prompt from aggregated movement metrics.
    """
    return f"""
The user just completed a {metrics["exercise"]} set.

Metrics:
- Reps: {metrics["reps"]}
- Average depth: {metrics["avg_depth"]}
- Knee alignment: {metrics["knee_alignment"]}
- Tempo: {metrics["tempo"]}
- Symmetry score: {metrics["symmetry"]}

Task:
1. Give ONE short correction cue (if needed)
2. Give ONE motivating sentence
3. Keep the answer under 3 sentences.
""".strip()


def generate_coaching_feedback(metrics: dict) -> str:
    """
    Send the coaching prompt to Gemini and return the response.
    """
    client = get_gemini_client()

    prompt = build_coaching_prompt(metrics)

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )

    return response.text