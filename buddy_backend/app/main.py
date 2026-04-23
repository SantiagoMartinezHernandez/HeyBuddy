from fastapi import FastAPI
from dotenv import load_dotenv
from app.ai.gemini_client import generate_coaching_feedback

load_dotenv()
app = FastAPI(title="HeyBuddy Gemini Backend")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ai/feedback")
def ai_feedback(metrics: dict):
    """
    Receive aggregated movement metrics and return coaching feedback.
    """
    text = generate_coaching_feedback(metrics)
    return {"feedback": text}