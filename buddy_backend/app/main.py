cat > buddy_backend/app/main.py <<'EOF'
from fastapi import FastAPI
from dotenv import load_dotenv

from app.ai.gemini_client import get_gemini_model

load_dotenv()

app = FastAPI(title="HeyBuddy Gemini Backend")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/ai/hello")
def ai_hello():
    model = get_gemini_model()
    response = model.generate_content(
        "Give a short, motivating fitness coaching greeting."
    )
    return {"text": response.text.strip()}
EOF