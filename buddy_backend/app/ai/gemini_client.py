cat > buddy_backend/app/ai/gemini_client.py <<'EOF'
import os
import google.generativeai as genai

def get_gemini_model():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")

    genai.configure(api_key=api_key)

    return genai.GenerativeModel(
        model_name="models/gemini-1.5-pro"
    )
EOF