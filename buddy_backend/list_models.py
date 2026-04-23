import os
from dotenv import load_dotenv
from google import genai

"""
List all Gemini models available for *this* Google Cloud project & API key.
Run with:
  python list_models.py
"""

def main():
    load_dotenv()  # ✅ lädt buddy_backend/.env

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set in environment (.env)")

    client = genai.Client(api_key=api_key)

    print("Available models:\n")
    for model in client.models.list():
        print(model.name)

if __name__ == "__main__":
    main()