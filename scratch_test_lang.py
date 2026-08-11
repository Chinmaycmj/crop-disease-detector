import os
import json
from dotenv import load_dotenv

load_dotenv()

from app import get_ai_treatment

def test_language(lang_code):
    print(f"\\n--- Testing with lang_code='{lang_code}' ---")
    result = get_ai_treatment("Tomato Early Blight", lang_code=lang_code)
    with open(f"test_out_{lang_code}.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Saved to test_out_{lang_code}.json")

if __name__ == "__main__":
    test_language("kn")
    test_language("hi")
