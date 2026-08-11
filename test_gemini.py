"""
Standalone test script to verify structured JSON output from the Gemini API.
Run with:  python test_gemini.py

Requires the GEMINI_API_KEY environment variable to be set.
"""

import json
import os
import sys
from typing import List

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# 1. Read the API key from the environment
# ---------------------------------------------------------------------------
api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    print("ERROR: GEMINI_API_KEY environment variable is not set.")
    print()
    print("To fix this, set the variable before running the script:")
    print()
    print("  Windows (PowerShell):")
    print('    $env:GEMINI_API_KEY = "your_key_here"')
    print()
    print("  Windows (CMD):")
    print('    set GEMINI_API_KEY=your_key_here')
    print()
    print("  Linux / macOS:")
    print('    export GEMINI_API_KEY="your_key_here"')
    print()
    print("You can get an API key from: https://aistudio.google.com/apikey")
    sys.exit(1)

# ---------------------------------------------------------------------------
# 2. Import the SDK and configure the client
# ---------------------------------------------------------------------------
try:
    from google import genai
    from google.genai import types
except ImportError:
    print("ERROR: google-genai package is not installed.")
    print("Install it with:  pip install google-genai")
    sys.exit(1)

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# 3. Define the structured output schema using Pydantic
# ---------------------------------------------------------------------------
# This Pydantic model is passed directly to the SDK's response_schema
# parameter. The SDK converts it into a JSON Schema that Gemini enforces,
# guaranteeing the response matches this exact structure.


class DiseaseInfo(BaseModel):
    """Structured information about a crop disease."""

    disease_name: str = Field(description="Full name of the disease")
    description: str = Field(description="1-2 sentence description of the disease")
    symptoms: List[str] = Field(description="List of observable symptoms")
    treatment: List[str] = Field(description="List of treatment methods")
    prevention: List[str] = Field(description="List of prevention strategies")
    severity: str = Field(description="Severity level: Low, Medium, or High")
    farmer_advice: str = Field(
        description="1-2 sentences of practical, simple advice for farmers"
    )
    affected_crops: List[str] = Field(
        description="List of crop species commonly affected by this disease"
    )
    recovery_timeline: str = Field(
        description="Estimated time for treatment to show results, e.g. 7-14 days with consistent treatment"
    )


# ---------------------------------------------------------------------------
# 4. Configure the generation request for structured JSON output
# ---------------------------------------------------------------------------
client = genai.Client(api_key=api_key)

GENERATION_CONFIG = types.GenerateContentConfig(
    response_mime_type="application/json",  # Forces JSON output
    response_schema=DiseaseInfo,            # Enforces our exact schema
)

MODEL = "gemini-3.5-flash"

# Test with TWO diseases to confirm structured output is consistent
TEST_DISEASES = [
    "Tomato Early Blight",
    "Tomato Late Blight",
]


def pretty_print(data: dict) -> None:
    """Print each field of the disease info in a readable format."""
    print(f"  Disease Name:   {data.get('disease_name', 'N/A')}")
    print(f"  Description:    {data.get('description', 'N/A')}")

    print("  Symptoms:")
    for i, symptom in enumerate(data.get("symptoms", []), 1):
        print(f"    {i}. {symptom}")

    print("  Treatment:")
    for i, treatment in enumerate(data.get("treatment", []), 1):
        print(f"    {i}. {treatment}")

    print("  Prevention:")
    for i, prevention in enumerate(data.get("prevention", []), 1):
        print(f"    {i}. {prevention}")

    print(f"  Severity:       {data.get('severity', 'N/A')}")

    print("  Affected Crops:")
    for i, crop in enumerate(data.get("affected_crops", []), 1):
        print(f"    {i}. {crop}")

    print(f"  Recovery:       {data.get('recovery_timeline', 'N/A')}")
    print(f"  Farmer Advice:  {data.get('farmer_advice', 'N/A')}")


# ---------------------------------------------------------------------------
# 5. Send test prompts and validate structured responses
# ---------------------------------------------------------------------------
REQUIRED_FIELDS = [
    "disease_name", "description", "symptoms",
    "treatment", "prevention", "severity", "farmer_advice",
    "affected_crops", "recovery_timeline",
]

all_passed = True

for disease in TEST_DISEASES:
    prompt = f"Give information about {disease}"
    print(f'\nSending prompt: "{prompt}"')
    print("=" * 60)

    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=GENERATION_CONFIG,
        )

        raw_text = response.text

        # Try to parse the JSON response
        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError as je:
            print(f"ERROR: Response is not valid JSON — {je}")
            print(f"Raw response:\n{raw_text}")
            all_passed = False
            continue

        # Validate all 7 required fields are present
        missing = [f for f in REQUIRED_FIELDS if f not in data]
        if missing:
            print(f"WARNING: Missing fields: {missing}")
            all_passed = False

        pretty_print(data)
        print("=" * 60)
        print("OK — Valid JSON with all fields.")

    except Exception as e:
        print(f"ERROR: API call failed — {type(e).__name__}: {e}")
        print()
        print("Possible causes:")
        print("  - Invalid or expired API key")
        print("  - No internet connection")
        print("  - API rate limit exceeded")
        print("  - Google API service outage")
        all_passed = False

# ---------------------------------------------------------------------------
# 6. Final summary
# ---------------------------------------------------------------------------
print()
print("=" * 60)
if all_passed:
    print("SUCCESS: All test diseases returned clean, parseable JSON")
    print(f"         with all {len(REQUIRED_FIELDS)} fields populated.")
else:
    print("FAILURE: One or more tests had issues. See errors above.")
    sys.exit(1)
