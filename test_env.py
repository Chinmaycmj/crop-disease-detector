import os
import streamlit as st
from dotenv import load_dotenv
from google import genai

load_dotenv()

st.title("🧪 Gemini API Test")

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("❌ GEMINI_API_KEY not found in .env")
else:
    st.success("✅ .env loaded successfully!")
    
    try:
        client = genai.Client(api_key=api_key)
        
        # Explicitly pass gemini-2.0-flash
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents='Say "API Connection Successful!" in 5 words or less.',
        )
        st.success("🎉 API Response Received:")
        st.write(response.text)
        
    except Exception as e:
        st.error(f"❌ Error: {e}")