import os
import json
import textwrap
import base64
from io import BytesIO
from datetime import datetime
from typing import List

import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, mm
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
    Table, TableStyle, ListFlowable, ListItem, PageBreak,
)
from reportlab.lib import colors

# ---------------------------------------------------------------------------
# Gemini API Setup
# ---------------------------------------------------------------------------
load_dotenv()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Initialise the Gemini client once (None if no key is configured)
_gemini_client = None
if GEMINI_API_KEY:
    try:
        from google import genai
        from google.genai import types
        _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception:
        _gemini_client = None


# ---------------------------------------------------------------------------
# Pydantic schema — enforces the exact JSON shape Gemini must return
# ---------------------------------------------------------------------------
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


def get_ai_treatment(disease_name, lang_code="en", model_name="gemini-3.5-flash"):
    """
    Call Gemini for structured disease info.

    Returns a dict with keys: disease_name, description, symptoms,
    treatment, prevention, severity, farmer_advice.
    Returns None if the call fails for any reason.
    """
    if _gemini_client is None:
        return None

    lang_names = {"en": "English", "kn": "Kannada", "hi": "Hindi"}
    target_lang = lang_names.get(lang_code, "English")

    try:
        from google.genai import types as _t  # already imported at module level
        
        prompt = (
            f"Give information about {disease_name}. "
            f"Respond entirely in {target_lang} language, including all field values in the JSON."
        )
        
        response = _gemini_client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=_t.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=DiseaseInfo,
            ),
        )
        data = json.loads(response.text)
        # Ensure all required keys exist
        for key in ("description", "symptoms", "treatment", "prevention"):
            if key not in data:
                return None
        return data
    except Exception as e:
        print(f"Gemini call failed: {e}")
        return None

# ── Multilingual UI Labels ──────────────────────────────────────────
UI_LABELS = {
    "en": {
        "page_title": "Crop Disease Detector",
        "hero_title": "🌱 Crop Disease Detection &amp; Care Assistant",
        "hero_subtitle": "Upload a leaf image to receive instant disease diagnosis powered by MobileNetV2 deep learning",
        "how_it_works_title": "#### 🔬 How It Works",
        "how_it_works_1": "Upload a photo of an affected crop leaf",
        "how_it_works_2": "Our MobileNetV2 model analyzes the image locally",
        "how_it_works_3": "Receive an instant diagnosis &amp; treatment plan",
        "upload_title": "#### 📤 Upload Leaf Image",
        "upload_placeholder": "Choose a leaf image...",
        "powered_by": "🌿 Powered by MobileNetV2 · TensorFlow",
        "uploaded_image_caption": "📷 Uploaded Leaf Image",
        "model_not_found": "Model file not found at `{MODEL_PATH}`. Please ensure `plant_disease_model.h5` is placed inside the `model/` directory.",
        "analyzing_spinner": "Analyzing image using local model...",
        "ai_spinner": "Getting AI-powered treatment analysis...",
        "source_ai": "✨ AI-Powered Analysis",
        "source_offline": "📚 Offline Reference Data",
        "offline_note": " (offline data available in English only)",
        "pdf_english_note": "(Report will be generated in English)",
        "severity": "⚠️ Severity",
        "affected_crops": "🌾 Affected Crops",
        "recovery_timeline": "⏱️ Recovery Timeline",
        "farmer_advice": "🧑\u200d🌾 Farmer Advice",
        "diagnosis_result": "Diagnosis Result",
        "model_confidence": "Model Confidence",
        "treatment_plan": "📋 Recommended Treatment Plan",
        "about_condition": "🔍 About This Condition",
        "symptoms": "🩺 Symptoms",
        "treatment_steps": "💊 Treatment Steps",
        "prevention": "🛡️ Prevention",
        "download_pdf": "\U0001f4c4 Download Report (PDF)",
        "empty_title": "Ready to Diagnose Your Crop",
        "empty_text": "Upload a leaf image using the sidebar panel to get an instant AI-powered disease diagnosis and personalized treatment recommendations.",
    },
    "kn": {
        "page_title": "ಬೆಳೆ ರೋಗ ಪತ್ತೆಕಾರಕ",
        "hero_title": "🌱 ಬೆಳೆ ರೋಗ ಪತ್ತೆ ಮತ್ತು ಆರೈಕೆ ಸಹಾಯಕ",
        "hero_subtitle": "MobileNetV2 ಡೀಪ್ ಲರ್ನಿಂಗ್ ಆಧಾರಿತ ತ್ವರಿತ ರೋಗ ಪತ್ತೆಗಾಗಿ ಎಲೆಯ ಚಿತ್ರವನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ",
        "how_it_works_title": "#### 🔬 ಇದು ಹೇಗೆ ಕೆಲಸ ಮಾಡುತ್ತದೆ",
        "how_it_works_1": "ರೋಗಗ್ರಸ್ತ ಬೆಳೆಯ ಎಲೆಯ ಫೋಟೋವನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ",
        "how_it_works_2": "ನಮ್ಮ MobileNetV2 ಮಾಡೆಲ್ ಚಿತ್ರವನ್ನು ಸ್ಥಳೀಯವಾಗಿ ವಿಶ್ಲೇಷಿಸುತ್ತದೆ",
        "how_it_works_3": "ತ್ವರಿತ ರೋಗ ಪತ್ತೆ ಮತ್ತು ಚಿಕಿತ್ಸಾ ಯೋಜನೆಯನ್ನು ಪಡೆಯಿರಿ",
        "upload_title": "#### 📤 ಎಲೆಯ ಚಿತ್ರವನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ",
        "upload_placeholder": "ಎಲೆಯ ಚಿತ್ರವನ್ನು ಆಯ್ಕೆಮಾಡಿ...",
        "powered_by": "🌿 MobileNetV2 · TensorFlow ನಿಂದ ನಡೆಸಲ್ಪಡುತ್ತಿದೆ",
        "uploaded_image_caption": "📷 ಅಪ್‌ಲೋಡ್ ಮಾಡಿದ ಎಲೆಯ ಚಿತ್ರ",
        "model_not_found": "`{MODEL_PATH}` ನಲ್ಲಿ ಮಾಡೆಲ್ ಫೈಲ್ ಕಂಡುಬಂದಿಲ್ಲ. ದಯವಿಟ್ಟು `plant_disease_model.h5` ಅನ್ನು `model/` ಡೈರೆಕ್ಟರಿಯೊಳಗೆ ಇರಿಸಲಾಗಿದೆಯೇ ಎಂದು ಖಚಿತಪಡಿಸಿಕೊಳ್ಳಿ.",
        "analyzing_spinner": "ಸ್ಥಳೀಯ ಮಾಡೆಲ್ ಬಳಸಿ ಚಿತ್ರವನ್ನು ವಿಶ್ಲೇಷಿಸಲಾಗುತ್ತಿದೆ...",
        "ai_spinner": "AI-ಆಧಾರಿತ ಚಿಕಿತ್ಸಾ ವಿಶ್ಲೇಷಣೆಯನ್ನು ಪಡೆಯಲಾಗುತ್ತಿದೆ...",
        "source_ai": "✨ AI-ಆಧಾರಿತ ವಿಶ್ಲೇಷಣೆ",
        "source_offline": "📚 ಆಫ್‌ಲೈನ್ ಉಲ್ಲೇಖ ಡೇಟಾ",
        "offline_note": " (offline data available in English only)",
        "pdf_english_note": "(ವರದಿ ಇಂಗ್ಲಿಷ್ನಲ್ಲಿ ರಚಿಸಲಾಗುತ್ತದೆ)",
        "severity": "⚠️ ತೀವ್ರತೆ",
        "affected_crops": "🌾 ಬಾಧಿತ ಬೆಳೆಗಳು",
        "recovery_timeline": "⏱️ ಚೇತರಿಕೆ ಸಮಯ",
        "farmer_advice": "🧑\u200d🌾 ರೈತರಿಗೆ ಸಲಹೆ",
        "diagnosis_result": "ರೋಗ ಪತ್ತೆ ಫಲಿತಾಂಶ",
        "model_confidence": "ಮಾಡೆಲ್ ವಿಶ್ವಾಸಾರ್ಹತೆ",
        "treatment_plan": "📋 ಶಿಫಾರಸು ಮಾಡಿದ ಚಿಕಿತ್ಸಾ ಯೋಜನೆ",
        "about_condition": "🔍 ಈ ರೋಗದ ಬಗ್ಗೆ",
        "symptoms": "🩺 ಲಕ್ಷಣಗಳು",
        "treatment_steps": "💊 ಚಿಕಿತ್ಸಾ ಹಂತಗಳು",
        "prevention": "🛡️ ತಡೆಗಟ್ಟುವಿಕೆ",
        "download_pdf": "\U0001f4c4 ವರದಿ ಡೌನ್‌ಲೋಡ್ ಮಾಡಿ (PDF)",
        "empty_title": "ನಿಮ್ಮ ಬೆಳೆಯನ್ನು ಪರೀಕ್ಷಿಸಲು ಸಿದ್ಧವಾಗಿದೆ",
        "empty_text": "ತ್ವರಿತ AI-ಆಧಾರಿತ ರೋಗ ಪತ್ತೆ ಮತ್ತು ವೈಯಕ್ತಿಕಗೊಳಿಸಿದ ಚಿಕಿತ್ಸಾ ಶಿಫಾರಸುಗಳನ್ನು ಪಡೆಯಲು ಸೈಡ್‌ಬಾರ್ ಬಳಸಿ ಎಲೆಯ ಚಿತ್ರವನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ.",
    },
    "hi": {
        "page_title": "फसल रोग संसूचक",
        "hero_title": "🌱 फसल रोग पहचान और देखभाल सहायक",
        "hero_subtitle": "MobileNetV2 डीप लर्निंग द्वारा संचालित त्वरित रोग निदान प्राप्त करने के लिए पत्ती की छवि अपलोड करें",
        "how_it_works_title": "#### 🔬 यह कैसे काम करता है",
        "how_it_works_1": "प्रभावित फसल की पत्ती की फोटो अपलोड करें",
        "how_it_works_2": "हमारा MobileNetV2 मॉडल स्थानीय स्तर पर छवि का विश्लेषण करता है",
        "how_it_works_3": "त्वरित निदान और उपचार योजना प्राप्त करें",
        "upload_title": "#### 📤 पत्ती की छवि अपलोड करें",
        "upload_placeholder": "पत्ती की छवि चुनें...",
        "powered_by": "🌿 MobileNetV2 · TensorFlow द्वारा संचालित",
        "uploaded_image_caption": "📷 अपलोड की गई पत्ती की छवि",
        "model_not_found": "`{MODEL_PATH}` पर मॉडल फ़ाइल नहीं मिली। कृपया सुनिश्चित करें कि `plant_disease_model.h5` को `model/` डायरेक्टरी के अंदर रखा गया है।",
        "analyzing_spinner": "स्थानीय मॉडल का उपयोग करके छवि का विश्लेषण किया जा रहा है...",
        "ai_spinner": "AI-संचालित उपचार विश्लेषण प्राप्त किया जा रहा है...",
        "source_ai": "✨ AI-संचालित विश्लेषण",
        "source_offline": "📚 ऑफ़लाइन संदर्भ डेटा",
        "offline_note": " (offline data available in English only)",
        "pdf_english_note": "(रिपोर्ट अंग्रेज़ी में जनरेट होगी)",
        "severity": "⚠️ गंभीरता",
        "affected_crops": "🌾 प्रभावित फसलें",
        "recovery_timeline": "⏱️ रिकवरी का समय",
        "farmer_advice": "🧑\u200d🌾 किसान सलाह",
        "diagnosis_result": "निदान परिणाम",
        "model_confidence": "मॉडल आत्मविश्वास",
        "treatment_plan": "📋 अनुशंसित उपचार योजना",
        "about_condition": "🔍 इस स्थिति के बारे में",
        "symptoms": "🩺 लक्षण",
        "treatment_steps": "💊 उपचार के कदम",
        "prevention": "🛡️ रोकथाम",
        "download_pdf": "\U0001f4c4 रिपोर्ट डाउनलोड करें (PDF)",
        "empty_title": "अपनी फसल का निदान करने के लिए तैयार",
        "empty_text": "त्वरित AI-संचालित रोग निदान और व्यक्तिगत उपचार सिफारिशें प्राप्त करने के लिए साइडबार पैनल का उपयोग करके पत्ती की छवि अपलोड करें।",
    }
}

# Set page configuration
st.set_page_config(
    page_title="Crop Disease Detector",
    page_icon="🌱",
    layout="wide"
)


def load_css(file_path):
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")


# Load Local Trained Model
MODEL_PATH = "model/plant_disease_model.h5"

@st.cache_resource
def load_plant_model():
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH)
    return None

model = load_plant_model()

# Class Names matching PlantVillage dataset
CLASS_NAMES = [
    'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight',
    'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus',
    'Tomato_healthy'
]

# Local Care Guidelines (No External API Required)
TREATMENTS = {
    'Early_blight': {
        'description': 'A fungal disease caused by Alternaria solani that affects older leaves first, gradually moving upward through the plant canopy.',
        'symptoms': ['Dark brown concentric rings (target-shaped lesions) on lower leaves', 'Yellowing tissue surrounding the spots', 'Premature leaf drop and reduced fruit yield'],
        'treatment': ['Remove and destroy infected lower leaves immediately', 'Apply copper-based fungicide (e.g., Bordeaux mixture) every 7\u201310 days', 'Switch to chlorothalonil if copper alone is insufficient', 'Ensure adequate spacing for air circulation'],
        'prevention': ['Practice 2\u20133 year crop rotation with non-solanaceous crops', 'Use certified disease-free seed and resistant varieties', 'Mulch around plants to prevent soil splash onto foliage'],
    },
    'Late_blight': {
        'description': 'A devastating oomycete disease caused by Phytophthora infestans, capable of destroying an entire field within days under cool, wet conditions.',
        'symptoms': ['Water-soaked, dark green to brown lesions on leaves and stems', 'White fuzzy mold on leaf undersides in humid conditions', 'Firm, dark rot spreading rapidly across fruit and tubers'],
        'treatment': ['Apply systemic fungicide immediately (e.g., Mancozeb or Metalaxyl)', 'Remove and burn all visibly infected plant material', 'Avoid overhead irrigation\u2014use drip systems instead', 'Harvest remaining healthy tubers promptly if outbreak is severe'],
        'prevention': ['Plant resistant cultivars when available', 'Eliminate volunteer potato plants and cull piles', 'Monitor weather forecasts\u2014apply preventive fungicide before wet spells'],
    },
    'Bacterial_spot': {
        'description': 'Caused by Xanthomonas species, this bacterial infection thrives in warm, humid environments and spreads via rain splash and contaminated tools.',
        'symptoms': ['Small, dark, water-soaked spots on leaves that turn brown', 'Raised, scab-like lesions on fruit surfaces', 'Leaf edges may appear scorched; severe defoliation possible'],
        'treatment': ['Spray copper hydroxide mixed with mancozeb at first sign of symptoms', 'Apply bactericide every 5\u20137 days during wet weather', 'Avoid working in the field when foliage is wet', 'Remove severely infected plants to limit spread'],
        'prevention': ['Use pathogen-free certified seeds or transplants', 'Sanitize tools, stakes, and cages between uses', 'Rotate crops away from peppers and tomatoes for 2+ years'],
    },
    'Leaf_Mold': {
        'description': 'A fungal disease caused by Passalora fulva (syn. Cladosporium fulvum), primarily affecting tomatoes grown in greenhouses or high-tunnel environments.',
        'symptoms': ['Pale green to yellow spots on upper leaf surfaces', 'Olive-green to brown velvety mold on leaf undersides', 'Leaves curl, wither, and drop prematurely'],
        'treatment': ['Improve greenhouse ventilation immediately', 'Reduce relative humidity below 85% using fans or venting', 'Apply fungicides containing chlorothalonil or mancozeb', 'Prune lower leaves to increase air flow around the canopy'],
        'prevention': ['Space plants generously and avoid overcrowding', 'Use drip irrigation to keep foliage dry', 'Select leaf-mold-resistant tomato varieties (Cf gene cultivars)'],
    },
    'Septoria_leaf_spot': {
        'description': 'Caused by the fungus Septoria lycopersici, this disease targets tomato foliage and can cause severe defoliation, reducing fruit quality and yield.',
        'symptoms': ['Numerous small circular spots (1\u20133 mm) with dark borders and tan centers', 'Tiny black pycnidia (fruiting bodies) visible in spot centers', 'Yellowing and premature dropping of lower leaves'],
        'treatment': ['Remove and destroy affected leaves promptly', 'Apply organic copper or sulfur-based fungicide every 7\u201310 days', 'Use chlorothalonil for heavier infestations', 'Ensure plants are staked or caged to keep foliage off the ground'],
        'prevention': ['Avoid overhead watering\u2014use drip or soaker hoses', 'Mulch around the base to prevent soil-splash transmission', 'Rotate tomato planting locations annually'],
    },
    'Spider_mites': {
        'description': 'Two-spotted spider mites (Tetranychus urticae) are tiny arachnids that feed on leaf cell contents, thriving in hot, dry, and dusty conditions.',
        'symptoms': ['Fine stippling or speckling on upper leaf surfaces', 'Leaves turn bronze or yellow, then dry out and drop', 'Fine silk webbing visible on leaf undersides and between stems'],
        'treatment': ['Spray insecticidal soap or horticultural oil, covering leaf undersides thoroughly', 'Apply neem oil every 5\u20137 days until mites are controlled', 'Introduce predatory mites (Phytoseiulus persimilis) as biological control', 'Increase humidity around plants by misting or overhead irrigation'],
        'prevention': ['Keep the growing area free of dust and weeds', 'Avoid excessive nitrogen fertilization which promotes lush growth mites prefer', 'Inspect new transplants carefully before introducing them to the field'],
    },
    'YellowLeaf__Curl_Virus': {
        'description': 'Tomato Yellow Leaf Curl Virus (TYLCV) is a begomovirus transmitted by the whitefly Bemisia tabaci. Once infected, plants cannot be cured.',
        'symptoms': ['Severe upward curling and cupping of young leaves', 'Pronounced yellowing of leaf margins and interveinal areas', 'Stunted growth with shortened internodes; flowers may drop'],
        'treatment': ['Remove and destroy all infected plants immediately', 'Control whitefly populations with yellow sticky traps', 'Apply systemic insecticides (e.g., imidacloprid) to remaining healthy plants', 'Use reflective silver mulch to repel whiteflies'],
        'prevention': ['Plant TYLCV-resistant tomato varieties', 'Install fine-mesh insect netting over seedbeds and transplants', 'Maintain a host-free period between growing seasons to break the virus cycle'],
    },
    'mosaic_virus': {
        'description': 'Tomato Mosaic Virus (ToMV) is a highly stable tobamovirus spread mechanically through contaminated hands, tools, and seed. No chemical cure exists.',
        'symptoms': ['Mottled light and dark green mosaic pattern on leaves', 'Leaf distortion, curling, and fern-like narrowing', 'Uneven ripening and internal browning of fruit'],
        'treatment': ['Rogue out and destroy all infected plants\u2014do not compost', 'Disinfect all tools and hands with 10% bleach or milk solution', 'Avoid handling healthy plants after touching infected ones', 'There is no curative treatment once a plant is infected'],
        'prevention': ['Use certified virus-free seed or ToMV-resistant varieties', 'Wash hands with soap and water before entering the growing area', 'Do not smoke near plants\u2014tobacco products can harbor related viruses'],
    },
    'healthy': {
        'description': 'No disease detected. The leaf appears healthy with normal coloration, texture, and structure.',
        'symptoms': ['No visible lesions, spots, or discoloration', 'Normal leaf shape and turgor', 'No signs of pest damage or wilting'],
        'treatment': ['No treatment required at this time', 'Continue regular watering and balanced fertilization', 'Maintain periodic visual inspection of the crop'],
        'prevention': ['Rotate crops each season to break pest and disease cycles', 'Use disease-resistant varieties suited to your region', 'Maintain proper plant spacing for good air circulation'],
    },
}

def get_treatment(disease_name):
    """Return a structured treatment dict for the matched disease key."""
    for key, info in TREATMENTS.items():
        if key in disease_name:
            return info
    return {
        'description': 'The specific condition could not be matched to a known disease profile.',
        'symptoms': ['Consult the uploaded image for visible signs'],
        'treatment': ['Ensure balanced irrigation and proper soil nutrition', 'Consult a local agricultural extension specialist for in-person diagnosis'],
        'prevention': ['Practice crop rotation and field sanitation', 'Use disease-resistant cultivars'],
    }


# ── PDF Report Generator ────────────────────────────────────────────
def generate_pdf_report(disease_name, confidence, treatment_advice, uploaded_image):
    """Generate a professional PDF report and return it as bytes."""
    T = UI_LABELS["en"]
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        topMargin=30 * mm,
        bottomMargin=25 * mm,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        title="Crop Disease Detection Report",
    )

    # ── Styles ──
    styles = getSampleStyleSheet()
    green_dark = HexColor("#1B5E20")
    green_mid = HexColor("#2E7D32")
    grey_text = HexColor("#444444")

    title_style = ParagraphStyle(
        "ReportTitle", parent=styles["Title"],
        fontSize=22, leading=28, textColor=green_dark,
        spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle", parent=styles["Normal"],
        fontSize=10, leading=14, textColor=HexColor("#888888"),
        spaceAfter=20, alignment=TA_CENTER,
    )
    heading_style = ParagraphStyle(
        "SectionHead", parent=styles["Heading2"],
        fontSize=14, leading=18, textColor=green_mid,
        spaceBefore=18, spaceAfter=8,
        borderWidth=0, borderPadding=0,
    )
    subheading_style = ParagraphStyle(
        "SubHead", parent=styles["Heading3"],
        fontSize=11, leading=15, textColor=green_dark,
        spaceBefore=12, spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "BodyCustom", parent=styles["Normal"],
        fontSize=10, leading=15, textColor=grey_text,
    )
    list_item_style = ParagraphStyle(
        "ListBody", parent=styles["Normal"],
        fontSize=10, leading=15, textColor=grey_text,
        leftIndent=0,
    )

    # ── Build story ──
    story = []
    page_w = A4[0] - 40 * mm  # usable width

    # Title block
    story.append(Paragraph("Crop Disease Detection Report", title_style))
    report_time = datetime.now().strftime("%B %d, %Y  %I:%M %p")
    story.append(Paragraph(f"Report Generated On: {report_time}", subtitle_style))

    # Divider
    divider_data = [[""]]
    divider = Table(divider_data, colWidths=[page_w])
    divider.setStyle(TableStyle([
        ("LINEBELOW", (0, 0), (-1, -1), 1, green_mid),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(divider)

    # ── Prediction Result ──
    story.append(Paragraph(T["diagnosis_result"], heading_style))

    conf_color = HexColor("#2E7D32") if confidence > 80 else (
        HexColor("#E65100") if confidence > 50 else HexColor("#C62828")
    )
    result_data = [
        [Paragraph(f'<b>{T["diagnosis_result"]}</b>', body_style),
         Paragraph(disease_name, body_style)],
        [Paragraph(f'<b>{T["model_confidence"]}</b>', body_style),
         Paragraph(f'<font color="{conf_color.hexval()}">{confidence:.1f}%</font>', body_style)],
    ]
    result_table = Table(result_data, colWidths=[page_w * 0.35, page_w * 0.65])
    result_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 0), (-1, -1), 0.5, HexColor("#E0E0E0")),
    ]))
    story.append(result_table)
    story.append(Spacer(1, 6))

    # ── Medical Information ──
    story.append(Paragraph(T["treatment_plan"], heading_style))

    # About
    story.append(Paragraph(T["about_condition"], subheading_style))
    story.append(Paragraph(treatment_advice["description"], body_style))

    # Symptoms (bullet list)
    story.append(Paragraph(T["symptoms"], subheading_style))
    symptom_items = [
        ListItem(Paragraph(s, list_item_style)) for s in treatment_advice["symptoms"]
    ]
    story.append(ListFlowable(symptom_items, bulletType="bullet", start="",
                              bulletFontSize=8, leftIndent=18, spaceBefore=2, spaceAfter=2))

    # Treatment Steps (numbered list)
    story.append(Paragraph(T["treatment_steps"], subheading_style))
    treatment_items = [
        ListItem(Paragraph(t, list_item_style)) for t in treatment_advice["treatment"]
    ]
    story.append(ListFlowable(treatment_items, bulletType="1",
                              bulletFontSize=10, leftIndent=18, spaceBefore=2, spaceAfter=2))

    # Prevention Tips (bullet list)
    story.append(Paragraph(T["prevention"], subheading_style))
    prevention_items = [
        ListItem(Paragraph(p, list_item_style)) for p in treatment_advice["prevention"]
    ]
    story.append(ListFlowable(prevention_items, bulletType="bullet", start="",
                              bulletFontSize=8, leftIndent=18, spaceBefore=2, spaceAfter=2))

    # Severity (only present when AI-powered)
    severity = treatment_advice.get("severity")
    if severity:
        sev_color_map = {"Low": "#2E7D32", "Medium": "#E65100", "High": "#C62828"}
        sev_hex = sev_color_map.get(severity, "#444444")
        story.append(Paragraph(T["severity"], subheading_style))
        story.append(Paragraph(
            f'<font color="{sev_hex}"><b>{severity}</b></font>', body_style
        ))

    # Affected Crops
    affected_crops = treatment_advice.get("affected_crops")
    if affected_crops:
        story.append(Paragraph(T["affected_crops"], subheading_style))
        crop_items = [
            ListItem(Paragraph(c, list_item_style)) for c in affected_crops
        ]
        story.append(ListFlowable(crop_items, bulletType="bullet", start="",
                                  bulletFontSize=8, leftIndent=18, spaceBefore=2, spaceAfter=2))

    # Recovery Timeline
    recovery_timeline = treatment_advice.get("recovery_timeline")
    if recovery_timeline:
        story.append(Paragraph(T["recovery_timeline"], subheading_style))
        story.append(Paragraph(recovery_timeline, body_style))

    # Farmer Advice (only present when AI-powered)
    farmer_advice_text = treatment_advice.get("farmer_advice")
    if farmer_advice_text:
        story.append(Paragraph(T["farmer_advice"], subheading_style))
        story.append(Paragraph(farmer_advice_text, body_style))

    # ── Uploaded Image ──
    story.append(Paragraph(T["uploaded_image_caption"], heading_style))
    img_buf = BytesIO()
    img_copy = uploaded_image.copy()
    img_copy.thumbnail((800, 800))
    img_copy.save(img_buf, format="PNG")
    img_buf.seek(0)

    orig_w, orig_h = img_copy.size
    max_img_w = page_w * 0.75
    max_img_h = 200 * mm
    scale = min(max_img_w / orig_w, max_img_h / orig_h, 1.0)
    draw_w = orig_w * scale
    draw_h = orig_h * scale

    rl_img = RLImage(img_buf, width=draw_w, height=draw_h)
    story.append(rl_img)
    story.append(Spacer(1, 12))

    # ── Footer callback ──
    def _footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(HexColor("#999999"))
        footer_y = 15 * mm
        canvas.drawString(20 * mm, footer_y,
                          "Generated by Crop Disease Detection System")
        canvas.drawRightString(A4[0] - 20 * mm, footer_y,
                               f"Page {doc.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return buffer.getvalue()


# ── UI Layout ───────────────────────────────────────────────────────

# Language Selector in Sidebar
lang_options = {"English": "en", "ಕನ್ನಡ (Kannada)": "kn", "हिंदी (Hindi)": "hi"}
selected_lang_label = st.sidebar.selectbox("🌐 Language / ಭಾಷೆ / भाषा", list(lang_options.keys()))
lang_code = lang_options[selected_lang_label]
T = UI_LABELS[lang_code]

st.markdown(f"""
<div class="hero-banner">
    <h1>{T['hero_title']}</h1>
    <p>{T['hero_subtitle']}</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown(f"""
<div class="sidebar-how">
    {T['how_it_works_title']}
    <ol>
        <li>{T['how_it_works_1']}</li>
        <li>{T['how_it_works_2']}</li>
        <li>{T['how_it_works_3']}</li>
    </ol>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown(T['upload_title'])
uploaded_file = st.sidebar.file_uploader(
    T['upload_placeholder'], type=["jpg", "jpeg", "png"], label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.caption(T['powered_by'])

col1, col2 = st.columns([1, 1])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    
    with col1:
        # Render image inside a styled card
        display_img = image.copy()
        display_img.thumbnail((800, 800))
        buf = BytesIO()
        display_img.save(buf, format="JPEG", quality=85)
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        html_content = textwrap.dedent(f"""
        <div class="image-card">
            <img src="data:image/jpeg;base64,{img_b64}" alt="Uploaded leaf">
            <div class="caption">{T['uploaded_image_caption']}</div>
        </div>
        """)
        st.markdown(html_content, unsafe_allow_html=True)
    
    with col2:
        if model is None:
            st.error(T['model_not_found'].format(MODEL_PATH=MODEL_PATH))
        else:
            with st.spinner(T['analyzing_spinner']):
                # Preprocess image
                img_resized = image.resize((224, 224))
                img_array = np.array(img_resized) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                # Predict using offline TensorFlow model
                predictions = model.predict(img_array)
                predicted_idx = np.argmax(predictions[0])
                confidence = float(np.max(predictions[0])) * 100
                
                disease_label = CLASS_NAMES[predicted_idx] if predicted_idx < len(CLASS_NAMES) else "Unknown"
                clean_name = disease_label.replace("___", " - ").replace("_", " ")

            # Confidence color-coding
            if confidence > 80:
                badge_class, bar_color = "badge-high", "#2E7D32"
            elif confidence > 50:
                badge_class, bar_color = "badge-medium", "#E65100"
            else:
                badge_class, bar_color = "badge-low", "#C62828"

            with st.spinner(T['ai_spinner']):
                treatment_advice = get_ai_treatment(clean_name, lang_code=lang_code)
                used_ai = treatment_advice is not None
                if treatment_advice is None:
                    treatment_advice = get_treatment(disease_label)

            # Source badge
            if used_ai:
                source_badge = f'<span class="source-badge source-ai">{T["source_ai"]}</span>'
            else:
                offline_suffix = T["offline_note"] if lang_code != "en" else ""
                source_badge = f'<span class="source-badge source-offline">{T["source_offline"]}{offline_suffix}</span>'

            # --- Build result-card HTML via concatenation (NO blank lines) ---
            severity = treatment_advice.get("severity")
            farmer_advice = treatment_advice.get("farmer_advice")
            affected_crops = treatment_advice.get("affected_crops")
            recovery_timeline = treatment_advice.get("recovery_timeline")

            # Severity badge
            severity_html = ""
            if severity:
                sev_class_map = {"Low": "badge-high", "Medium": "badge-medium", "High": "badge-low"}
                sev_class = sev_class_map.get(severity, "badge-medium")
                severity_html = (
                    '<div class="info-section">'
                    f'<div class="section-header">{T["severity"]}</div>'
                    f'<div class="section-body"><span class="confidence-badge {sev_class}">{severity}</span></div>'
                    '</div>'
                )

            # Affected crops
            crops_html = ""
            if affected_crops:
                crops_li = ''.join(f'<li>{c}</li>' for c in affected_crops)
                crops_html = (
                    '<div class="info-section">'
                    f'<div class="section-header">{T["affected_crops"]}</div>'
                    f'<div class="section-body"><ul>{crops_li}</ul></div>'
                    '</div>'
                )

            # Recovery timeline
            timeline_html = ""
            if recovery_timeline:
                timeline_html = (
                    '<div class="info-section">'
                    f'<div class="section-header">{T["recovery_timeline"]}</div>'
                    f'<div class="section-body">{recovery_timeline}</div>'
                    '</div>'
                )

            # Farmer advice
            farmer_html = ""
            if farmer_advice:
                farmer_html = (
                    '<div class="info-section">'
                    f'<div class="section-header">{T["farmer_advice"]}</div>'
                    f'<div class="section-body">{farmer_advice}</div>'
                    '</div>'
                )

            symptoms_li = ''.join(f'<li>{s}</li>' for s in treatment_advice['symptoms'])
            treatment_li = ''.join(f'<li>{t}</li>' for t in treatment_advice['treatment'])
            prevention_li = ''.join(f'<li>{p}</li>' for p in treatment_advice['prevention'])

            html_content = (
                '<div class="result-card">'
                f'<div class="diagnosis-label">{T["diagnosis_result"]}</div>'
                f'<div class="diagnosis-name">{clean_name}</div>'
                '<div class="confidence-container">'
                '<div class="confidence-header">'
                f'<span class="confidence-title">{T["model_confidence"]}</span>'
                f'<span class="confidence-badge {badge_class}">{confidence:.1f}%</span>'
                '</div>'
                f'<div class="progress-bar-bg"><div class="progress-bar-fill" style="width:{confidence:.1f}%; background:{bar_color};"></div></div>'
                '</div>'
                '<div class="treatment-box">'
                f'<div class="treatment-title">{T["treatment_plan"]} {source_badge}</div>'
                f'{severity_html}'
                '<div class="info-section">'
                f'<div class="section-header">{T["about_condition"]}</div>'
                f'<div class="section-body">{treatment_advice["description"]}</div>'
                '</div>'
                '<div class="info-section">'
                f'<div class="section-header">{T["symptoms"]}</div>'
                f'<div class="section-body"><ul>{symptoms_li}</ul></div>'
                '</div>'
                '<div class="info-section">'
                f'<div class="section-header">{T["treatment_steps"]}</div>'
                f'<div class="section-body"><ol>{treatment_li}</ol></div>'
                '</div>'
                '<div class="info-section">'
                f'<div class="section-header">{T["prevention"]}</div>'
                f'<div class="section-body"><ul>{prevention_li}</ul></div>'
                '</div>'
                f'{crops_html}'
                f'{timeline_html}'
                f'{farmer_html}'
                '</div>'
                '</div>'
            )
            st.markdown(html_content, unsafe_allow_html=True)

            # ── Download PDF Report Button ──
            st.markdown('<div class="download-section">', unsafe_allow_html=True)
            st.caption(T["pdf_english_note"])

            if lang_code != "en" and used_ai:
                with st.spinner("Preparing English PDF content..."):
                    pdf_treatment_advice = get_ai_treatment(clean_name, lang_code="en")
                    if pdf_treatment_advice is None:
                        pdf_treatment_advice = get_treatment(disease_label)
            else:
                pdf_treatment_advice = treatment_advice
                
            pdf_bytes = generate_pdf_report(
                disease_name=clean_name,
                confidence=confidence,
                treatment_advice=pdf_treatment_advice,
                uploaded_image=image
            )
            st.download_button(
                label=T["download_pdf"],
                data=pdf_bytes,
                file_name=f"crop_disease_report_{disease_label}.pdf",
                mime="application/pdf",
            )
            st.markdown('</div>', unsafe_allow_html=True)
else:
    html_content = textwrap.dedent(f"""
    <div class="empty-state">
        <div class="empty-icon">🍃</div>
        <div class="empty-title">{T["empty_title"]}</div>
        <div class="empty-text">
            {T["empty_text"]}
        </div>
    </div>
    """)
    st.markdown(html_content, unsafe_allow_html=True)