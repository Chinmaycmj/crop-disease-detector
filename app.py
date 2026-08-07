import os
import textwrap
import base64
from io import BytesIO
from datetime import datetime
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
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
    story.append(Paragraph("Prediction Result", heading_style))

    conf_color = HexColor("#2E7D32") if confidence > 80 else (
        HexColor("#E65100") if confidence > 50 else HexColor("#C62828")
    )
    result_data = [
        [Paragraph("<b>Disease Name</b>", body_style),
         Paragraph(disease_name, body_style)],
        [Paragraph("<b>Model Confidence</b>", body_style),
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
    story.append(Paragraph("Medical Information", heading_style))

    # About
    story.append(Paragraph("About This Condition", subheading_style))
    story.append(Paragraph(treatment_advice["description"], body_style))

    # Symptoms (bullet list)
    story.append(Paragraph("Symptoms", subheading_style))
    symptom_items = [
        ListItem(Paragraph(s, list_item_style)) for s in treatment_advice["symptoms"]
    ]
    story.append(ListFlowable(symptom_items, bulletType="bullet", start="",
                              bulletFontSize=8, leftIndent=18, spaceBefore=2, spaceAfter=2))

    # Treatment Steps (numbered list)
    story.append(Paragraph("Treatment Steps", subheading_style))
    treatment_items = [
        ListItem(Paragraph(t, list_item_style)) for t in treatment_advice["treatment"]
    ]
    story.append(ListFlowable(treatment_items, bulletType="1",
                              bulletFontSize=10, leftIndent=18, spaceBefore=2, spaceAfter=2))

    # Prevention Tips (bullet list)
    story.append(Paragraph("Prevention Tips", subheading_style))
    prevention_items = [
        ListItem(Paragraph(p, list_item_style)) for p in treatment_advice["prevention"]
    ]
    story.append(ListFlowable(prevention_items, bulletType="bullet", start="",
                              bulletFontSize=8, leftIndent=18, spaceBefore=2, spaceAfter=2))

    # ── Uploaded Image ──
    story.append(Paragraph("Uploaded Image", heading_style))
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
st.markdown("""
<div class="hero-banner">
    <h1>🌱 Crop Disease Detection &amp; Care Assistant</h1>
    <p>Upload a leaf image to receive instant disease diagnosis powered by MobileNetV2 deep learning</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div class="sidebar-how">
    <h4>🔬 How It Works</h4>
    <ol>
        <li>Upload a photo of an affected crop leaf</li>
        <li>Our MobileNetV2 model analyzes the image locally</li>
        <li>Receive an instant diagnosis &amp; treatment plan</li>
    </ol>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("#### 📤 Upload Leaf Image")
uploaded_file = st.sidebar.file_uploader(
    "Choose a leaf image...", type=["jpg", "jpeg", "png"], label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.caption("🌿 Powered by MobileNetV2 · TensorFlow")

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
            <div class="caption">📷 Uploaded Leaf Image</div>
        </div>
        """)
        st.markdown(html_content, unsafe_allow_html=True)
    
    with col2:
        if model is None:
            st.error(f"Model file not found at `{MODEL_PATH}`. Please ensure `plant_disease_model.h5` is placed inside the `model/` directory.")
        else:
            with st.spinner("Analyzing image using local model..."):
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

            treatment_advice = get_treatment(disease_label)

            html_content = textwrap.dedent(f"""
            <div class="result-card">
                <div class="diagnosis-label">Diagnosis Result</div>
                <div class="diagnosis-name">{clean_name}</div>
                <div class="confidence-container">
                    <div class="confidence-header">
                        <span class="confidence-title">Model Confidence</span>
                        <span class="confidence-badge {badge_class}">{confidence:.1f}%</span>
                    </div>
                    <div class="progress-bar-bg">
                        <div class="progress-bar-fill" style="width:{confidence:.1f}%; background:{bar_color};"></div>
                    </div>
                </div>
                <div class="treatment-box">
                    <div class="treatment-title">📋 Recommended Treatment Plan</div>
                    <div class="info-section">
                        <div class="section-header">🔍 About This Condition</div>
                        <div class="section-body">{treatment_advice['description']}</div>
                    </div>
                    <div class="info-section">
                        <div class="section-header">🩺 Symptoms</div>
                        <div class="section-body"><ul>{''.join(f'<li>{s}</li>' for s in treatment_advice['symptoms'])}</ul></div>
                    </div>
                    <div class="info-section">
                        <div class="section-header">💊 Treatment Steps</div>
                        <div class="section-body"><ol>{''.join(f'<li>{t}</li>' for t in treatment_advice['treatment'])}</ol></div>
                    </div>
                    <div class="info-section">
                        <div class="section-header">🛡️ Prevention</div>
                        <div class="section-body"><ul>{''.join(f'<li>{p}</li>' for p in treatment_advice['prevention'])}</ul></div>
                    </div>
                </div>
            </div>
            """)
            st.markdown(html_content, unsafe_allow_html=True)

            # ── Download PDF Report Button ──
            pdf_bytes = generate_pdf_report(
                disease_name=clean_name,
                confidence=confidence,
                treatment_advice=treatment_advice,
                uploaded_image=image,
            )
            st.download_button(
                label="\U0001f4c4 Download Report (PDF)",
                data=pdf_bytes,
                file_name=f"crop_disease_report_{disease_label}.pdf",
                mime="application/pdf",
            )
else:
    html_content = textwrap.dedent("""
    <div class="empty-state">
        <div class="empty-icon">🍃</div>
        <div class="empty-title">Ready to Diagnose Your Crop</div>
        <div class="empty-text">
            Upload a leaf image using the sidebar panel to get an instant
            AI-powered disease diagnosis and personalized treatment recommendations.
        </div>
    </div>
    """)
    st.markdown(html_content, unsafe_allow_html=True)