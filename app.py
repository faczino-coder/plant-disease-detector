import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "Plant Disease Detector",
    page_icon  = "🌿",
    layout     = "centered"
)

# ── Load Model & Configs ──────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("models/plant_disease_model.h5")

@st.cache_resource
def load_configs():
    with open("data/label_map.json")     as f: label_map     = json.load(f)
    with open("data/treatment_map.json") as f: treatment_map = json.load(f)
    return label_map, treatment_map

model                    = load_model()
label_map, treatment_map = load_configs()

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
    <style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Main App */
.stApp {
    background: linear-gradient(
        180deg,
        #f4f8f2 0%,
        #edf7ed 50%,
        #e8f5e9 100%
    );
    color: #2e3b2e;
}

#MainMenu,
footer,
header {
    visibility: hidden;
}

/* Hero */
.hero {
    background:
    linear-gradient(
        rgba(46,125,50,0.88),
        rgba(67,160,71,0.88)
    ),
    url('https://images.unsplash.com/photo-1500937386664-56d1dfef3854');

    background-size: cover;
    background-position: center;
    border-radius: 20px;
    padding: 1.5rem 2rem;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 10px 30px rgba(46,125,50,0.25);
}

.hero h1 {
    font-size: 1.8rem;
    font-weight: 800;
    color: white;
    margin: 0;
}

.hero p {
    color: rgba(255,255,255,0.95);
    font-size: 1rem;
    margin-top: 0.6rem;
}

/* Upload Area */
.upload-box {
    background-color: white;
    border: 2px dashed #66bb6a;
    border-radius: 16px;
    padding: 2.5rem;
    text-align: center;
    margin: 1rem 0;
    box-shadow: 0 5px 20px rgba(0,0,0,0.05);
}

.upload-box h3 {
    color: #2e7d32;
    font-size: 1.1rem;
}

/* Standard Card */
.card {
    background: white;
    border: 1px solid #dcedc8;
    border-radius: 14px;
    padding: 1.5rem;
    margin: 1rem 0;
    box-shadow: 0 5px 20px rgba(0,0,0,0.05);
}

/* Healthy */
.healthy-card {
    background: linear-gradient(
        135deg,
        #e8f5e9 0%,
        #f1f8e9 100%
    );
    border: 2px solid #43a047;
    border-radius: 14px;
    padding: 1.5rem;
    margin: 1rem 0;
    box-shadow: 0 0 25px rgba(67,160,71,0.15);
}

/* Disease */
.disease-card {
    background: linear-gradient(
        135deg,
        #fff8e1 0%,
        #fffde7 100%
    );
    border: 2px solid #ef6c00;
    border-radius: 14px;
    padding: 1.5rem;
    margin: 1rem 0;
    box-shadow: 0 0 25px rgba(239,108,0,0.15);
}

/* Critical */
.critical-card {
    background: linear-gradient(
        135deg,
        #ffebee 0%,
        #fff5f5 100%
    );
    border: 2px solid #d32f2f;
    border-radius: 14px;
    padding: 1.5rem;
    margin: 1rem 0;
    box-shadow: 0 0 25px rgba(211,47,47,0.15);
}

.card-title {
    font-size: 1.2rem;
    font-weight: 800;
    margin-bottom: 0.8rem;
    color: #1b5e20;
}

.card-meta {
    color: #5f6f5f;
    font-size: 0.92rem;
    margin: 0.4rem 0;
}

.card-meta span {
    color: #2e3b2e;
    font-weight: 600;
}

/* Badges */
.badge {
    display: inline-block;
    padding: 0.35rem 0.8rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 700;
    margin-top: 0.6rem;
}

.badge-none,
.badge-healthy {
    background: #e8f5e9;
    color: #2e7d32;
}

.badge-moderate {
    background: #fff3e0;
    color: #ef6c00;
}

.badge-high {
    background: #ffe0b2;
    color: #e65100;
}

.badge-critical {
    background: #ffebee;
    color: #d32f2f;
}

/* Confidence */
.conf-wrap {
    margin: 1rem 0 0.5rem;
}

.conf-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.85rem;
    color: #5f6f5f;
    margin-bottom: 4px;
}

.conf-bar-bg {
    background-color: #dfe7df;
    border-radius: 10px;
    height: 10px;
    overflow: hidden;
}

.conf-bar-fill {
    height: 10px;
    border-radius: 10px;
}

/* Titles */
.section-title {
    font-size: 1rem;
    font-weight: 800;
    color: #2e7d32;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin: 1.5rem 0 0.8rem;
}

/* Treatment Steps */
.step-item {
    display: flex;
    align-items: flex-start;
    gap: 0.8rem;
    background-color: #f1f8e9;
    border: 1px solid #c5e1a5;
    border-radius: 10px;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
    color: #2e3b2e;
    font-size: 0.92rem;
}

.step-num {
    background: #43a047;
    color: white;
    font-weight: 800;
    font-size: 0.75rem;
    min-width: 24px;
    height: 24px;
    border-radius: 50%;
    display: flex;
    justify-content: center;
    align-items: center;
}

/* Prevention */
.prev-item {
    display: flex;
    align-items: flex-start;
    gap: 0.8rem;
    background-color: #e8f5e9;
    border: 1px solid #a5d6a7;
    border-radius: 10px;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
    color: #2e3b2e;
    font-size: 0.92rem;
}

/* Top Predictions */
.top3-item {
    background-color: #f7faf7;
    border: 1px solid #dcedc8;
    border-radius: 10px;
    padding: 0.7rem 1rem;
    margin: 0.4rem 0;
    color: #5f6f5f;
}

/* Divider */
.divider {
    border: none;
    border-top: 1px solid #c8e6c9;
    margin: 1.5rem 0;
}

/* Footer */
.footer {
    text-align: center;
    color: #558b2f;
    font-size: 0.85rem;
    font-weight: 600;
    padding: 1.5rem 0 0.5rem;
}

/* File Uploader */
.stFileUploader > div {
    background-color: white !important;
    border: 2px dashed #66bb6a !important;
    border-radius: 12px !important;
}
/* File uploader label */
label[data-testid="stWidgetLabel"] {
    color: #2e7d32 !important;
    font-weight: 700 !important;
}
/* Expander */
div[data-testid="stExpander"] {
    background-color: white;
    border: 1px solid #dcedc8;
    border-radius: 12px;
}     
    </style>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
    <div class="hero">
        <h1>🌿 Plant Disease Detector</h1>
        <p>Upload a leaf photo → Get instant diagnosis + treatment advice</p>
    </div>
""", unsafe_allow_html=True)

# ── Supported Crops ───────────────────────────────────────────────────────────
with st.expander("📋 Supported Crops & Diseases"):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🍅 Tomato**")
        for cls in label_map["class_names"]:
            if cls.startswith("Tomato"):
                name = cls.replace("Tomato___", "").replace("_", " ")
                icon = "✅" if "healthy" in cls.lower() else "🔴"
                st.markdown(f"{icon} {name}")
    with col2:
        st.markdown("**🌽 Maize (Corn)**")
        for cls in label_map["class_names"]:
            if cls.startswith("Corn"):
                name = cls.replace("Corn_(maize)___", "").replace("_", " ")
                icon = "✅" if "healthy" in cls.lower() else "🔴"
                st.markdown(f"{icon} {name}")

# ── Upload ────────────────────────────────────────────────────────────────────
st.markdown('<p class="section-title">📸 Upload Leaf Image</p>',
            unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choose a JPG or PNG image of a leaf",
    type = ["jpg", "jpeg", "png"],
    help = "Take a clear close-up photo of a single leaf in good lighting"
)

# ── Helper Functions ──────────────────────────────────────────────────────────
def predict(image):
    img_array   = np.array(image.resize((224, 224)).convert("RGB")) / 255.0
    predictions = model.predict(np.expand_dims(img_array, 0), verbose=0)[0]
    top3_idx    = predictions.argsort()[-3:][::-1]
    return [
        {
            "class"     : label_map["idx_to_class"][str(i)],
            "confidence": float(predictions[i]) * 100
        }
        for i in top3_idx
    ]

def get_badge(urgency):
    u = urgency.lower()
    if u == "none":
        return '<span class="badge badge-none">✅ Healthy</span>'
    elif "critical" in u:
        return '<span class="badge badge-critical">🚨 Critical</span>'
    elif "high" in u:
        return '<span class="badge badge-high">⚠️ High</span>'
    elif "moderate" in u:
        return '<span class="badge badge-moderate">🔶 Moderate</span>'
    return f'<span class="badge badge-moderate">{urgency}</span>'

def build_card(card_class, icon, disease_name, crop_name,
               cause, symptoms, urgency, confidence, conf_color):

    # Build optional rows safely
    if cause:
        cause_row = (
            "<div class='card-meta'>🦠 Cause &nbsp;&nbsp;"
            f"<span>{cause}</span></div>"
        )
    else:
        cause_row = ""

    if symptoms:
        symptoms_row = (
            "<div class='card-meta'>👁️ Symptoms &nbsp;&nbsp;"
            f"<span>{symptoms}</span></div>"
        )
    else:
        symptoms_row = ""

    badge = get_badge(urgency)

    conf_bar = (
        "<div class='conf-wrap'>"
            "<div class='conf-label'>"
                "<span>Confidence</span>"
                f"<span>{confidence:.1f}%</span>"
            "</div>"
            "<div class='conf-bar-bg'>"
                f"<div class='conf-bar-fill' "
                f"style='width:{confidence}%; background:{conf_color};'>"
                "</div>"
            "</div>"
        "</div>"
    )

    return (
        f"<div class='{card_class}'>"
            f"<div class='card-title'>{icon} {disease_name}</div>"
            f"<div class='card-meta'>🌱 Crop &nbsp;&nbsp;<span>{crop_name}</span></div>"
            + cause_row
            + symptoms_row
            + badge
            + conf_bar
        + "</div>"
    )

# ── Main Logic ────────────────────────────────────────────────────────────────
if uploaded_file:
    image = Image.open(uploaded_file)

    # Show image
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image, caption="Uploaded Leaf", use_column_width=True)

    # Predict
    with st.spinner("🔍 Analysing leaf..."):
        top3       = predict(image)
        best       = top3[0]
        best_class = best["class"]
        confidence = best["confidence"]
        info       = treatment_map.get(best_class, None)

    is_healthy = "healthy" in best_class.lower()
    urgency    = info["urgency"]  if info else "Unknown"
    cause      = info["cause"]    if info and info["cause"] else ""
    symptoms   = info["symptoms"] if info and info["symptoms"] else ""

    if is_healthy:
        card_class, icon, conf_color = "healthy-card", "✅", "#39d353"
    elif "critical" in urgency.lower():
        card_class, icon, conf_color = "critical-card", "🚨", "#e74c3c"
    else:
        card_class, icon, conf_color = "disease-card",  "⚠️", "#e67e22"

    disease_name = info["disease"] if info else best_class.replace("_", " ")
    crop_name    = info["crop"]    if info else "Unknown"

    # ── Diagnosis Card ────────────────────────────────────────────────────────
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">🔬 Diagnosis Result</p>',
                unsafe_allow_html=True)

    card_html = build_card(
        card_class, icon, disease_name, crop_name,
        cause, symptoms, urgency, confidence, conf_color
    )
    st.markdown(card_html, unsafe_allow_html=True)

    # ── Top 3 ─────────────────────────────────────────────────────────────────
    with st.expander("📊 Top 3 Predictions"):
        for i, pred in enumerate(top3, 1):
            name = (pred["class"]
                    .replace("Corn_(maize)___", "Corn: ")
                    .replace("Tomato___", "Tomato: ")
                    .replace("_", " "))
            conf = pred["confidence"]
            item = (
                f"<div class='top3-item'>"
                    f"<div style='display:flex;justify-content:space-between;margin-bottom:4px'>"
                        f"<span>#{i} {name}</span>"
                        f"<span style='color:#e6edf3;font-weight:600'>{conf:.1f}%</span>"
                    f"</div>"
                    f"<div class='conf-bar-bg'>"
                        f"<div class='conf-bar-fill' "
                        f"style='width:{conf}%;background:#58a6ff;'></div>"
                    f"</div>"
                f"</div>"
            )
            st.markdown(item, unsafe_allow_html=True)

    # ── Treatment ─────────────────────────────────────────────────────────────
    if info:
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown('<p class="section-title">💊 Treatment Recommendations</p>',
                    unsafe_allow_html=True)

        st.markdown("<div class='card'>", unsafe_allow_html=True)

        st.markdown(
            "<p style='color:#39d353;font-weight:700;margin-bottom:0.5rem'>"
            "🛠️ Treatment Steps</p>",
            unsafe_allow_html=True
        )
        for i, step in enumerate(info["treatment"], 1):
            st.markdown(
                f"<div class='step-item'>"
                    f"<div class='step-num'>{i}</div>"
                    f"<div>{step}</div>"
                f"</div>",
                unsafe_allow_html=True
            )

        st.markdown(
            "<p style='color:#58a6ff;font-weight:700;margin:1rem 0 0.5rem'>"
            "🛡️ Prevention Tips</p>",
            unsafe_allow_html=True
        )
        for tip in info.get("prevention", []):
            st.markdown(
                f"<div class='prev-item'>"
                    f"<div>🔹</div>"
                    f"<div>{tip}</div>"
                f"</div>",
                unsafe_allow_html=True
            )

        st.markdown("</div>", unsafe_allow_html=True)

    # ── Disclaimer ────────────────────────────────────────────────────────────
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#484f58;font-size:0.82rem;text-align:center'>"
        "⚠️ For guidance only. Consult a professional agronomist for severe infections."
        "</p>",
        unsafe_allow_html=True
    )

else:
    st.markdown("""
        <div class="upload-box">
            <h3>📁 No image uploaded yet</h3>
            <p style="color:#484f58;font-size:0.88rem">
                Upload a clear photo of a tomato or maize leaf to get started
            </p>
        </div>
    """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    "<div class='footer'>"
    "🌿 PlantVillage Disease Classifier — Tomato & Maize &nbsp;|&nbsp; Powered by MobileNetV2"
    "</div>",
    unsafe_allow_html=True
)