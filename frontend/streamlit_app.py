"""
streamlit_app.py
----------------
Streamlit Web Dashboard for the Bank Document Classifier & OCR pipeline.

Run with:
    streamlit run frontend/streamlit_app.py
"""

import os
import sys
import json
from PIL import Image
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.models.transfer_models import build_model
from src.dataset import EVAL_TRANSFORMS
from src.gradcam import run_gradcam
from src.ocr_extraction import extract_cheque_fields
import torch

st.set_page_config(
    page_title="Bank Document Classifier & OCR",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #10b981;
    }
    .metric-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 16px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🏦 Bank Document Classification & Information Extraction AI")
st.caption("End-to-end Deep Learning Document Classifier with Grad-CAM Explainability & OCR field extraction.")

# Sidebar Settings
st.sidebar.header("⚙️ Model & Settings")
selected_model = st.sidebar.selectbox(
    "Select Neural Architecture",
    options=config.MODEL_NAMES,
    index=2,  # resnet50
    format_func=lambda m: f"{m.upper()} (Trained)"
)

inference_mode = st.sidebar.radio(
    "Inference Mode",
    options=["⚡ Classification", "🔥 Grad-CAM Explainability", "📋 OCR Field Extraction"]
)

# Tabs
tab1, tab2, tab3 = st.tabs(["🚀 Live Classifier", "📊 Model Benchmarks", "📖 Pipeline Architecture"])

with tab1:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("1. Document Input")
        
        # Sample Document Picker
        st.markdown("**Quick Sample Test:**")
        sample_cols = st.columns(4)
        sample_choice = None
        for i, c in enumerate(config.CLASS_NAMES[:4]):
            if sample_cols[i].button(f"{c.replace('_', ' ').title()}", key=f"s_{c}"):
                sample_choice = os.path.join(config.DATA_RAW_DIR, c, f"{c}_0000.png")
        
        sample_cols2 = st.columns(4)
        for i, c in enumerate(config.CLASS_NAMES[4:]):
            if sample_cols2[i].button(f"{c.replace('_', ' ').title()}", key=f"s_{c}"):
                sample_choice = os.path.join(config.DATA_RAW_DIR, c, f"{c}_0000.png")

        uploaded_file = st.file_uploader("Or Upload Custom Bank Document", type=["png", "jpg", "jpeg"])

        img_to_process = None
        if uploaded_file is not None:
            img_to_process = Image.open(uploaded_file).convert("RGB")
            st.image(img_to_process, caption="Uploaded Document Preview", use_container_width=True)
        elif sample_choice and os.path.exists(sample_choice):
            img_to_process = Image.open(sample_choice).convert("RGB")
            st.image(img_to_process, caption=f"Sample Document: {os.path.basename(sample_choice)}", use_container_width=True)
            st.session_state['sample_path'] = sample_choice

    with col2:
        st.subheader("2. AI Analysis & Results")

        if img_to_process is not None:
            if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
                with st.spinner("Analyzing document with PyTorch backbone..."):
                    device = torch.device(config.DEVICE)
                    ckpt_path = os.path.join(config.MODELS_DIR, f"{selected_model}.pt")
                    
                    if not os.path.exists(ckpt_path):
                        st.error(f"Checkpoint for {selected_model} not found in {config.MODELS_DIR}!")
                    else:
                        checkpoint = torch.load(ckpt_path, map_location=device)
                        model = build_model(selected_model, config.NUM_CLASSES, pretrained=False).to(device)
                        model.load_state_dict(checkpoint["state_dict"])
                        model.eval()

                        tensor = EVAL_TRANSFORMS(img_to_process).unsqueeze(0).to(device)
                        with torch.no_grad():
                            outputs = model(tensor)
                            probs = torch.softmax(outputs, dim=1)[0]
                            class_idx = int(probs.argmax().item())
                            confidence = float(probs[class_idx].item())

                        pred_class = config.CLASS_NAMES[class_idx]

                        st.success(f"### Predicted Label: **{pred_class.replace('_', ' ').upper()}**")
                        st.metric("Confidence Score", f"{confidence * 100:.2f}%")

                        # Probabilities Breakdown
                        st.markdown("#### Class Probabilities:")
                        prob_dict = {config.CLASS_NAMES[i]: float(probs[i].item()) for i in range(len(config.CLASS_NAMES))}
                        st.bar_chart(prob_dict)

                        # Grad-CAM
                        if "Grad-CAM" in inference_mode:
                            st.markdown("#### 🔥 Grad-CAM Activation Heatmap")
                            # Save temp
                            temp_path = os.path.join(config.OUTPUTS_DIR, "temp_stream.png")
                            img_to_process.save(temp_path)
                            save_path, _, _ = run_gradcam(temp_path, selected_model)
                            st.image(save_path, caption=f"Attention Map ({selected_model})", use_container_width=True)

                        # OCR Extraction
                        if "OCR" in inference_mode:
                            st.markdown("#### 📋 Extracted Structured Fields")
                            temp_path = os.path.join(config.OUTPUTS_DIR, "temp_stream.png")
                            img_to_process.save(temp_path)
                            if pred_class == "cheque":
                                fields = extract_cheque_fields(temp_path)
                                st.json(fields)
                            else:
                                st.info(f"OCR extraction is tuned for cheques. Classified document is '{pred_class}'.")

with tab2:
    st.subheader("Model Comparison Benchmarks")
    json_path = os.path.join(config.REPORTS_DIR, "model_comparison.json")
    if os.path.exists(json_path):
        with open(json_path) as f:
            data = json.load(f)
        st.dataframe(data, use_container_width=True)

    st.subheader("Confusion Matrix")
    cm_model = st.selectbox("Inspect Confusion Matrix for Model", config.MODEL_NAMES)
    cm_path = os.path.join(config.REPORTS_DIR, f"confusion_matrix_{cm_model}.png")
    if os.path.exists(cm_path):
        st.image(cm_path, caption=f"Confusion Matrix - {cm_model}", use_container_width=True)

with tab3:
    st.subheader("Pipeline Architecture (README.md Summary)")
    st.markdown("""
    1. **Synthetic Data Generation**: `python3 data/generate_synthetic_data.py --per_class 60`
    2. **OpenCV Preprocessing**: `python3 src/data_preprocessing.py` (Denoise, deskew, auto-crop)
    3. **Model Training**: `python3 -m src.train --model all --epochs 8`
    4. **Model Evaluation**: `python3 -m src.evaluate`
    5. **Grad-CAM Explainability**: `python3 -m src.gradcam --image ... --model resnet50`
    6. **OCR Field Extraction**: `python3 -m src.ocr_extraction --image ...`
    7. **REST API & Web UI**: `uvicorn api.main:app --reload --port 8000`
    """)
