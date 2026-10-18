
import streamlit as st
from PIL import Image
from pathlib import Path
import sys

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from ml.inference import build_model, predict, CLASS_NAMES


# ─────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Environmental Monitoring AI",
    page_icon="🌍",
    layout="wide"
)


# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────

WEIGHTS_PATH = (
    PROJECT_ROOT
    / "models"
    / "eurosat_vit_5class_best.weights.h5"
)


# ─────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────

@st.cache_resource
def load_model():
    return build_model(str(WEIGHTS_PATH))


# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────

st.title("🌍 AI Environmental Monitoring Platform")

st.markdown(
    """
    **AI-powered land-use and land-cover classification**

    Upload a satellite image to identify its dominant land-use category
    using a Vision Transformer (ViT-B/16) model.
    """
)

st.divider()


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────

st.sidebar.header("Monitoring Categories")

for class_name in CLASS_NAMES:
    st.sidebar.write(f"• {class_name}")


# ─────────────────────────────────────────────
# Image upload
# ─────────────────────────────────────────────

uploaded_file = st.file_uploader(
    "Upload a satellite image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded Image")
        st.image(
            image,
            caption="Satellite Image",
            use_container_width=True
        )

    with st.spinner("Analyzing satellite image..."):

        model = load_model()
        result = predict(model, image)

    with col2:

        st.subheader("AI Classification")

        st.success(
            f"Predicted Land Use: **{result['class']}**"
        )

        st.metric(
            "Confidence",
            f"{result['confidence'] * 100:.2f}%"
        )

        st.subheader("Class Probabilities")

        for class_name, probability in zip(
            CLASS_NAMES,
            result["probabilities"]
        ):
            st.write(
                f"**{class_name}** — "
                f"{probability * 100:.2f}%"
            )

            st.progress(float(probability))


else:

    st.info(
        "Upload a satellite image above to begin environmental analysis."
    )


# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────

st.divider()

st.caption(
    "AI Environmental Monitoring Platform | "
    "Vision Transformer trained on a 5-class EuroSAT-derived dataset"
)

