"""
NURA-CT  —  Neural Understanding & Radiological Analysis for CT
Streamlit GUI for stroke segmentation, tumor classification, and bright-region detection.
"""

import os
import io
import tempfile
import numpy as np
from PIL import Image, ImageDraw
import streamlit as st

# ─── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NURA-CT",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    /* Global overrides */
    .stApp {
        background: linear-gradient(160deg, #0a0e1a 0%, #0f1629 40%, #121a30 100%);
    }

    /* Hero section */
    .hero-title {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        font-size: 3rem;
        background: linear-gradient(135deg, #60a5fa, #a78bfa, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0;
        letter-spacing: -1px;
    }
    .hero-sub {
        font-family: 'Inter', sans-serif;
        font-weight: 400;
        font-size: 1rem;
        color: #94a3b8;
        text-align: center;
        margin-top: 4px;
        margin-bottom: 32px;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1220 0%, #111827 100%);
        border-right: 1px solid rgba(99, 102, 241, 0.15);
    }
    section[data-testid="stSidebar"] .stMarkdown h1 {
        color: #e2e8f0;
    }

    /* Cards */
    .mode-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 16px;
        backdrop-filter: blur(12px);
    }
    .mode-card h3 {
        color: #e2e8f0;
        margin-top: 0;
    }
    .mode-card p {
        color: #94a3b8;
        font-size: 0.9rem;
    }

    /* Result box */
    .result-box {
        background: rgba(16, 185, 129, 0.08);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 10px;
        padding: 20px;
        margin-top: 16px;
    }
    .result-box.warning {
        background: rgba(245, 158, 11, 0.08);
        border-color: rgba(245, 158, 11, 0.3);
    }
    .result-box.error {
        background: rgba(239, 68, 68, 0.08);
        border-color: rgba(239, 68, 68, 0.3);
    }

    /* Stat pills */
    .stat-row {
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
        margin: 12px 0;
    }
    .stat-pill {
        background: rgba(99, 102, 241, 0.1);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 8px;
        padding: 10px 18px;
        color: #c7d2fe;
        font-family: 'Inter', monospace;
        font-size: 0.85rem;
    }
    .stat-pill span {
        color: #818cf8;
        font-weight: 600;
    }

    /* Hide default streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* File uploader styling */
    .stFileUploader > div {
        border-color: rgba(99, 102, 241, 0.3) !important;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# CORE ANALYSIS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

# ─── Bright-region detection (from LLM.py — always available) ──────────────────
def detect_bright_region(image: Image.Image):
    """Threshold-based bright region detector.  Works on any 2-D brain image."""
    gray = image.convert("L")
    arr = np.asarray(gray, dtype=np.float64)
    threshold = arr.mean() + arr.std()
    mask = arr > threshold

    if not mask.any():
        return None, None, None

    ys, xs = np.where(mask)
    box = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
    area_pct = mask.sum() / mask.size * 100
    return box, center, area_pct


def draw_detection(image: Image.Image, box, center):
    """Draw bounding box + centre marker and return annotated image."""
    img = image.copy()
    draw = ImageDraw.Draw(img)
    draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="#22d3ee", width=3)
    r = max(4, int(min(img.size) * 0.015))
    cx, cy = center
    draw.ellipse([(cx - r, cy - r), (cx + r, cy + r)], fill="#f43f5e", outline="#fbbf24")
    draw.text((box[0], box[1] - 14), "Possible abnormality", fill="#22d3ee")
    return img


# ─── Tumor classification (needs trained model) ───────────────────────────────
TUMOR_MODEL_PATH = "brain_tumor_model.h5"
TUMOR_CLASSES = ["Glioma", "Meningioma", "Pituitary"]


def load_tumor_model():
    """Load the tumor classification CNN.  Returns None if unavailable."""
    if not os.path.exists(TUMOR_MODEL_PATH):
        return None
    try:
        import tensorflow as tf
        return tf.keras.models.load_model(TUMOR_MODEL_PATH)
    except Exception:
        return None


def classify_tumor(model, image: Image.Image):
    """Run the tumor CNN on a PIL image and return (class_name, confidence, probs)."""
    import tensorflow as tf
    img = image.resize((150, 150)).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    preds = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(preds))
    return TUMOR_CLASSES[idx], float(preds[idx]), {TUMOR_CLASSES[i]: float(preds[i]) for i in range(3)}


# ─── Stroke segmentation (needs trained model) ────────────────────────────────
STROKE_MODEL_PATH = "Stroke Code and Data/stroke_segmentation_model_3d.h5"


def load_stroke_model():
    if not os.path.exists(STROKE_MODEL_PATH):
        return None
    try:
        import tensorflow as tf
        return tf.keras.models.load_model(STROKE_MODEL_PATH)
    except Exception:
        return None


def segment_stroke(model, nifti_path):
    """Run 3-D U-Net on a NIfTI volume.  Returns (ct_volume, mask_slice, slice_idx)."""
    import nibabel as nib
    from skimage.transform import resize as sk_resize

    ct = nib.load(nifti_path).get_fdata()
    ct_resized = sk_resize(ct, (128, 128, 64), mode="constant", preserve_range=True)
    inp = np.expand_dims(ct_resized, axis=(0, -1))
    pred = model.predict(inp, verbose=0)
    binary = (pred > 0.5).astype(np.float32)

    for i in range(binary.shape[1]):
        if np.any(binary[0, i, :, :, 0]):
            return ct, binary[0, i, :, :, 0], i
    return ct, None, None


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🧠 NURA-CT")
    st.caption("Neural Understanding & Radiological Analysis")
    st.markdown("---")

    mode = st.radio(
        "Analysis Mode",
        ["🔍 Bright-Region Detection", "🧬 Tumor Classification", "🩻 Stroke Segmentation"],
        index=0,
    )

    st.markdown("---")
    st.markdown("##### Model Status")

    # Check model availability
    tumor_ready = os.path.exists(TUMOR_MODEL_PATH)
    stroke_ready = os.path.exists(STROKE_MODEL_PATH)

    st.markdown("🟢 Bright-Region Detector — *always ready*")
    tumor_icon = "🟢" if tumor_ready else "🔴"
    tumor_info = "`brain_tumor_model.h5`" if tumor_ready else "*run `Tumor Train.py` first*"
    st.markdown(f"{tumor_icon} Tumor CNN — {tumor_info}")
    stroke_icon = "🟢" if stroke_ready else "🔴"
    stroke_info = "`ready`" if stroke_ready else "*run `Main Code.py` first*"
    st.markdown(f"{stroke_icon} Stroke 3D U-Net — {stroke_info}")

    st.markdown("---")
    st.markdown(
        "<div style='color:#64748b; font-size:0.75rem; text-align:center;'>"
        "NURA-CT v1.0 &nbsp;·&nbsp; For research purposes only.<br>"
        "Not intended for clinical diagnosis."
        "</div>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="hero-title">NURA-CT</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-sub">Neural Understanding &amp; Radiological Analysis for CT &nbsp;·&nbsp; '
    'Upload a brain scan to begin analysis</p>',
    unsafe_allow_html=True,
)

# ─── MODE: Bright-Region Detection ────────────────────────────────────────────
if mode == "🔍 Bright-Region Detection":
    st.markdown(
        '<div class="mode-card">'
        "<h3>🔍 Bright-Region Detection</h3>"
        "<p>A fast, model-free heuristic that identifies abnormally bright regions in 2-D brain images "
        "using adaptive thresholding (mean + σ). Works on JPEG, PNG, BMP — no trained model required.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader("Upload a brain image", type=["jpg", "jpeg", "png", "bmp", "tiff"], key="bright")

    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")

        with st.spinner("Analyzing image..."):
            box, center, area_pct = detect_bright_region(image)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("##### Original")
            st.image(image, use_container_width=True)

        with col2:
            st.markdown("##### Analysis Result")
            if box is not None:
                annotated = draw_detection(image, box, center)
                st.image(annotated, use_container_width=True)
            else:
                st.image(image, use_container_width=True)

        if box is not None:
            st.markdown(
                '<div class="result-box">'
                "<strong>✅ Possible abnormality detected</strong><br>"
                f'<div class="stat-row">'
                f'<div class="stat-pill">Center: <span>({center[0]:.0f}, {center[1]:.0f})</span></div>'
                f'<div class="stat-pill">Bounding Box: <span>({box[0]}, {box[1]}) → ({box[2]}, {box[3]})</span></div>'
                f'<div class="stat-pill">Bright Area: <span>{area_pct:.1f}%</span> of image</div>'
                f"</div>"
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="result-box warning">'
                "<strong>⚠️ No prominent bright region found</strong><br>"
                "The image did not contain pixels exceeding the adaptive threshold (mean + σ)."
                "</div>",
                unsafe_allow_html=True,
            )

# ─── MODE: Tumor Classification ───────────────────────────────────────────────
elif mode == "🧬 Tumor Classification":
    st.markdown(
        '<div class="mode-card">'
        "<h3>🧬 Brain Tumor Classification</h3>"
        "<p>A 4-layer CNN trained on MRI data classifies brain images into three categories: "
        "<strong>Glioma</strong>, <strong>Meningioma</strong>, or <strong>Pituitary</strong> tumor.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    if not tumor_ready:
        st.markdown(
            '<div class="result-box error">'
            "<strong>🔴 Model not found</strong><br>"
            "Run <code>python 'Tumor Train.py'</code> first to train the CNN and generate "
            "<code>brain_tumor_model.h5</code>."
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        uploaded = st.file_uploader("Upload an MRI image", type=["jpg", "jpeg", "png", "bmp", "tiff"], key="tumor")

        if uploaded is not None:
            image = Image.open(uploaded).convert("RGB")

            with st.spinner("Loading model & classifying..."):
                model = load_tumor_model()
                if model is None:
                    st.error("Failed to load model.")
                else:
                    cls_name, confidence, probs = classify_tumor(model, image)

            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown("##### Uploaded MRI")
                st.image(image, use_container_width=True)

            with col2:
                st.markdown("##### Classification Result")
                st.markdown(
                    '<div class="result-box">'
                    f"<strong>Predicted: {cls_name}</strong> &nbsp; ({confidence * 100:.1f}% confidence)"
                    "</div>",
                    unsafe_allow_html=True,
                )
                st.markdown("")
                for label, prob in sorted(probs.items(), key=lambda x: -x[1]):
                    pct = prob * 100
                    color = "#22d3ee" if label == cls_name else "#475569"
                    st.markdown(
                        f"<div style='margin-bottom:8px;'>"
                        f"<span style='color:#e2e8f0;font-weight:600;'>{label}</span>"
                        f"<div style='background:#1e293b;border-radius:6px;height:24px;margin-top:4px;overflow:hidden;'>"
                        f"<div style='background:{color};height:100%;width:{pct}%;border-radius:6px;'></div>"
                        f"</div>"
                        f"<span style='color:#94a3b8;font-size:0.8rem;'>{pct:.1f}%</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

# ─── MODE: Stroke Segmentation ────────────────────────────────────────────────
elif mode == "🩻 Stroke Segmentation":
    st.markdown(
        '<div class="mode-card">'
        "<h3>🩻 Stroke Segmentation</h3>"
        "<p>A 3D U-Net segments stroke regions in NIfTI (.nii) CT scan volumes. "
        "The model highlights the first detected slice containing a stroke prediction.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    if not stroke_ready:
        st.markdown(
            '<div class="result-box error">'
            "<strong>🔴 Model not found</strong><br>"
            "Run <code>cd 'Stroke Code and Data' && python 'Main Code.py'</code> first to train the "
            "3D U-Net and generate <code>stroke_segmentation_model_3d.h5</code>."
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        uploaded = st.file_uploader("Upload a NIfTI CT scan (.nii)", type=["nii"], key="stroke")

        if uploaded is not None:
            with st.spinner("Running 3D U-Net segmentation..."):
                with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as tmp:
                    tmp.write(uploaded.read())
                    tmp_path = tmp.name

                model = load_stroke_model()
                if model is None:
                    st.error("Failed to load stroke model.")
                else:
                    ct_vol, mask_slice, slice_idx = segment_stroke(model, tmp_path)
                    os.unlink(tmp_path)

                    if mask_slice is not None:
                        import matplotlib
                        matplotlib.use("Agg")
                        import matplotlib.pyplot as plt

                        fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor="#0f1629")

                        axes[0].imshow(ct_vol[:, :, slice_idx], cmap="gray")
                        axes[0].set_title("Original CT Slice", color="white", fontsize=13)
                        axes[0].axis("off")

                        axes[1].imshow(ct_vol[:, :, slice_idx], cmap="gray")
                        axes[1].imshow(mask_slice, alpha=0.35, cmap="Reds")
                        axes[1].set_title("Stroke Segmentation Overlay", color="white", fontsize=13)
                        axes[1].axis("off")

                        for ax in axes:
                            ax.set_facecolor("#0f1629")
                        fig.tight_layout()

                        buf = io.BytesIO()
                        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                                    facecolor="#0f1629", edgecolor="none")
                        buf.seek(0)
                        plt.close(fig)

                        st.image(buf, use_container_width=True)

                        st.markdown(
                            '<div class="result-box">'
                            f"<strong>✅ Stroke region detected</strong><br>"
                            f'<div class="stat-row">'
                            f'<div class="stat-pill">Slice: <span>#{slice_idx}</span></div>'
                            f'<div class="stat-pill">Volume shape: <span>{ct_vol.shape}</span></div>'
                            f"</div>"
                            "</div>",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            '<div class="result-box warning">'
                            "<strong>⚠️ No stroke region detected</strong><br>"
                            "The model did not find any positive predictions above the 0.5 threshold "
                            "in any slice of this volume."
                            "</div>",
                            unsafe_allow_html=True,
                        )
