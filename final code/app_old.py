"""
╔══════════════════════════════════════════════════════════════╗
║   🌿  CropGuard AI — Multi-Crop Disease Detection           ║
║   T1.7 (38-class unified model) + T1.8 (PWA-ready)         ║
╚══════════════════════════════════════════════════════════════╝
"""

import os, json, time, random
from pathlib import Path
import numpy as np
from PIL import Image
import streamlit as st

# ─── Page config (must be first) ─────────────────────────────
st.set_page_config(
    page_title="CropGuard AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Constants ────────────────────────────────────────────────
ROOT = Path(__file__).parent
DISEASE_INFO_PATH = ROOT / "disease_info.json"
CLASS_NAMES_PATH  = ROOT / "class_names.json"
MODEL_PATH        = ROOT / "best_model.pth"

IMG_SIZE  = 224
DEMO_MODE = not MODEL_PATH.exists()          # True when no trained model yet

# ─── Load resources ───────────────────────────────────────────
@st.cache_resource
def load_disease_info():
    with open(DISEASE_INFO_PATH, encoding="utf-8") as f:
        return json.load(f)

@st.cache_resource
def load_class_names():
    if CLASS_NAMES_PATH.exists():
        with open(CLASS_NAMES_PATH, encoding="utf-8") as f:
            return json.load(f)
    # Fallback: derive from disease_info keys
    info = load_disease_info()
    return list(info.keys())

def lookup_disease(raw_class: str, disease_info: dict) -> tuple:
    """
    Robust lookup: tries exact match, then fuzzy normalisation,
    then token overlap. Always returns (info_dict, matched_key).
    """
    # 1. Exact match
    if raw_class in disease_info:
        return disease_info[raw_class], raw_class

    # 2. Normalise both sides: lowercase, collapse underscores/spaces
    def norm(s):
        import re
        return re.sub(r"[\s_]+", " ", s).lower().strip()

    raw_norm = norm(raw_class)
    for key in disease_info:
        if norm(key) == raw_norm:
            return disease_info[key], key

    # 3. Contains match
    for key in disease_info:
        k = norm(key)
        if raw_norm in k or k in raw_norm:
            return disease_info[key], key

    # 4. Best token overlap
    raw_tokens = set(raw_norm.split())
    best_key, best_score = None, 0
    for key in disease_info:
        tokens = set(norm(key).split())
        score  = len(raw_tokens & tokens)
        if score > best_score:
            best_score, best_key = score, key
    if best_key and best_score > 0:
        return disease_info[best_key], best_key

    return {}, raw_class

@st.cache_resource(show_spinner="Loading AI model…")
def load_model():
    """Load PyTorch model. Returns None in demo mode."""
    if DEMO_MODE:
        return None, None
    try:
        import torch, timm
        from torchvision import transforms

        ckpt = torch.load(MODEL_PATH, map_location="cpu")
        class_names = ckpt.get("class_names", load_class_names())
        num_classes = len(class_names)

        model = timm.create_model("mobilenetv3_small_100",
                                  pretrained=False,
                                  num_classes=num_classes)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        tfm = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])
        return model, (tfm, class_names)
    except Exception as e:
        st.warning(f"Model load error: {e} — falling back to demo mode.")
        return None, None

# ─── Inference helpers ────────────────────────────────────────
def predict_real(img: Image.Image, model, tfm, class_names, top_k=5):
    import torch
    x = tfm(img.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)[0]
        probs  = torch.softmax(logits, dim=0).numpy()
    top_idx  = np.argsort(probs)[::-1][:top_k]
    return [(class_names[i], float(probs[i])) for i in top_idx]

def predict_demo(img: Image.Image, top_k=5):
    """
    Deterministic demo: hash the image pixels slightly so the same image
    always returns the same prediction, but different images differ.
    """
    info = load_disease_info()
    classes = list(info.keys())
    arr     = np.array(img.convert("RGB").resize((32, 32)))
    seed    = int(arr.mean() * 100) % len(classes)
    rng     = np.random.RandomState(seed)
    raw     = rng.dirichlet(np.ones(len(classes)) * 0.3)
    # Boost top prediction so it looks realistic
    top_idx = np.argsort(raw)[::-1][:top_k]
    raw[top_idx[0]] *= 6
    raw /= raw.sum()
    top_idx = np.argsort(raw)[::-1][:top_k]
    return [(classes[i], float(raw[i])) for i in top_idx]

def run_inference(img: Image.Image):
    model, extras = load_model()
    if model is not None:
        tfm, class_names = extras
        preds = predict_real(img, model, tfm, class_names)
    else:
        preds = predict_demo(img)
    return preds

# ─── Gemini description (optional) ───────────────────────────
def fetch_gemini_description(class_key: str, api_key: str) -> str | None:
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        m = genai.GenerativeModel("gemini-1.5-flash")
        prompt = (
            f"You are an agricultural expert writing for Indian farmers. "
            f"The crop disease is: '{class_key.replace('_', ' ')}'. "
            f"Write exactly 2 short, simple sentences in plain English (no jargon): "
            f"1) What the disease is and what damage it causes. "
            f"2) The single most important action the farmer should take today."
        )
        resp = m.generate_content(prompt)
        return resp.text.strip()
    except Exception:
        return None

# ─── UI helpers ───────────────────────────────────────────────
SEVERITY_COLOR = {
    "none":     ("#22c55e", "🟢"),
    "moderate": ("#f59e0b", "🟡"),
    "high":     ("#ef4444", "🔴"),
    "critical": ("#7c3aed", "🟣"),
}

def severity_badge(severity: str) -> str:
    color, emoji = SEVERITY_COLOR.get(severity, ("#6b7280", "⚪"))
    label = {"none": "HEALTHY", "moderate": "MODERATE",
             "high": "HIGH RISK", "critical": "CRITICAL"}.get(severity, severity.upper())
    return (f'<span style="background:{color};color:white;padding:3px 10px;'
            f'border-radius:20px;font-size:0.78rem;font-weight:700;'
            f'letter-spacing:0.05em">{emoji} {label}</span>')

# ─── Custom CSS ───────────────────────────────────────────────
CUSTOM_CSS = """
<style>
  /* Import fonts */
  @import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

  html, body, [class*="css"] { font-family: 'Sora', sans-serif; }

  /* Main background */
  .main .block-container { padding-top: 1.5rem; max-width: 1100px; }

  /* Hero banner */
  .hero-banner {
    background: linear-gradient(135deg, #064e3b 0%, #065f46 40%, #047857 100%);
    border-radius: 16px; padding: 28px 36px; margin-bottom: 24px;
    display: flex; align-items: center; gap: 20px;
    box-shadow: 0 10px 40px rgba(6, 78, 59, 0.4);
  }
  .hero-title { font-size: 2rem; font-weight: 800; color: #ecfdf5; margin: 0; line-height: 1.2; }
  .hero-sub   { font-size: 0.9rem; color: #a7f3d0; margin-top: 4px; }

  /* Cards */
  .result-card {
    background: white; border-radius: 14px; padding: 22px 26px;
    box-shadow: 0 2px 16px rgba(0,0,0,0.08); margin-bottom: 16px;
    border: 1px solid #e5e7eb;
  }
  .result-card.top-result { border-left: 5px solid #059669; }
  .result-card.alt-result { border-left: 5px solid #d1d5db; opacity: 0.85; }

  /* Confidence bar */
  .conf-bar-wrap { background: #f3f4f6; border-radius: 99px; height: 10px; margin: 8px 0 4px; overflow: hidden; }
  .conf-bar      { height: 100%; border-radius: 99px; transition: width 0.8s ease; }

  /* Action box */
  .action-box {
    background: #f0fdf4; border: 1px solid #86efac; border-radius: 10px;
    padding: 14px 18px; margin-top: 12px;
  }
  .action-box h4 { color: #166534; margin: 0 0 6px; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.06em; }
  .action-box p  { color: #15803d; margin: 0; font-size: 0.92rem; line-height: 1.5; }

  /* Demo banner */
  .demo-banner {
    background: linear-gradient(90deg, #fef3c7, #fde68a);
    border: 1px solid #fbbf24; border-radius: 10px;
    padding: 10px 18px; margin-bottom: 18px; font-size: 0.88rem; color: #92400e;
  }

  /* Info chip */
  .info-chip {
    display: inline-block; background: #ecfdf5; color: #065f46;
    border: 1px solid #6ee7b7; border-radius: 20px;
    padding: 3px 12px; font-size: 0.8rem; font-weight: 600; margin: 3px 2px;
  }

  /* Hide Streamlit default elements */
  footer { visibility: hidden; }
  #MainMenu { visibility: hidden; }

  /* Sample image grid */
  .sample-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: 10px; }

  /* Stats row */
  .stat-box { background:#f9fafb; border-radius:10px; padding:14px; text-align:center; border:1px solid #e5e7eb; }
  .stat-num { font-size:1.8rem; font-weight:800; color:#059669; }
  .stat-lbl { font-size:0.78rem; color:#6b7280; margin-top:2px; }

  code { font-family: 'JetBrains Mono', monospace; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    gemini_key = st.text_input(
        "Gemini API Key (optional)",
        type="password",
        help="Adds live AI-generated farmer tips. Leave blank to use bundled descriptions.",
        placeholder="AIza…"
    )

    st.divider()

    st.markdown("### 📊 Model Info")
    st.markdown("""
    <div class="stat-box" style="margin-bottom:8px">
      <div class="stat-num">38</div>
      <div class="stat-lbl">Disease Classes</div>
    </div>
    <div class="stat-box" style="margin-bottom:8px">
      <div class="stat-num">14</div>
      <div class="stat-lbl">Crop Types</div>
    </div>
    <div class="stat-box" style="margin-bottom:8px">
      <div class="stat-num">54K</div>
      <div class="stat-lbl">Training Images</div>
    </div>
    <div class="stat-box">
      <div class="stat-num">MV3</div>
      <div class="stat-lbl">MobileNetV3-Small</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("### 🌾 Supported Crops")
    crops = ["Apple", "Blueberry", "Cherry", "Corn", "Grape",
             "Orange", "Peach", "Bell Pepper", "Potato",
             "Raspberry", "Soybean", "Squash", "Strawberry", "Tomato"]
    for c in crops:
        st.markdown(f"<span class='info-chip'>{c}</span>", unsafe_allow_html=True)

    st.divider()
    st.markdown(
        "<div style='font-size:0.75rem;color:#9ca3af;text-align:center'>"
        "PlantVillage Dataset · Mohanty et al. 2016<br>"
        "T1.7 Unified Model · T1.8 PWA-Ready"
        "</div>",
        unsafe_allow_html=True,
    )

# ═══════════════════════════════════════════════════════════════
# MAIN CONTENT
# ═══════════════════════════════════════════════════════════════

# Hero
st.markdown("""
<div class="hero-banner">
  <div style="font-size:3rem">🌿</div>
  <div>
    <div class="hero-title">CropGuard AI</div>
    <div class="hero-sub">
      Photograph a sick leaf → Instant disease identification → Farmer-friendly action plan
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# Demo mode notice
if DEMO_MODE:
    st.markdown("""
    <div class="demo-banner">
      🧪 <b>Demo Mode</b> — No trained model found at <code>best_model.pth</code>.
      Predictions are <b>illustrative</b>. Train the model using the Kaggle script and
      place <code>best_model.pth</code> in this folder to enable real inference.
    </div>
    """, unsafe_allow_html=True)

# ── Upload section ────────────────────────────────────────────
col_upload, col_result = st.columns([1, 1.4], gap="large")

with col_upload:
    st.markdown("### 📸 Upload Leaf Image")

    uploaded = st.file_uploader(
        "Drag & drop or click to upload",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

    # Sample images
    st.markdown("**Or try a sample:**")
    sample_dir = ROOT / "sample_images"
    samples = list(sample_dir.glob("*.jpg")) + list(sample_dir.glob("*.png")) if sample_dir.exists() else []

    if samples:
        scols = st.columns(min(3, len(samples)))
        for i, sp in enumerate(samples[:6]):
            with scols[i % 3]:
                thumb = Image.open(sp).resize((80, 80))
                if st.button(sp.stem.replace("_", " ")[:12], key=f"smp_{i}",
                             use_container_width=True):
                    st.session_state["sample_path"] = str(sp)
                st.image(thumb, use_container_width=True)
    else:
        st.info("📁 Add sample leaf images to `sample_images/` folder to enable quick demos.", icon="💡")

    # Resolve image source
    img_pil = None
    source_label = ""
    if uploaded:
        img_pil = Image.open(uploaded).convert("RGB")
        source_label = uploaded.name
        st.session_state.pop("sample_path", None)
    elif "sample_path" in st.session_state:
        try:
            img_pil = Image.open(st.session_state["sample_path"]).convert("RGB")
            source_label = Path(st.session_state["sample_path"]).name
        except Exception:
            pass

    if img_pil:
        st.image(img_pil, caption=source_label, use_container_width=True)

# ── Results section ───────────────────────────────────────────
with col_result:
    st.markdown("### 🔬 Analysis Results")

    if img_pil is None:
        st.markdown("""
        <div style="
          border: 2px dashed #d1fae5; border-radius: 14px;
          padding: 48px 24px; text-align: center; color: #6b7280;
          background: #f9fafb;">
          <div style="font-size:2.5rem;margin-bottom:12px">🌱</div>
          <div style="font-weight:600;font-size:1rem;margin-bottom:6px">Waiting for a leaf image…</div>
          <div style="font-size:0.85rem">Upload a photo of a crop leaf to get instant disease detection and farming advice.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Run inference
        with st.spinner("🔍 Analysing leaf…"):
            t0    = time.time()
            preds = run_inference(img_pil)
            dt    = time.time() - t0

        disease_info = load_disease_info()

        # ── Top prediction ──
        top_class, top_conf = preds[0]
        info, matched_key = lookup_disease(top_class, disease_info)
        display_name = info.get("display_name", top_class.replace("_", " "))
        severity     = info.get("severity", "moderate")
        description  = info.get("description", "")
        action       = info.get("action", "")
        emoji        = info.get("emoji", "🌿")

        # ── Debug panel (helps diagnose key mismatches) ──
        with st.expander("🛠 Debug — raw model output (click to inspect)", expanded=False):
            st.markdown("**Raw class names from model → matched JSON key:**")
            for rc, rc_conf in preds:
                _, mk = lookup_disease(rc, disease_info)
                match_ok = "✅" if mk != rc or rc in disease_info else "⚠️ fuzzy"
                st.code(f"{rc_conf*100:5.1f}%  |  model: {rc!r}\n              mapped: {mk!r}  {match_ok}")

        sev_color = SEVERITY_COLOR.get(severity, ("#6b7280", "⚪"))[0]
        badge_html = severity_badge(severity)

        # Optionally fetch Gemini description
        if gemini_key and gemini_key.strip():
            live_desc = fetch_gemini_description(top_class, gemini_key.strip())
            if live_desc:
                description = live_desc

        conf_pct  = top_conf * 100
        bar_color = sev_color

        st.markdown(f"""
        <div class="result-card top-result">
          <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:4px">
            <div style="font-size:1.25rem;font-weight:800;color:#111827">
              {emoji} {display_name}
            </div>
            {badge_html}
          </div>
          <div style="font-size:0.82rem;color:#6b7280;margin-bottom:8px">
            Model confidence
          </div>
          <div class="conf-bar-wrap">
            <div class="conf-bar" style="width:{conf_pct:.1f}%;background:{bar_color}"></div>
          </div>
          <div style="font-size:1rem;font-weight:700;color:{bar_color}">{conf_pct:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

        # Description + Action
        if description:
            st.markdown(
                f"<p style='color:#374151;line-height:1.65;font-size:0.93rem'>{description}</p>",
                unsafe_allow_html=True)

        if action:
            st.markdown(f"""
            <div class="action-box">
              <h4>✅ Recommended Action</h4>
              <p>{action}</p>
            </div>
            """, unsafe_allow_html=True)

        # Meta row
        st.markdown(
            f"<div style='margin-top:10px;font-size:0.78rem;color:#9ca3af'>"
            f"⏱ Inference: {dt*1000:.0f} ms &nbsp;|&nbsp; "
            f"{'🧪 Demo Mode' if DEMO_MODE else '🤖 Live Model'} &nbsp;|&nbsp; "
            f"PlantVillage 38-class</div>",
            unsafe_allow_html=True)

        # ── Alternative predictions ──
        if len(preds) > 1:
            with st.expander("🔢 Top-5 alternative predictions"):
                for rank, (cls, conf) in enumerate(preds[1:], 2):
                    alt_info, _ = lookup_disease(cls, disease_info)
                    alt_name = alt_info.get("display_name", cls.replace("_", " "))
                    alt_emoji = alt_info.get("emoji", "🌿")
                    alt_sev   = alt_info.get("severity", "moderate")
                    alt_color = SEVERITY_COLOR.get(alt_sev, ("#6b7280", "⚪"))[0]
                    alt_pct   = conf * 100
                    st.markdown(f"""
                    <div class="result-card alt-result" style="padding:14px 18px;margin-bottom:8px">
                      <div style="display:flex;justify-content:space-between;align-items:center">
                        <span style="font-weight:600;color:#374151">{rank}. {alt_emoji} {alt_name}</span>
                        <span style="font-weight:700;color:{alt_color}">{alt_pct:.1f}%</span>
                      </div>
                      <div class="conf-bar-wrap">
                        <div class="conf-bar" style="width:{alt_pct:.1f}%;background:{alt_color}"></div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

# ── About section ─────────────────────────────────────────────
with st.expander("ℹ️ About this app · How it works · Technical details"):
    st.markdown("""
    ### How CropGuard AI Works

    **T1.7 — Multi-Crop Unified Model**
    1. A MobileNetV3-Small backbone (pre-trained on ImageNet) is fine-tuned on the full
       [PlantVillage dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
       — 54,303 images across **38 disease classes** spanning 14 crops.
    2. Training uses a two-stage regime: the backbone is frozen for epoch 1 (head-only),
       then unfrozen for epochs 2–3 with a lower learning rate.
    3. Data augmentation (random crop, flip, colour jitter, rotation) makes the model
       robust to real-world mobile phone photography conditions.

    **T1.8 — Mobile / PWA Ready**
    - The Kaggle training script exports the model to **ONNX → TFLite** via `onnx2tf`.
    - The Streamlit app uses a `manifest.json` + service worker for full PWA install on mobile.
    - MobileNetV3-Small keeps the TFLite model under **~5 MB** for offline-capable deployment.

    **Disease descriptions** are pre-generated, farmer-friendly explanations bundled as
    `disease_info.json`. If you provide a Gemini API key, descriptions are generated live
    for richer, context-aware advice.

    **Dataset citation:** Mohanty, S.P., Hughes, D., and Salathé, M. (2016).
    *Using deep learning for image-based plant disease detection.* Frontiers in Plant Science.
    """)

# ── PWA manifest injection ────────────────────────────────────
st.markdown("""
<link rel="manifest" href="/app/static/manifest.json">
<meta name="theme-color" content="#064e3b">
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-title" content="CropGuard AI">
""", unsafe_allow_html=True)
