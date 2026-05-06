"""
╔══════════════════════════════════════════════════════════════════╗
║   🌿  CropGuard AI — Multi-Crop Disease Detection               ║
║   T1.7 (38-class unified model) + T1.8 (PWA-ready)             ║
║                                                                  ║
║   Original assignment requirements fully preserved:             ║
║   ✅ Upload leaf photo → disease prediction                     ║
║   ✅ Confidence bar per class                                   ║
║   ✅ Farmer-friendly description + recommended action           ║
║   ✅ 38-class PlantVillage model (MobileNetV3-Small)            ║
║   ✅ PWA manifest for mobile install (T1.8)                     ║
║                                                                  ║
║   Extended farmer-facing features:                               ║
║   ✅ Field Diary — scan history with thumbnails                 ║
║   ✅ Batch Scan — multi-leaf field-level verdict                ║
║   ✅ Weather Risk — live spread risk (Open-Meteo, no key)       ║
║   ✅ Treatment Guide — chemicals, INR costs, where to buy       ║
║   ✅ GradCAM — visual explanation shown inline to farmer        ║
║   ✅ Report Export — shareable PNG card                         ║
║   ✅ Language — English / Hindi / Telugu / Marathi              ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os, json, time, re
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image
import streamlit as st

# ─── Page config (MUST be first Streamlit call) ───────────────────
st.set_page_config(
    page_title="CropGuard AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Constants ────────────────────────────────────────────────────
ROOT              = Path(__file__).parent
DISEASE_INFO_PATH = ROOT / "disease_info.json"
CLASS_NAMES_PATH  = ROOT / "class_names.json"
MODEL_PATH        = ROOT / "best_model.pth"
IMG_SIZE          = 224
DEMO_MODE         = not MODEL_PATH.exists()

# ─── Farmer Tools (graceful import — app works without this file) ─
try:
    from farmer_tools import (
        FieldDiary, get_weather, assess_spread_risk,
        analyse_batch, generate_report_image, export_report_bytes,
        get_treatment, t, LANGUAGE_OPTIONS,
    )
    FARMER_TOOLS_OK = True
except ImportError:
    FARMER_TOOLS_OK = False
    def t(key, lang="en"): return key
    LANGUAGE_OPTIONS = {"English": "en"}

# ─── Session state init ───────────────────────────────────────────
for k, v in [("lang","en"), ("active_tab","scan"), ("sample_path", None)]:
    if k not in st.session_state:
        st.session_state[k] = v
if "diary" not in st.session_state and FARMER_TOOLS_OK:
    st.session_state["diary"] = FieldDiary()

lang = st.session_state["lang"]

# ══════════════════════════════════════════════════════════════════
# RESOURCE LOADERS
# ══════════════════════════════════════════════════════════════════
@st.cache_resource
def load_disease_info():
    with open(DISEASE_INFO_PATH, encoding="utf-8") as f:
        return json.load(f)

@st.cache_resource
def load_class_names():
    if CLASS_NAMES_PATH.exists():
        with open(CLASS_NAMES_PATH, encoding="utf-8") as f:
            return json.load(f)
    return list(load_disease_info().keys())

def lookup_disease(raw_class: str, disease_info: dict) -> tuple:
    if raw_class in disease_info:
        return disease_info[raw_class], raw_class
    def norm(s):
        return re.sub(r"[\s_]+", " ", s).lower().strip()
    raw_norm = norm(raw_class)
    for key in disease_info:
        if norm(key) == raw_norm:
            return disease_info[key], key
    for key in disease_info:
        k = norm(key)
        if raw_norm in k or k in raw_norm:
            return disease_info[key], key
    raw_tokens = set(raw_norm.split())
    best_key, best_score = None, 0
    for key in disease_info:
        score = len(raw_tokens & set(norm(key).split()))
        if score > best_score:
            best_score, best_key = score, key
    if best_key and best_score > 0:
        return disease_info[best_key], best_key
    return {}, raw_class

@st.cache_resource(show_spinner="Loading AI model…")
def load_model():
    if DEMO_MODE:
        return None, None
    try:
        import torch, timm
        from timm.data import resolve_data_config
        from timm.data.transforms_factory import create_transform
        ckpt        = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        class_names = ckpt.get("class_names", load_class_names())
        model       = timm.create_model("mobilenetv3_small_100",
                                        pretrained=False, num_classes=len(class_names))
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        data_cfg               = resolve_data_config({}, model=model)
        data_cfg["input_size"] = (3, 224, 224)
        data_cfg["mean"]       = (0.485, 0.456, 0.406)
        data_cfg["std"]        = (0.229, 0.224, 0.225)
        tfm = create_transform(**data_cfg, is_training=False)
        return model, (tfm, class_names)
    except Exception as e:
        st.warning(f"Model load error: {e}")
        import traceback; st.code(traceback.format_exc())
        return None, None

# ══════════════════════════════════════════════════════════════════
# INFERENCE
# ══════════════════════════════════════════════════════════════════
def predict_real(img, model, tfm, class_names, top_k=5):
    import torch
    x = tfm(img.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)[0]
        probs  = torch.softmax(logits, dim=0).numpy()
    top_idx = np.argsort(probs)[::-1][:top_k]
    st.session_state["_dbg_logit_max"] = float(logits.max())
    st.session_state["_dbg_logit_min"] = float(logits.min())
    st.session_state["_dbg_top1_prob"] = float(probs[top_idx[0]])
    st.session_state["_last_model"]    = model
    st.session_state["_last_timm"]     = "mobilenetv3_small_100"
    return [(class_names[i], float(probs[i])) for i in top_idx]

def predict_demo(img, top_k=5):
    info    = load_disease_info()
    classes = list(info.keys())
    arr     = np.array(img.convert("RGB").resize((32, 32)))
    seed    = int(arr.mean() * 100) % len(classes)
    rng     = np.random.RandomState(seed)
    raw     = rng.dirichlet(np.ones(len(classes)) * 0.3)
    top_idx = np.argsort(raw)[::-1][:top_k]
    raw[top_idx[0]] *= 6
    raw /= raw.sum()
    top_idx = np.argsort(raw)[::-1][:top_k]
    return [(classes[i], float(raw[i])) for i in top_idx]

def run_inference(img):
    model, extras = load_model()
    if model is not None:
        tfm, class_names = extras
        return predict_real(img, model, tfm, class_names)
    return predict_demo(img)

# ── GradCAM inline ────────────────────────────────────────────────
def get_gradcam_overlay(img_pil, class_idx=None):
    try:
        from gradcam import explain_image
        model = st.session_state.get("_last_model")
        tname = st.session_state.get("_last_timm", "mobilenetv3_small_100")
        if model is None:
            return None
        overlay, _ = explain_image(img_pil, model, tname,
                                   class_idx=class_idx, img_size=224, alpha=0.5)
        return overlay
    except Exception:
        return None

# ══════════════════════════════════════════════════════════════════
# UI HELPERS
# ══════════════════════════════════════════════════════════════════
SEVERITY_COLOR = {
    "none":     ("#22c55e", "🟢"),
    "moderate": ("#f59e0b", "🟡"),
    "high":     ("#ef4444", "🔴"),
    "critical": ("#7c3aed", "🟣"),
}

def severity_badge(severity, lang="en"):
    color, emoji = SEVERITY_COLOR.get(severity, ("#6b7280","⚪"))
    labels = {"none": t("healthy",lang), "moderate": t("moderate",lang),
              "high": t("high_risk",lang), "critical": t("critical",lang)}
    label  = labels.get(severity, severity.upper())
    return (f'<span style="background:{color};color:white;padding:3px 10px;'
            f'border-radius:20px;font-size:0.78rem;font-weight:700;'
            f'letter-spacing:0.04em">{emoji} {label}</span>')

def conf_bar_html(pct, color):
    return (f'<div style="background:#f3f4f6;border-radius:99px;height:10px;'
            f'margin:6px 0 2px;overflow:hidden">'
            f'<div style="width:{pct:.1f}%;height:100%;border-radius:99px;'
            f'background:{color};transition:width 0.6s ease"></div></div>'
            f'<div style="font-size:1rem;font-weight:700;color:{color}">{pct:.1f}%</div>')

# ══════════════════════════════════════════════════════════════════
# CUSTOM CSS
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
html,body,[class*="css"]{font-family:'Sora',sans-serif;}
.main .block-container{padding-top:1.2rem;max-width:1200px;}
.hero-banner{
  background:linear-gradient(135deg,#064e3b 0%,#065f46 40%,#047857 100%);
  border-radius:16px;padding:24px 32px;margin-bottom:20px;
  box-shadow:0 8px 32px rgba(6,78,59,0.35);display:flex;align-items:center;gap:16px;
}
.hero-title{font-size:1.9rem;font-weight:800;color:#ecfdf5;margin:0;line-height:1.2;}
.hero-sub{font-size:0.88rem;color:#a7f3d0;margin-top:4px;}
.result-card{background:white;border-radius:14px;padding:20px 24px;
  box-shadow:0 2px 14px rgba(0,0,0,0.07);margin-bottom:14px;border:1px solid #e5e7eb;}
.result-card.top-result{border-left:5px solid #059669;}
.result-card.alt-result{border-left:5px solid #d1d5db;opacity:0.82;}
.action-box{background:#f0fdf4;border:1px solid #86efac;border-radius:10px;
  padding:14px 18px;margin-top:10px;}
.action-box h4{color:#166534;margin:0 0 5px;font-size:0.82rem;text-transform:uppercase;letter-spacing:0.06em;}
.action-box p{color:#15803d;margin:0;font-size:0.9rem;line-height:1.55;}
.weather-box{background:#fefce8;border:1px solid #fde047;border-radius:10px;padding:12px 16px;margin-top:10px;}
.demo-banner{background:linear-gradient(90deg,#fef3c7,#fde68a);border:1px solid #fbbf24;
  border-radius:10px;padding:10px 18px;margin-bottom:16px;font-size:0.88rem;color:#92400e;}
.info-chip{display:inline-block;background:#ecfdf5;color:#065f46;border:1px solid #6ee7b7;
  border-radius:20px;padding:3px 12px;font-size:0.78rem;font-weight:600;margin:2px;}
footer{visibility:hidden;}
#MainMenu{visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🌐 Language / भाषा")
    sel_lang_label = st.selectbox("Language", list(LANGUAGE_OPTIONS.keys()),
                                  label_visibility="collapsed")
    st.session_state["lang"] = LANGUAGE_OPTIONS[sel_lang_label]
    lang = st.session_state["lang"]

    st.divider()
    st.markdown("## 👨‍🌾 Farmer Profile")
    farmer_name   = st.text_input(t("farmer_name",lang), key="farmer_name",
                                  placeholder="e.g. Ramesh Kumar")
    field_name    = st.text_input(t("field_name",lang), key="field_name",
                                  placeholder="e.g. North Field, Plot B")
    location_name = st.text_input("Location / Village", key="location_name",
                                  placeholder="e.g. Warangal, Telangana")

    st.divider()
    st.markdown("## 🤖 AI Descriptions")
    gemini_key = st.text_input("Gemini API Key (optional)", type="password",
                               placeholder="AIza…")

    st.divider()
    st.markdown("## 🌦 Weather Risk")
    use_weather = st.checkbox("Enable weather-based spread risk", value=False)
    w_lat = w_lon = None
    if use_weather:
        w_lat = st.number_input("Latitude",  value=17.38, format="%.4f")
        w_lon = st.number_input("Longitude", value=78.49, format="%.4f")
        st.caption("Default: Hyderabad. Open-Meteo — free, no API key.")

    st.divider()
    st.markdown("## 📊 Model")
    st.markdown("""
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-bottom:8px">
      <div style="background:#f9fafb;border-radius:8px;padding:10px;text-align:center;border:1px solid #e5e7eb">
        <div style="font-size:1.4rem;font-weight:800;color:#059669">38</div>
        <div style="font-size:0.72rem;color:#6b7280">Classes</div></div>
      <div style="background:#f9fafb;border-radius:8px;padding:10px;text-align:center;border:1px solid #e5e7eb">
        <div style="font-size:1.4rem;font-weight:800;color:#059669">14</div>
        <div style="font-size:0.72rem;color:#6b7280">Crops</div></div>
      <div style="background:#f9fafb;border-radius:8px;padding:10px;text-align:center;border:1px solid #e5e7eb">
        <div style="font-size:1.4rem;font-weight:800;color:#059669">54K</div>
        <div style="font-size:0.72rem;color:#6b7280">Images</div></div>
      <div style="background:#f9fafb;border-radius:8px;padding:10px;text-align:center;border:1px solid #e5e7eb">
        <div style="font-size:1.4rem;font-weight:800;color:#059669">MV3</div>
        <div style="font-size:0.72rem;color:#6b7280">Backbone</div></div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("### 🌾 Crops")
    for c in ["Apple","Blueberry","Cherry","Corn","Grape","Orange","Peach",
              "Bell Pepper","Potato","Raspberry","Soybean","Squash","Strawberry","Tomato"]:
        st.markdown(f"<span class='info-chip'>{c}</span>", unsafe_allow_html=True)
    st.divider()
    st.markdown(f"""
    <div style='font-size:0.74rem;color:#9ca3af;text-align:center'>
      PlantVillage · Mohanty et al. 2016<br>T1.7 Unified · T1.8 PWA-Ready<br><br>
      📞 {t('kvk_helpline', lang)}
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="hero-banner">
  <div style="font-size:2.8rem">🌿</div>
  <div>
    <div class="hero-title">CropGuard AI</div>
    <div class="hero-sub">
      {t('upload_prompt', lang)} → Instant disease ID → Farmer-friendly action
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

if DEMO_MODE:
    st.markdown("""
    <div class="demo-banner">
      🧪 <b>Demo Mode</b> — No trained model found (<code>best_model.pth</code>).
      Predictions are illustrative. Run <code>kaggle_train.py</code> on Kaggle GPU to train.
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════
tab_scan, tab_batch, tab_diary, tab_about = st.tabs([
    "🔍 Scan Leaf",
    f"🌾 {t('field_scan', lang)}",
    f"📔 {t('scan_history', lang)}",
    "ℹ️ About",
])

# ══════════════════════════════════════════════════════════════════
# TAB 1 — SINGLE LEAF SCAN  (core assignment requirement)
# ══════════════════════════════════════════════════════════════════
with tab_scan:
    col_upload, col_result = st.columns([1, 1.45], gap="large")

    # ── Upload + sample selector ──────────────────────────────────
    with col_upload:
        st.markdown(f"### 📸 Upload Leaf")
        uploaded = st.file_uploader("Upload leaf", type=["jpg","jpeg","png","webp"],
                                    label_visibility="collapsed")

        # Sample grid
        sample_dir  = ROOT / "sample_images"
        all_samples = sorted(
            list(sample_dir.glob("*.jpg")) +
            list(sample_dir.glob("*.jpeg")) +
            list(sample_dir.glob("*.png"))
        ) if sample_dir.exists() else []

        if all_samples:
            grouped = defaultdict(list)
            for sp in all_samples:
                crop = sp.stem.split("__")[0].replace("_"," ").split("(")[0].strip()
                grouped[crop].append(sp)
            crop_opts = sorted(grouped.keys())
            sel_crop  = st.selectbox("Filter by crop",
                                     ["All crops"] + crop_opts,
                                     key="crop_filter",
                                     label_visibility="collapsed")
            visible   = all_samples if sel_crop == "All crops" else grouped[sel_crop]
            visible   = visible[:12]
            for row_start in range(0, len(visible), 3):
                row = visible[row_start:row_start+3]
                cols = st.columns(3)
                for col, sp in zip(cols, row):
                    with col:
                        parts = sp.stem.split("__")
                        label = (parts[1].replace("_"," ").title()
                                 .rstrip("0123456789").strip()
                                 if len(parts) >= 2
                                 else sp.stem.replace("_"," "))[:16]
                        try:
                            st.image(Image.open(sp).convert("RGB").resize((90,90)),
                                     use_container_width=True)
                        except Exception:
                            pass
                        if st.button(label, key=f"smp_{sp.stem}",
                                     use_container_width=True):
                            st.session_state["sample_path"] = str(sp)
                            st.rerun()
        else:
            st.info("📁 Run `python setup_samples.py` then restart.", icon="💡")

        # Resolve active image
        img_pil      = None
        source_label = ""
        if uploaded:
            img_pil = Image.open(uploaded).convert("RGB")
            source_label = uploaded.name
            st.session_state["sample_path"] = None
        elif st.session_state.get("sample_path"):
            try:
                img_pil      = Image.open(st.session_state["sample_path"]).convert("RGB")
                source_label = Path(st.session_state["sample_path"]).name
            except Exception:
                pass

        if img_pil:
            st.image(img_pil, caption=source_label, use_container_width=True)

    # ── Results ───────────────────────────────────────────────────
    with col_result:
        st.markdown("### 🔬 Analysis Results")

        if img_pil is None:
            st.markdown("""
            <div style="border:2px dashed #d1fae5;border-radius:14px;
              padding:48px 24px;text-align:center;color:#6b7280;background:#f9fafb;">
              <div style="font-size:2.5rem;margin-bottom:10px">🌱</div>
              <div style="font-weight:600;font-size:1rem;margin-bottom:6px">
                Waiting for a leaf image…</div>
              <div style="font-size:0.85rem">
                Upload a leaf photo or pick a sample to get instant disease detection.</div>
            </div>""", unsafe_allow_html=True)
        else:
            # ── INFERENCE (assignment core) ────────────────────────
            with st.spinner(f"🔍 {t('analysing', lang)}"):
                t0    = time.time()
                preds = run_inference(img_pil)
                dt    = time.time() - t0

            disease_info = load_disease_info()
            top_class, top_conf = preds[0]
            info, matched_key   = lookup_disease(top_class, disease_info)
            display_name = info.get("display_name", top_class.replace("_"," "))
            severity     = info.get("severity", "moderate")
            description  = info.get("description", "")
            action       = info.get("action", "")
            emoji        = info.get("emoji", "🌿")
            crop         = info.get("crop", "Unknown")
            disease      = info.get("disease", top_class)

            # Optional Gemini live description
            if gemini_key and gemini_key.strip():
                try:
                    import google.generativeai as genai
                    genai.configure(api_key=gemini_key.strip())
                    m    = genai.GenerativeModel("gemini-1.5-flash")
                    resp = m.generate_content(
                        f"You are an expert writing for Indian farmers. "
                        f"Disease: '{top_class.replace('_',' ')}'. "
                        f"Write 2 short simple sentences: "
                        f"1) what it is and damage caused. "
                        f"2) most important action today.")
                    description = resp.text.strip()
                except Exception:
                    pass

            sev_color  = SEVERITY_COLOR.get(severity, ("#6b7280","⚪"))[0]
            badge_html = severity_badge(severity, lang)
            conf_pct   = top_conf * 100

            # ── PRIMARY RESULT CARD (assignment requirement) ───────
            st.markdown(f"""
            <div class="result-card top-result">
              <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:4px">
                <div style="font-size:1.2rem;font-weight:800;color:#111827">
                  {emoji} {display_name}</div>
                {badge_html}
              </div>
              <div style="font-size:0.8rem;color:#6b7280;margin-bottom:4px">
                {t('confidence', lang)}</div>
              {conf_bar_html(conf_pct, sev_color)}
            </div>""", unsafe_allow_html=True)

            # ── DESCRIPTION (assignment requirement) ───────────────
            if description:
                st.markdown(
                    f"<p style='color:#374151;line-height:1.65;font-size:0.92rem'>"
                    f"{description}</p>", unsafe_allow_html=True)

            # ── ACTION (assignment requirement) ────────────────────
            if action:
                st.markdown(f"""
                <div class="action-box">
                  <h4>✅ {t('recommended_action', lang)}</h4>
                  <p>{action}</p>
                </div>""", unsafe_allow_html=True)

            # ── TREATMENT COST (new) ───────────────────────────────
            treatment = get_treatment(disease) if FARMER_TOOLS_OK else None
            if treatment:
                with st.expander(f"💊 {t('treatment_cost', lang)} — ₹ Costs & Chemicals"):
                    for ch in treatment.get("chemicals", []):
                        cost = ch.get("cost_per_kg",
                               ch.get("cost_per_litre",
                               ch.get("cost_per_g","")))
                        st.markdown(
                            f"• **{ch['name']}** — {ch['dose']} | "
                            f"Cost: {cost} | Uses: {ch.get('applications','-')}×")
                    if treatment.get("organic"):
                        st.markdown(f"🌿 **Organic option:** {treatment['organic']}")
                    if treatment.get("cost_estimate_per_acre"):
                        st.info(f"💰 Estimated total cost per acre: "
                                f"**{treatment['cost_estimate_per_acre']}**")
                    if treatment.get("where_to_buy"):
                        st.markdown(f"🏪 **Where to buy:** {treatment['where_to_buy']}")
                    if treatment.get("critical_timing"):
                        st.warning(f"⏰ **Timing:** {treatment['critical_timing']}")

            # ── WEATHER RISK (new) ────────────────────────────────
            if use_weather and FARMER_TOOLS_OK and w_lat and w_lon:
                with st.spinner("🌦 Checking today's spread risk…"):
                    weather = get_weather(w_lat, w_lon)
                risk = assess_spread_risk(top_class, weather)
                rc   = {"low":"#22c55e","moderate":"#f59e0b",
                        "high":"#ef4444","critical":"#7c3aed"}.get(
                        risk["risk_level"], "#6b7280")
                st.markdown(f"""
                <div class="weather-box">
                  <b>🌦 {t('weather_risk', lang)}:
                  <span style="color:{rc}">
                    {risk['risk_level'].upper()} ({risk['risk_score']}/100)
                  </span></b><br>
                  <span style="font-size:0.87rem">{risk['reason'][:220]}</span>
                </div>""", unsafe_allow_html=True)
                spray_msg = (f"⚠️ **{t('spray_today',lang)}**"
                             if risk["spray_today"]
                             else f"✅ {t('no_spray_needed',lang)}")
                st.markdown(spray_msg)
                if weather:
                    st.caption(
                        f"🌡 {weather['temp_c']}°C  "
                        f"💧 {weather['humidity']}% RH  "
                        f"🌧 {weather['precip_mm']}mm")

            # ── GRADCAM EXPLANATION (new — shown to farmer) ───────
            if not DEMO_MODE:
                with st.expander(f"🔥 {t('gradcam_title', lang)}"):
                    with st.spinner("Generating visual explanation…"):
                        overlay = get_gradcam_overlay(img_pil)
                    if overlay:
                        gc1, gc2 = st.columns(2)
                        with gc1:
                            st.image(img_pil.resize((224,224)),
                                     caption="Original leaf",
                                     use_container_width=True)
                        with gc2:
                            st.image(overlay,
                                     caption="🔴 Red = disease focus area",
                                     use_container_width=True)
                        st.caption(t("gradcam_desc", lang))
                    else:
                        st.info("Place gradcam.py in the project folder to enable this.")

            # ── TOP-5 ALTERNATIVES (assignment requirement) ────────
            if len(preds) > 1:
                with st.expander("🔢 Top-5 alternative predictions"):
                    for rank, (cls, conf) in enumerate(preds[1:], 2):
                        ai, _ = lookup_disease(cls, disease_info)
                        an    = ai.get("display_name", cls.replace("_"," "))
                        ae    = ai.get("emoji","🌿")
                        asev  = ai.get("severity","moderate")
                        ac    = SEVERITY_COLOR.get(asev,("#6b7280","⚪"))[0]
                        ap    = conf * 100
                        st.markdown(f"""
                        <div class="result-card alt-result"
                          style="padding:12px 16px;margin-bottom:6px">
                          <div style="display:flex;justify-content:space-between;align-items:center">
                            <span style="font-weight:600;color:#374151">
                              {rank}. {ae} {an}</span>
                            <span style="font-weight:700;color:{ac}">{ap:.1f}%</span>
                          </div>
                          {conf_bar_html(ap, ac)}
                        </div>""", unsafe_allow_html=True)

            # ── META ROW ──────────────────────────────────────────
            st.markdown(
                f"<div style='margin-top:8px;font-size:0.76rem;color:#9ca3af'>"
                f"⏱ {dt*1000:.0f} ms &nbsp;|&nbsp; "
                f"{'🧪 Demo' if DEMO_MODE else '🤖 Live model'} &nbsp;|&nbsp; "
                f"PlantVillage 38-class MobileNetV3</div>",
                unsafe_allow_html=True)

            # ── DEBUG PANEL ───────────────────────────────────────
            with st.expander("🛠 Debug", expanded=False):
                lmax = st.session_state.get("_dbg_logit_max")
                lmin = st.session_state.get("_dbg_logit_min")
                t1p  = st.session_state.get("_dbg_top1_prob")
                if lmax is not None:
                    ok = (lmax - lmin) > 3.0 and t1p > 0.5
                    st.markdown(f"Logit range: {lmin:.2f}→{lmax:.2f} | "
                                f"Top-1: {t1p*100:.1f}% | "
                                f"{'✅ Healthy output' if ok else '⚠️ Low confidence'}")
                for rc, rc_conf in preds:
                    _, mk = lookup_disease(rc, disease_info)
                    flag  = "✅" if rc in disease_info else "⚠️ fuzzy"
                    st.code(f"{rc_conf*100:5.1f}%  model:{rc!r}\n"
                            f"          mapped:{mk!r} {flag}")

            # ── SAVE + DOWNLOAD (new) ─────────────────────────────
            st.divider()
            c_save, c_dl = st.columns(2)
            with c_save:
                notes_input = st.text_input(
                    "Notes", key="scan_notes",
                    label_visibility="collapsed",
                    placeholder=t("notes_placeholder", lang))
                if st.button(f"💾 {t('save_scan', lang)}", use_container_width=True):
                    if FARMER_TOOLS_OK:
                        diary = st.session_state.get("diary", FieldDiary())
                        diary.save_scan(
                            crop=crop, disease=disease,
                            display_name=display_name,
                            confidence=top_conf, severity=severity,
                            action=action, img_pil=img_pil,
                            notes=notes_input,
                            lat=w_lat, lon=w_lon,
                            location_name=st.session_state.get("location_name",""))
                        st.success("✅ Saved to Field Diary!")
                    else:
                        st.error("farmer_tools.py not found.")

            with c_dl:
                if st.button(f"📄 {t('download_report', lang)}", use_container_width=True):
                    if FARMER_TOOLS_OK:
                        wr = None
                        if use_weather and w_lat:
                            wr = assess_spread_risk(top_class, get_weather(w_lat, w_lon))
                        rpt = generate_report_image(
                            leaf_img=img_pil, display_name=display_name,
                            confidence=top_conf, severity=severity,
                            description=description, action=action, crop=crop,
                            farmer_name=st.session_state.get("farmer_name",""),
                            field_name=st.session_state.get("field_name",""),
                            location=st.session_state.get("location_name",""),
                            weather_risk=wr)
                        st.download_button(
                            "⬇️ Download PNG",
                            export_report_bytes(rpt),
                            file_name=f"cropguard_{disease.replace(' ','_')}.png",
                            mime="image/png",
                            use_container_width=True)
                    else:
                        st.error("farmer_tools.py not found.")

# ══════════════════════════════════════════════════════════════════
# TAB 2 — BATCH / FIELD SCAN
# ══════════════════════════════════════════════════════════════════
with tab_batch:
    st.markdown(f"### 🌾 {t('field_scan', lang)}")
    st.markdown(
        "Upload photos of **multiple leaves from different spots in your field**. "
        "CropGuard will give you an overall field infection rate and urgency verdict.")

    batch_files = st.file_uploader(
        "Upload 2–10 leaf images",
        type=["jpg","jpeg","png","webp"],
        accept_multiple_files=True,
        label_visibility="collapsed")

    if batch_files and len(batch_files) >= 2:
        if st.button("🔍 Analyse Field", type="primary", use_container_width=True):
            with st.spinner(f"Analysing {len(batch_files)} leaves…"):
                batch_imgs = [Image.open(f).convert("RGB") for f in batch_files]
                if FARMER_TOOLS_OK:
                    result = analyse_batch(
                        batch_imgs, run_inference,
                        load_disease_info(), lookup_disease)
                else:
                    result = None
                    st.error("farmer_tools.py not found.")

            if result:
                uc = {"none":"#22c55e","moderate":"#f59e0b",
                      "high":"#ef4444","critical":"#7c3aed"}.get(result["urgency"],"#6b7280")
                # Field verdict
                st.markdown(f"""
                <div style="background:{uc};color:white;border-radius:12px;
                  padding:18px 24px;margin:12px 0;text-align:center">
                  <div style="font-size:1.4rem;font-weight:800">{result['field_verdict']}</div>
                  <div style="font-size:0.88rem;margin-top:6px;opacity:0.9">
                    {t('infection_rate',lang)}: <b>{result['infection_pct']}%</b>
                    of {result['n_images']} leaves sampled
                  </div>
                </div>""", unsafe_allow_html=True)

                st.markdown(f"""
                <div class="action-box">
                  <h4>✅ Field-Level Action</h4>
                  <p>{result['field_action']}</p>
                </div>""", unsafe_allow_html=True)

                if result["disease_counts"]:
                    st.markdown("**Disease breakdown:**")
                    for d, cnt in sorted(result["disease_counts"].items(),
                                         key=lambda x: x[1], reverse=True):
                        pct = cnt / result["n_images"] * 100
                        st.markdown(f"• **{d}**: {cnt}/{result['n_images']} leaves ({pct:.0f}%)")

                # Thumbnail grid
                st.markdown("**Per-leaf results:**")
                icols = st.columns(min(5, len(batch_imgs)))
                for i, (img, res) in enumerate(zip(batch_imgs, result["per_image"])):
                    with icols[i % len(icols)]:
                        st.image(img.resize((100,100)), use_container_width=True)
                        sev   = res.get("severity","moderate")
                        color = SEVERITY_COLOR.get(sev,("#6b7280","⚪"))[0]
                        name  = res.get("display_name","Unknown")[:18]
                        conf  = res.get("confidence",0) * 100
                        st.markdown(
                            f"<div style='font-size:0.72rem;font-weight:600;"
                            f"color:{color};text-align:center'>{name}<br>{conf:.0f}%</div>",
                            unsafe_allow_html=True)
    elif batch_files and len(batch_files) == 1:
        st.info("Upload at least 2 leaf images for field analysis.", icon="ℹ️")
    else:
        st.markdown("""
        <div style="border:2px dashed #d1fae5;border-radius:12px;padding:36px;
          text-align:center;color:#6b7280;background:#f9fafb">
          <div style="font-size:2rem;margin-bottom:8px">🌾</div>
          <div style="font-weight:600">Upload 2–10 leaf photos from different field spots</div>
          <div style="font-size:0.85rem;margin-top:6px">
            Get a field-level infection rate and urgency verdict</div>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# TAB 3 — FIELD DIARY
# ══════════════════════════════════════════════════════════════════
with tab_diary:
    st.markdown(f"### 📔 {t('scan_history', lang)}")
    if not FARMER_TOOLS_OK:
        st.error("farmer_tools.py not found in project folder.")
    else:
        diary: FieldDiary = st.session_state.get("diary", FieldDiary())
        scans = diary.get_all()
        if not scans:
            st.markdown("""
            <div style="border:2px dashed #e5e7eb;border-radius:12px;padding:36px;
              text-align:center;color:#6b7280;background:#f9fafb">
              <div style="font-size:2rem;margin-bottom:8px">📔</div>
              <div style="font-weight:600">No scans saved yet</div>
              <div style="font-size:0.85rem;margin-top:6px">
                Scan a leaf and click "Save to Field Diary"</div>
            </div>""", unsafe_allow_html=True)
        else:
            stats = diary.summary_stats()
            sc1, sc2, sc3, sc4 = st.columns(4)
            for col, val, lbl in [
                (sc1, stats.get("total_scans",0), "Total Scans"),
                (sc2, f"{stats.get('healthy_pct',0):.0f}%", "Healthy"),
                (sc3, stats.get("unique_crops",0), "Crops"),
                (sc4, (stats.get("top_disease",(None,None)) or ("—",))[0]
                      if stats.get("top_disease") else "—", "Top Disease"),
            ]:
                col.markdown(
                    f"<div style='background:white;border-radius:10px;padding:14px;"
                    f"text-align:center;border:1px solid #e5e7eb'>"
                    f"<div style='font-size:1.5rem;font-weight:800;color:#059669'>{val}</div>"
                    f"<div style='font-size:0.75rem;color:#6b7280'>{lbl}</div></div>",
                    unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Severity bar chart
            sev_counts = stats.get("severity_counts",{})
            if sev_counts:
                st.markdown("**Severity distribution:**")
                total = max(sum(sev_counts.values()), 1)
                sev_cols = st.columns(len(sev_counts))
                for col, (sev, cnt) in zip(sev_cols, sev_counts.items()):
                    color = SEVERITY_COLOR.get(sev,("#6b7280","⚪"))[0]
                    pct   = cnt/total*100
                    col.markdown(
                        f"<div style='text-align:center'>"
                        f"<div style='background:{color};height:{max(int(pct),4)}px;"
                        f"border-radius:4px;margin:0 4px'></div>"
                        f"<div style='font-size:0.73rem;margin-top:4px;font-weight:600'>"
                        f"{sev.title()}</div>"
                        f"<div style='font-size:0.7rem;color:#6b7280'>{cnt}</div>"
                        f"</div>", unsafe_allow_html=True)

            st.divider()
            st.markdown(f"**{len(scans)} scans recorded:**")
            for scan in scans[:25]:
                sc_color = SEVERITY_COLOR.get(scan["severity"],("#6b7280","⚪"))[0]
                ts_str   = scan["timestamp"][:16].replace("T"," ")
                r1, r2, r3 = st.columns([0.7, 3.3, 0.7])
                with r1:
                    if scan.get("thumbnail_pil"):
                        try:
                            st.image(scan["thumbnail_pil"], width=56)
                        except Exception:
                            st.write("🌿")
                with r2:
                    st.markdown(
                        f"**{scan['display_name']}** "
                        f"<span style='color:{sc_color};font-weight:700'>"
                        f"{scan['confidence']*100:.0f}%</span> "
                        f"<span style='color:#9ca3af;font-size:0.78rem'>{ts_str}</span>",
                        unsafe_allow_html=True)
                    if scan.get("location_name"):
                        st.caption(f"📍 {scan['location_name']}")
                    if scan.get("notes"):
                        st.caption(f"📝 {scan['notes']}")
                with r3:
                    if st.button("🗑", key=f"del_{scan['scan_id']}"):
                        diary.delete(scan["scan_id"])
                        st.rerun()

            st.divider()
            csv_data = diary.export_csv()
            st.download_button("⬇️ Export diary as CSV",
                               csv_data.encode("utf-8"),
                               "cropguard_field_diary.csv", "text/csv")

# ══════════════════════════════════════════════════════════════════
# TAB 4 — ABOUT
# ══════════════════════════════════════════════════════════════════
with tab_about:
    st.markdown("""
    ### How CropGuard AI Works

    **T1.7 — Multi-Crop Unified Model (38 classes)**
    MobileNetV3-Small fine-tuned on PlantVillage (54,303 images, 14 crops).
    Two-stage training: freeze backbone for epoch 1, unfreeze for epochs 2–3.
    OneCycleLR + label smoothing + data augmentation. Achieves ~97% validation accuracy.

    **T1.8 — Mobile / PWA Ready**
    ONNX → TFLite export for Android/iOS native inference.
    PWA manifest + meta tags for "Add to Home Screen" on any HTTPS-hosted deployment.

    **Farmer Features**
    | Feature | What it does |
    |---------|-------------|
    | Field Diary | Saves every scan with thumbnail, GPS, notes — tracks season progression |
    | Batch Field Scan | Upload 5+ leaves, get field-level infection % + urgency verdict |
    | Weather Spread Risk | Live weather (Open-Meteo) × disease biology → spray-today decision |
    | Treatment Guide | Chemicals, INR costs, dosage, where to buy for each disease |
    | GradCAM Visual | Shows farmer which part of leaf triggered the AI prediction |
    | Report Export | PNG card a farmer can WhatsApp to their agronomist |
    | Language Toggle | Hindi, Telugu, Marathi UI labels |

    **Dataset:** Mohanty et al. 2016, PlantVillage, Frontiers in Plant Science.
    doi:10.3389/fpls.2016.01419

    **Emergency contacts**
    - Kisan Call Centre: **1800-180-1551** (free, Mon–Sat 6am–10pm)
    - PM Kisan Helpline: **155261**
    - Find your KVK: [kvk.icar.gov.in](https://kvk.icar.gov.in)
    """)

# ══════════════════════════════════════════════════════════════════
# PWA TAGS (T1.8 requirement)
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<link rel="manifest" href="/app/static/manifest.json">
<meta name="theme-color" content="#064e3b">
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-title" content="CropGuard AI">
""", unsafe_allow_html=True)
