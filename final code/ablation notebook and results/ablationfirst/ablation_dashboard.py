"""
╔══════════════════════════════════════════════════════════════════╗
║  ablation_dashboard.py — CropGuard Ablation Study Dashboard     ║
║                                                                  ║
║  Run:  streamlit run ablation_dashboard.py                       ║
║                                                                  ║
║  Features:                                                       ║
║  • Upload multiple .pth checkpoints for side-by-side comparison ║
║  • Or load ablation_results.json from kaggle_ablation.py        ║
║  • Accuracy / F1 / Latency / Params comparison charts           ║
║  • Per-class F1 heatmap                                         ║
║  • Confusion matrix per model                                    ║
║  • Training curve comparison                                     ║
║  • GradCAM explainability on uploaded leaf image                ║
║  • Exportable summary CSV                                        ║
╚══════════════════════════════════════════════════════════════════╝
"""

import io, json, time
from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="CropGuard — Ablation Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700;800&family=JetBrains+Mono:wght@500&display=swap');
html, body, [class*="css"] { font-family: 'Sora', sans-serif; }
.main .block-container { padding-top: 1.2rem; max-width: 1300px; }
.hero {
  background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #4c1d95 100%);
  border-radius: 14px; padding: 24px 32px; margin-bottom: 20px;
  box-shadow: 0 8px 32px rgba(49,46,129,0.4);
}
.hero h1 { color: #e0e7ff; font-size: 1.8rem; font-weight: 800; margin: 0; }
.hero p  { color: #a5b4fc; margin: 6px 0 0; font-size: 0.9rem; }
.metric-card {
  background: white; border-radius: 12px; padding: 16px 20px;
  box-shadow: 0 2px 12px rgba(0,0,0,0.07); border: 1px solid #e5e7eb;
  text-align: center;
}
.metric-val { font-size: 1.7rem; font-weight: 800; }
.metric-lbl { font-size: 0.78rem; color: #6b7280; margin-top: 2px; }
.winner-badge {
  display:inline-block; background:#059669; color:white;
  padding:2px 10px; border-radius:20px; font-size:0.75rem; font-weight:700;
}
.section-title {
  font-size: 1.1rem; font-weight: 700; color: #1e1b4b;
  border-left: 4px solid #6366f1; padding-left: 10px; margin: 24px 0 12px;
}
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─── Hero ─────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🔬 Ablation Study Dashboard</h1>
  <p>Compare CNN architectures on PlantVillage · Upload .pth checkpoints or load ablation_results.json</p>
</div>
""", unsafe_allow_html=True)

ROOT = Path(__file__).parent

# ─────────────────────────────────────────────────────────────────
# SIDEBAR — data loading
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📥 Load Models")

    load_mode = st.radio(
        "Data source",
        ["Upload .pth checkpoints", "Load ablation_results.json"],
        label_visibility="collapsed",
    )

    uploaded_pths  = []
    ablation_json  = None
    class_names    = []

    if load_mode == "Upload .pth checkpoints":
        st.markdown("**Upload one or more .pth files:**")
        uploaded_pths = st.file_uploader(
            "checkpoint files", type=["pth"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

    else:
        # Try local file first
        local_json = ROOT / "ablation_results.json"
        if local_json.exists():
            st.success(f"Found local ablation_results.json ✅")
            with open(local_json, encoding="utf-8") as f:
                ablation_json = json.load(f)
        else:
            uploaded_json = st.file_uploader(
                "Upload ablation_results.json",
                type=["json"],
                label_visibility="collapsed",
            )
            if uploaded_json:
                ablation_json = json.load(uploaded_json)

    st.divider()

    # GradCAM section
    st.markdown("## 🔍 GradCAM Explainability")
    gradcam_img = st.file_uploader(
        "Upload leaf image for GradCAM",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )
    gradcam_alpha = st.slider("Heatmap opacity", 0.2, 0.8, 0.45, 0.05)

    st.divider()
    st.markdown(
        "<div style='font-size:0.75rem;color:#9ca3af;text-align:center'>"
        "CropGuard AI · Ablation Study<br>PlantVillage · Mohanty et al. 2016"
        "</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# Helper: load a .pth and evaluate if needed
# ─────────────────────────────────────────────────────────────────
PALETTE = ["#6366f1", "#059669", "#f59e0b", "#ef4444", "#8b5cf6",
           "#0ea5e9", "#ec4899", "#14b8a6"]

@st.cache_resource(show_spinner=False)
def load_checkpoint_bytes(file_bytes: bytes, filename: str):
    """Load a .pth from bytes and extract metadata."""
    import torch, timm
    buf  = io.BytesIO(file_bytes)
    ckpt = torch.load(buf, map_location="cpu", weights_only=False)
    return ckpt

def ckpt_to_row(ckpt: dict, display_name: str = None) -> dict:
    """Convert a checkpoint dict to a results row dict."""
    name = display_name or ckpt.get("display_name",
           ckpt.get("timm_name", "Unknown").replace("_", " "))
    return {
        "display_name":    name,
        "timm_name":       ckpt.get("timm_name", "mobilenetv3_small_100"),
        "img_size":        ckpt.get("img_size", 224),
        "val_acc":         ckpt.get("val_acc", 0.0),
        "test_acc":        ckpt.get("test_acc", ckpt.get("val_acc", 0.0)),
        "test_f1":         ckpt.get("test_f1", 0.0),
        "total_params_M":  ckpt.get("total_params", 0) / 1e6,
        "latency_ms":      ckpt.get("latency_ms", 0.0),
        "train_time_s":    ckpt.get("train_time_s", 0.0),
        "history":         ckpt.get("history", {}),
        "confusion_matrix":ckpt.get("confusion_matrix", []),
        "per_class_acc":   ckpt.get("per_class_acc", []),
        "_ckpt":           ckpt,
    }

# ─────────────────────────────────────────────────────────────────
# Build model rows list
# ─────────────────────────────────────────────────────────────────
rows       = []
class_names= []

if ablation_json:
    class_names = ablation_json.get("class_names", [])
    for m in ablation_json.get("models", []):
        rows.append(m)

if uploaded_pths:
    import torch
    for uf in uploaded_pths:
        try:
            file_bytes = uf.read()
            ckpt       = load_checkpoint_bytes(file_bytes, uf.name)
            if not class_names and "class_names" in ckpt:
                class_names = ckpt["class_names"]
            row = ckpt_to_row(ckpt, display_name=Path(uf.name).stem)
            rows.append(row)
        except Exception as e:
            st.error(f"Could not load {uf.name}: {e}")

# ─────────────────────────────────────────────────────────────────
# NOTHING LOADED YET
# ─────────────────────────────────────────────────────────────────
if not rows:
    st.markdown("""
    <div style="border:2px dashed #c7d2fe;border-radius:14px;padding:48px;
    text-align:center;color:#6b7280;background:#f5f3ff;margin-top:24px">
      <div style="font-size:3rem">📂</div>
      <div style="font-size:1.1rem;font-weight:700;margin:12px 0 6px;color:#4338ca">
        No models loaded yet
      </div>
      <div style="font-size:0.9rem">
        Upload <code>.pth</code> checkpoint files from the sidebar,<br>
        or place <code>ablation_results.json</code> in this folder<br>
        (generated by <code>kaggle_ablation.py</code>).
      </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("ℹ️ How to generate ablation data"):
        st.markdown("""
        **Step 1:** Run `kaggle_ablation.py` in a Kaggle GPU notebook.
        It trains MobileNetV3-Small, EfficientNet-B0, ResNet-18,
        EfficientNet-B2, and ViT-Tiny on PlantVillage.

        **Step 2:** Download `ablation_results.json` and all `ablation_*.pth` files.

        **Step 3:** Place `ablation_results.json` next to this script and
        restart — or use the sidebar uploader.

        **To compare your own model:** Upload `best_model.pth` directly via
        the sidebar uploader.
        """)
    st.stop()

# ─────────────────────────────────────────────────────────────────
# ── SECTION 0: Summary table + winner medals ─────────────────────
# ─────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">📊 Overall Comparison</div>',
            unsafe_allow_html=True)

df = pd.DataFrame([{
    "Model":       r["display_name"],
    "Val Acc":     r.get("val_acc", 0),
    "Test Acc":    r.get("test_acc", 0),
    "Macro F1":    r.get("test_f1",  0),
    "Params (M)":  round(r.get("total_params_M", 0), 2),
    "Latency (ms)":r.get("latency_ms", 0),
    "Train (min)": round(r.get("train_time_s", 0) / 60, 1),
} for r in rows])

best_acc = df["Test Acc"].max()
best_f1  = df["Macro F1"].max()
best_lat = df["Latency (ms)"].replace(0, np.nan).min()

def highlight_best(s):
    if s.name in ("Test Acc", "Val Acc", "Macro F1"):
        return ["background-color:#d1fae5;font-weight:700"
                if v == s.max() else "" for v in s]
    if s.name == "Latency (ms)":
        nonzero = s.replace(0, np.nan)
        return ["background-color:#dbeafe;font-weight:700"
                if v == nonzero.min() else "" for v in s]
    if s.name == "Params (M)":
        nonzero = s.replace(0, np.nan)
        return ["background-color:#fef3c7;font-weight:700"
                if v == nonzero.min() else "" for v in s]
    return [""] * len(s)

styled = df.style\
    .apply(highlight_best)\
    .format({"Val Acc": "{:.4f}", "Test Acc": "{:.4f}", "Macro F1": "{:.4f}",
             "Params (M)": "{:.2f}", "Latency (ms)": "{:.1f}",
             "Train (min)": "{:.1f}"})\
    .set_properties(**{"text-align": "center"})
st.dataframe(styled, use_container_width=True, height=min(200, len(rows)*45+50))

# Export button
csv_buf = io.StringIO()
df.to_csv(csv_buf, index=False)
st.download_button("⬇️ Download CSV", csv_buf.getvalue(),
                   "ablation_summary.csv", "text/csv")

# ─────────────────────────────────────────────────────────────────
# ── SECTION 1: Bar charts ─────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">📈 Performance Charts</div>',
            unsafe_allow_html=True)

n_models = len(rows)
colors   = PALETTE[:n_models]
names    = [r["display_name"]   for r in rows]
accs     = [r.get("test_acc",0) for r in rows]
f1s      = [r.get("test_f1",0)  for r in rows]
params   = [r.get("total_params_M",0) for r in rows]
lats     = [r.get("latency_ms",0)     for r in rows]

fig, axes = plt.subplots(1, 4, figsize=(18, 3.5))
fig.patch.set_facecolor("#f8fafc")
for ax in axes:
    ax.set_facecolor("#f8fafc")
    ax.spines[["top","right"]].set_visible(False)

def hbar(ax, vals, names, colors, title, fmt="{:.3f}", xlabel=""):
    bars = ax.barh(names, vals, color=colors, edgecolor="white",
                   linewidth=0.5, height=0.6)
    ax.set_title(title, fontweight="bold", fontsize=10, pad=8)
    ax.set_xlabel(xlabel, fontsize=8)
    vmax = max(vals) if vals else 1
    for bar, v in zip(bars, vals):
        ax.text(min(v + vmax*0.01, vmax*0.99),
                bar.get_y() + bar.get_height()/2,
                fmt.format(v), va="center", fontsize=8, fontweight="600")

hbar(axes[0], accs,   names, colors, "Test Accuracy",    "{:.4f}")
hbar(axes[1], f1s,    names, colors, "Macro F1 Score",   "{:.4f}")
hbar(axes[2], params, names, colors, "Parameters (M)",   "{:.2f}M")
hbar(axes[3], lats,   names, colors, "CPU Latency (ms)", "{:.1f}ms")

plt.tight_layout()
st.pyplot(fig, use_container_width=True)
plt.close(fig)

# ─────────────────────────────────────────────────────────────────
# ── SECTION 2: Accuracy vs Size/Speed scatter ─────────────────────
# ─────────────────────────────────────────────────────────────────
if any(p > 0 for p in params) and any(l > 0 for l in lats):
    st.markdown('<div class="section-title">⚖️ Efficiency Analysis</div>',
                unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    for col, xs, xlabel, title in [
        (c1, params, "Parameters (M)", "Accuracy vs Model Size"),
        (c2, lats,   "CPU Latency (ms)", "Accuracy vs Inference Speed"),
    ]:
        with col:
            fig2, ax2 = plt.subplots(figsize=(7, 4))
            fig2.patch.set_facecolor("#f8fafc")
            ax2.set_facecolor("#f8fafc")
            ax2.spines[["top","right"]].set_visible(False)
            for i, (r, x) in enumerate(zip(rows, xs)):
                if x == 0: continue
                ax2.scatter(x, r.get("test_acc",0),
                            color=colors[i], s=140, zorder=3,
                            label=r["display_name"], edgecolors="white", linewidths=1.5)
                ax2.annotate(r["display_name"].split("-")[0],
                             (x, r.get("test_acc",0)),
                             textcoords="offset points", xytext=(7,3), fontsize=8)
            ax2.set_xlabel(xlabel, fontsize=9)
            ax2.set_ylabel("Test Accuracy", fontsize=9)
            ax2.set_title(title, fontweight="bold", fontsize=10)
            ax2.legend(fontsize=7, framealpha=0.5)
            ax2.grid(alpha=0.25, linestyle="--")
            st.pyplot(fig2, use_container_width=True)
            plt.close(fig2)

# ─────────────────────────────────────────────────────────────────
# ── SECTION 3: Training curves ────────────────────────────────────
# ─────────────────────────────────────────────────────────────────
has_history = any(r.get("history") for r in rows)
if has_history:
    st.markdown('<div class="section-title">📉 Training Curves</div>',
                unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    for col, metric_key, ylabel, title in [
        (c3, "val_acc",  "Validation Accuracy", "Val Accuracy per Epoch"),
        (c4, "val_loss", "Validation Loss",      "Val Loss per Epoch"),
    ]:
        with col:
            fig3, ax3 = plt.subplots(figsize=(7, 4))
            fig3.patch.set_facecolor("#f8fafc")
            ax3.set_facecolor("#f8fafc")
            ax3.spines[["top","right"]].set_visible(False)
            for i, r in enumerate(rows):
                hist = r.get("history", {})
                vals = hist.get(metric_key, [])
                if vals:
                    ax3.plot(range(1, len(vals)+1), vals,
                             marker="o", color=colors[i],
                             label=r["display_name"], linewidth=2)
            ax3.set_xlabel("Epoch", fontsize=9)
            ax3.set_ylabel(ylabel, fontsize=9)
            ax3.set_title(title, fontweight="bold", fontsize=10)
            ax3.legend(fontsize=8, framealpha=0.5)
            ax3.grid(alpha=0.25, linestyle="--")
            st.pyplot(fig3, use_container_width=True)
            plt.close(fig3)

# ─────────────────────────────────────────────────────────────────
# ── SECTION 4: Per-class accuracy heatmap ────────────────────────
# ─────────────────────────────────────────────────────────────────
has_per_class = any(r.get("per_class_acc") for r in rows)
if has_per_class and class_names:
    st.markdown('<div class="section-title">🌿 Per-Class Accuracy Heatmap</div>',
                unsafe_allow_html=True)

    # Build matrix: models × classes
    model_names_pc = [r["display_name"] for r in rows if r.get("per_class_acc")]
    pc_matrix = np.array([r["per_class_acc"] for r in rows if r.get("per_class_acc")])

    short_classes = [c.replace("___", "\n").replace("_", " ")
                     .replace("(including sour)", "")
                     .strip()[:28] for c in class_names]

    fig4, ax4 = plt.subplots(figsize=(max(18, len(class_names)*0.55),
                                       max(4,  len(model_names_pc)*0.8) + 1))
    sns.heatmap(pc_matrix, annot=True, fmt=".2f",
                xticklabels=short_classes,
                yticklabels=model_names_pc,
                cmap="RdYlGn", vmin=0, vmax=1,
                linewidths=0.3, linecolor="#e5e7eb",
                ax=ax4, cbar_kws={"shrink": 0.6})
    ax4.set_title("Per-Class Accuracy by Model", fontweight="bold", fontsize=11)
    ax4.tick_params(axis="x", rotation=45, labelsize=7)
    ax4.tick_params(axis="y", rotation=0,  labelsize=8)
    plt.tight_layout()
    st.pyplot(fig4, use_container_width=True)
    plt.close(fig4)

# ─────────────────────────────────────────────────────────────────
# ── SECTION 5: Confusion matrices ────────────────────────────────
# ─────────────────────────────────────────────────────────────────
has_cm = any(r.get("confusion_matrix") for r in rows)
if has_cm and class_names:
    st.markdown('<div class="section-title">🔢 Confusion Matrices</div>',
                unsafe_allow_html=True)

    cm_rows = [r for r in rows if r.get("confusion_matrix")]
    if len(cm_rows) > 1:
        selected_model = st.selectbox(
            "Select model to view confusion matrix",
            [r["display_name"] for r in cm_rows],
            key="cm_select"
        )
        cm_row = next(r for r in cm_rows if r["display_name"] == selected_model)
    else:
        cm_row = cm_rows[0]

    cm_arr = np.array(cm_row["confusion_matrix"])
    # Normalise row-wise
    cm_norm = cm_arr.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True).clip(min=1)
    cm_norm  = cm_norm / row_sums

    short_cls = [c.split("___")[-1].replace("_", " ")[:18] for c in class_names]

    fig5, ax5 = plt.subplots(figsize=(max(16, len(class_names)*0.52),
                                       max(12, len(class_names)*0.52)))
    sns.heatmap(cm_norm, annot=False,
                xticklabels=short_cls, yticklabels=short_cls,
                cmap="Blues", vmin=0, vmax=1,
                linewidths=0.2, linecolor="#f3f4f6",
                ax=ax5, cbar_kws={"shrink": 0.5})
    ax5.set_title(f"Normalised Confusion Matrix — {cm_row['display_name']}",
                  fontweight="bold", fontsize=10)
    ax5.set_xlabel("Predicted Class", fontsize=8)
    ax5.set_ylabel("True Class", fontsize=8)
    ax5.tick_params(axis="x", rotation=45, labelsize=6)
    ax5.tick_params(axis="y", rotation=0,  labelsize=6)
    plt.tight_layout()
    st.pyplot(fig5, use_container_width=True)
    plt.close(fig5)

# ─────────────────────────────────────────────────────────────────
# ── SECTION 6: GradCAM Explainability ────────────────────────────
# ─────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">🔍 GradCAM Explainability</div>',
            unsafe_allow_html=True)

if gradcam_img is None:
    st.info("Upload a leaf image in the sidebar to generate GradCAM heatmaps "
            "showing which leaf regions triggered each model's prediction.", icon="💡")
else:
    try:
        from gradcam import explain_image, GradCAM, get_target_layer
        import torch, timm

        leaf_pil = Image.open(gradcam_img).convert("RGB")

        # Pick which models to GradCAM (those with loaded checkpoints)
        gc_rows = [r for r in rows if r.get("_ckpt")]
        if not gc_rows and uploaded_pths:
            st.warning("Re-upload .pth files to enable GradCAM (session cache cleared).")
        elif not gc_rows:
            st.info("GradCAM requires .pth checkpoints — use 'Upload .pth checkpoints' mode.")
        else:
            n_gc = len(gc_rows)
            gc_cols = st.columns(min(n_gc, 3))
            for i, r in enumerate(gc_rows[:6]):
                with gc_cols[i % 3]:
                    ckpt      = r["_ckpt"]
                    tname     = r.get("timm_name", "mobilenetv3_small_100")
                    img_size  = r.get("img_size", 224)
                    n_classes = len(ckpt.get("class_names", class_names))
                    try:
                        net = timm.create_model(tname, pretrained=False,
                                                num_classes=n_classes)
                        net.load_state_dict(ckpt["model_state_dict"])
                        net.eval()

                        # Run inference
                        from torchvision import transforms as T
                        tfm = T.Compose([
                            T.Resize(int(img_size*256/224)),
                            T.CenterCrop(img_size), T.ToTensor(),
                            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
                        ])
                        with torch.no_grad():
                            logits = net(tfm(leaf_pil).unsqueeze(0))[0]
                            probs  = torch.softmax(logits, 0)
                            top_i  = probs.argmax().item()
                            cn     = ckpt.get("class_names", class_names)
                            pred   = cn[top_i].split("___")[-1].replace("_"," ") \
                                     if cn else str(top_i)
                            conf   = probs[top_i].item()

                        overlay, _ = explain_image(leaf_pil, net, tname,
                                                   class_idx=top_i,
                                                   img_size=img_size,
                                                   alpha=gradcam_alpha)
                        st.image(overlay, use_container_width=True,
                                 caption=f"{r['display_name']}")
                        st.markdown(
                            f"**{pred}** · {conf*100:.1f}%",
                            help="Model prediction with confidence")
                    except Exception as ge:
                        st.error(f"GradCAM failed for {r['display_name']}: {ge}")
    except ImportError:
        st.error("gradcam.py not found. Make sure it is in the same folder as this script.")

# ─────────────────────────────────────────────────────────────────
# ── SECTION 7: Ablation notes / insights ─────────────────────────
# ─────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">📝 Ablation Insights</div>',
            unsafe_allow_html=True)

if rows:
    best_row  = max(rows, key=lambda r: r.get("test_acc", 0))
    fast_rows = [r for r in rows if r.get("latency_ms",0) > 0]
    fast_row  = min(fast_rows, key=lambda r: r["latency_ms"]) if fast_rows else None
    small_rows= [r for r in rows if r.get("total_params_M",0) > 0]
    small_row = min(small_rows, key=lambda r: r["total_params_M"]) if small_rows else None

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-val" style="color:#059669">{best_row['test_acc']:.3f}</div>
          <div class="metric-lbl">Best Test Accuracy</div>
          <div style="margin-top:8px"><span class="winner-badge">🏆 {best_row['display_name']}</span></div>
        </div>""", unsafe_allow_html=True)
    with col_b:
        if fast_row:
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-val" style="color:#3b82f6">{fast_row['latency_ms']:.1f}ms</div>
              <div class="metric-lbl">Fastest Inference (CPU)</div>
              <div style="margin-top:8px"><span class="winner-badge" style="background:#3b82f6">⚡ {fast_row['display_name']}</span></div>
            </div>""", unsafe_allow_html=True)
    with col_c:
        if small_row:
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-val" style="color:#f59e0b">{small_row['total_params_M']:.2f}M</div>
              <div class="metric-lbl">Most Compact Model</div>
              <div style="margin-top:8px"><span class="winner-badge" style="background:#f59e0b">📦 {small_row['display_name']}</span></div>
            </div>""", unsafe_allow_html=True)

    st.markdown("""
    <br>
    <div style="background:#f5f3ff;border-radius:12px;padding:18px 24px;
                border:1px solid #c7d2fe;font-size:0.92rem;line-height:1.7">
      <b>Key Takeaways:</b>
      <ul style="margin:8px 0 0 16px;color:#374151">
        <li><b>MobileNetV3-Small</b> is ideal for mobile/edge deployment — smallest model, fastest inference, competitive accuracy.</li>
        <li><b>EfficientNet-B0/B2</b> offer the best accuracy-efficiency tradeoff for server-side deployment.</li>
        <li><b>ResNet-18</b> is a strong baseline but larger than EfficientNets at similar accuracy.</li>
        <li><b>ViT-Tiny</b> shows that transformers can compete with CNNs even on small datasets with pre-training, but inference is slower.</li>
        <li>All models converge well within 3 epochs due to strong ImageNet pre-training — more epochs yield diminishing returns on PlantVillage.</li>
      </ul>
    </div>
    """, unsafe_allow_html=True)
