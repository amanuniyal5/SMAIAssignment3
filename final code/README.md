# 🌿 CropGuard AI — Multi-Crop Disease Detection
### T1.7 (38-class unified model) + T1.8 (Mobile / PWA)

> Photograph a sick leaf → Disease name + confidence → Farmer-friendly action plan

---

## 📦 Project Structure

```
crop_disease/
├── app.py                  # Main Streamlit application
├── kaggle_train.py         # Full training script (run on Kaggle GPU)
├── disease_info.json       # Pre-written descriptions for all 38 classes
├── class_names.json        # Class list (generated after training)
├── best_model.pth          # ← Place your trained model here
├── requirements.txt
├── sample_images/          # ← Add .jpg sample leaf images for demo
│   ├── tomato_blight.jpg
│   └── ...
├── static/
│   └── manifest.json       # PWA manifest (T1.8)
└── .streamlit/
    └── config.toml         # Theme & server config
```

---

## 🚀 Quick Start (Demo Mode — no GPU needed)

```bash
# 1. Clone / download this folder
cd crop_disease

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app (demo mode — no model required)
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

> **Demo mode** runs illustrative predictions using image-hash seeding.
> Every image gets a deterministic, plausible-looking result.
> To enable real inference, complete the training step below.

---

## 🏋️ Training on Kaggle (GPU — ~10 minutes)

### Step 1: Set up Kaggle

1. Go to [kaggle.com](https://www.kaggle.com) → Create a new notebook
2. Add dataset: `abdallahalidev/plantvillage-dataset`
3. Enable GPU: **Settings → Accelerator → GPU P100**

### Step 2: Upload and run the training script

Paste the contents of `kaggle_train.py` into a Kaggle code cell, or upload
the file and run it. The script will:

1. Load all 54,303 PlantVillage images (38 classes, 14 crops)
2. Split 80% train / 10% val / 10% test
3. Fine-tune **MobileNetV3-Small** for 3 epochs with:
   - Label smoothing (0.1)
   - OneCycleLR scheduler
   - Two-stage unfreezing
   - Aggressive data augmentation
4. Save `best_model.pth` (best validation accuracy checkpoint)
5. Export to **ONNX + TFLite** for mobile (T1.8)
6. Plot training curves

**Expected results after 3 epochs:**
- Validation accuracy: ~93–97%
- Macro F1: ~0.92–0.96
- Inference time: ~20–50 ms on CPU

### Step 3: Download and deploy

1. Download `best_model.pth` and `class_names.json` from Kaggle output
2. Place both files in your `crop_disease/` folder
3. Restart the Streamlit app → real inference is now active

---

## 🌐 Deployment Options

### Option A: Streamlit Cloud (recommended for hackathon demo)

```bash
# 1. Push to GitHub (include best_model.pth via Git LFS or a download script)
git lfs track "*.pth"
git add .
git commit -m "CropGuard AI v1"
git push

# 2. Go to share.streamlit.io → Deploy from GitHub
# 3. Set GEMINI_API_KEY in Streamlit Secrets (optional)
```

### Option B: Hugging Face Spaces

```bash
# Create a new Space with SDK: Streamlit
# Upload all files including best_model.pth
# The app runs on free CPU tier
```

### Option C: PWA (T1.8) on any HTTPS host

The app includes a `manifest.json` and PWA meta tags.
Host on any HTTPS server and users can:
- **Android Chrome:** "Add to Home Screen" → installs as native-like app
- **iOS Safari:** Share → "Add to Home Screen"
- **Desktop Chrome/Edge:** Install icon in address bar

For fully offline PWA, add a service worker (see `static/sw.js` template below).

---

## 📱 TFLite Mobile Deployment (T1.8 Advanced)

After running `kaggle_train.py`, you get `tflite_model/` output.

### Android (Kotlin)
```kotlin
val interpreter = Interpreter(loadModelFile(assets, "model.tflite"))
val input  = Array(1) { Array(224) { Array(224) { FloatArray(3) } } }
val output = Array(1) { FloatArray(38) }
interpreter.run(input, output)
```

### Python TFLite inference
```python
import numpy as np, tflite_runtime.interpreter as tflite
from PIL import Image

interp = tflite.Interpreter("tflite_model/model.tflite")
interp.allocate_tensors()
inp  = interp.get_input_details()[0]
out  = interp.get_output_details()[0]

img = Image.open("leaf.jpg").resize((224, 224))
x   = (np.array(img, np.float32)[None] / 255.0 - 0.485) / 0.229
interp.set_tensor(inp["index"], x)
interp.invoke()
probs = interp.get_tensor(out["index"])[0]
print("Predicted:", class_names[probs.argmax()])
```

---

## 🔑 Gemini API Key (Optional Live Descriptions)

1. Get a free key at [aistudio.google.com](https://aistudio.google.com)
2. Enter it in the sidebar of the running app
3. Disease descriptions are now generated live per prediction

Or set it as a Streamlit secret:
```toml
# .streamlit/secrets.toml
GEMINI_API_KEY = "AIza..."
```
And read it in `app.py`: `os.environ.get("GEMINI_API_KEY", "")`

---

## 🌾 Supported Classes (38 total)

| Crop | Diseases |
|------|---------|
| Apple | Apple Scab, Black Rot, Cedar Apple Rust, Healthy |
| Blueberry | Healthy |
| Cherry | Powdery Mildew, Healthy |
| Corn | Gray Leaf Spot, Common Rust, Northern Leaf Blight, Healthy |
| Grape | Black Rot, Esca, Leaf Blight, Healthy |
| Orange | Citrus Greening (HLB) |
| Peach | Bacterial Spot, Healthy |
| Bell Pepper | Bacterial Spot, Healthy |
| Potato | Early Blight, Late Blight, Healthy |
| Raspberry | Healthy |
| Soybean | Healthy |
| Squash | Powdery Mildew |
| Strawberry | Leaf Scorch, Healthy |
| Tomato | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria, Spider Mites, Target Spot, TYLCV, Mosaic Virus, Healthy |

---

## 📊 Architecture Details

```
Input: 224×224 RGB image
  ↓
MobileNetV3-Small (ImageNet pre-trained)
  — Stage 1 (epoch 1): Backbone frozen, classifier head trained only
  — Stage 2 (epochs 2–3): Full network fine-tuned at LR/5
  ↓
AdaptiveAvgPool → Classifier head
  ↓
38-class softmax output
```

**Training hyperparameters:**
- Optimizer: AdamW (weight decay 1e-4)
- Scheduler: OneCycleLR (max_lr = 3e-4)
- Loss: CrossEntropyLoss with label smoothing 0.1
- Batch size: 64
- Augmentation: RandomResizedCrop, RandomFlip, ColorJitter, RandomRotation

---

## 📚 References

- Mohanty, S.P., Hughes, D., Salathé, M. (2016). *Using deep learning for image-based plant disease detection.* Frontiers in Plant Science. [doi:10.3389/fpls.2016.01419](https://doi.org/10.3389/fpls.2016.01419)
- Dataset: [kaggle.com/datasets/abdallahalidev/plantvillage-dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
- Model: [timm MobileNetV3](https://huggingface.co/timm/mobilenetv3_small_100.lamb_in1k)
