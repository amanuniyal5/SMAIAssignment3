"""
setup_samples.py
────────────────
Reads all images from crop_disease/plantvillage_sample/<ClassName>/
and copies them into crop_disease/sample_images/ with clean filenames.

Run once from your crop_disease/ folder:
    python setup_samples.py
"""

import shutil
from pathlib import Path

SRC  = Path("plantvillage_sample")   # crop_disease/plantvillage_sample/
OUT  = Path("sample_images")
OUT.mkdir(exist_ok=True)

# Clear old synthetic images
for old in OUT.glob("*"):
    old.unlink()
print(f"Cleared old sample_images/\n")

if not SRC.exists():
    print(f"ERROR: '{SRC}' folder not found.")
    print("Make sure plantvillage_sample/ is inside your crop_disease/ folder.")
    exit(1)

# Friendly display names for each class folder
DISPLAY = {
    "Apple__Apple_scab":                                "Apple — Apple Scab",
    "Apple__Black_rot":                                 "Apple — Black Rot",
    "Apple__Cedar_apple_rust":                          "Apple — Cedar Rust",
    "Apple__healthy":                                   "Apple — Healthy",
    "Blueberry__healthy":                               "Blueberry — Healthy",
    "Cherry_(including_sour)__healthy":                 "Cherry — Healthy",
    "Cherry_(including_sour)__Powdery_mildew":          "Cherry — Powdery Mildew",
    "Corn_(maize)__Cercospora_leaf_spot Gra":           "Corn — Gray Leaf Spot",
    "Corn_(maize)__Common_rust_":                       "Corn — Common Rust",
    "Corn_(maize)__healthy":                            "Corn — Healthy",
    "Corn_(maize)__Northern_Leaf_Blight":               "Corn — Northern Blight",
    "Grape__Black_rot":                                 "Grape — Black Rot",
    "Grape__Esca_(Black_Measles)":                      "Grape — Esca",
    "Grape__healthy":                                   "Grape — Healthy",
    "Grape__Leaf_blight_(Isariopsis_Leaf_Spot)":        "Grape — Leaf Blight",
    "Orange__Haunglongbing_(Citrus_greeni":             "Orange — Citrus Greening",
    "Peach__Bacterial_spot":                            "Peach — Bacterial Spot",
    "Peach__healthy":                                   "Peach — Healthy",
    "Pepper,_bell__Bacterial_spot":                     "Pepper — Bacterial Spot",
    "Pepper,_bell__healthy":                            "Pepper — Healthy",
    "Potato__Early_blight":                             "Potato — Early Blight",
    "Potato__healthy":                                  "Potato — Healthy",
    "Potato__Late_blight":                              "Potato — Late Blight",
    "Raspberry__healthy":                               "Raspberry — Healthy",
    "Soybean__healthy":                                 "Soybean — Healthy",
    "Squash__Powdery_mildew":                           "Squash — Powdery Mildew",
    "Strawberry__healthy":                              "Strawberry — Healthy",
    "Strawberry__Leaf_scorch":                          "Strawberry — Leaf Scorch",
    "Tomato__Bacterial_spot":                           "Tomato — Bacterial Spot",
    "Tomato__Early_blight":                             "Tomato — Early Blight",
    "Tomato__healthy":                                  "Tomato — Healthy",
    "Tomato__Late_blight":                              "Tomato — Late Blight",
    "Tomato__Leaf_Mold":                                "Tomato — Leaf Mold",
    "Tomato__Septoria_leaf_spot":                       "Tomato — Septoria Spot",
    "Tomato__Spider_mites Two-spotted_spi":             "Tomato — Spider Mites",
    "Tomato__Target_Spot":                              "Tomato — Target Spot",
    "Tomato__Tomato_mosaic_virus":                      "Tomato — Mosaic Virus",
    "Tomato__Tomato_Yellow_Leaf_Curl_Virus":            "Tomato — Yellow Curl Virus",
}

copied = 0
for class_dir in sorted(SRC.iterdir()):
    if not class_dir.is_dir():
        continue

    images = sorted(
        list(class_dir.glob("*.jpg")) +
        list(class_dir.glob("*.JPG")) +
        list(class_dir.glob("*.jpeg")) +
        list(class_dir.glob("*.png")) +
        list(class_dir.glob("*.PNG"))
    )
    if not images:
        print(f"  ⚠  No images in {class_dir.name} — skipping")
        continue

    # Use the actual folder name as the stem for the output filename
    # Replace spaces with underscores, strip special chars that break filesystems
    stem = class_dir.name.replace(" ", "_").replace(",", "").replace("(", "").replace(")", "")

    for i, img_path in enumerate(images):
        suffix  = img_path.suffix.lower()
        out_name = f"{stem}__{i+1}{suffix}"
        dest    = OUT / out_name
        shutil.copy2(img_path, dest)
        print(f"  ✓  {out_name}")
        copied += 1

print(f"\n✅  {copied} images copied to {OUT}/")
print("Restart Streamlit — sample buttons will now show real leaf photos.")
