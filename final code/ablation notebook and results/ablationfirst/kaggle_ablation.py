"""
╔══════════════════════════════════════════════════════════════════╗
║  kaggle_ablation.py — CropGuard Ablation Study                  ║
║  Trains 5 architectures on PlantVillage, saves all checkpoints  ║
║  and a unified ablation_results.json for the dashboard          ║
╚══════════════════════════════════════════════════════════════════╝

Run in a Kaggle notebook (GPU P100/T4) with:
    Dataset: abdallahalidev/plantvillage-dataset

Each model gets:
  • Same 80/10/10 split (fixed seed)
  • Same augmentation pipeline
  • Same 3-epoch two-stage fine-tuning regime
  • Accuracy, Macro-F1, inference latency, param count

Output files in /kaggle/working/:
  ablation_results.json          ← load into dashboard
  ablation_mobilenetv3_small.pth
  ablation_efficientnet_b0.pth
  ablation_resnet18.pth
  ablation_efficientnet_b2.pth
  ablation_vit_tiny.pth
  ablation_curves.png
"""

# ── Cell 1: Install ───────────────────────────────────────────────
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "timm", "torchmetrics", "matplotlib", "seaborn"], check=True)

# ── Cell 2: Imports ───────────────────────────────────────────────
import os, json, time, random
from pathlib import Path
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from torchmetrics import Accuracy, F1Score
from torchmetrics.classification import MulticlassConfusionMatrix
import timm

# ── Cell 3: Config ────────────────────────────────────────────────
SEED       = 42
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
DATA_ROOT  = Path("/kaggle/input/plantvillage-dataset/color")
OUT_DIR    = Path("/kaggle/working")
EPOCHS     = 3
BATCH_SIZE = 64
IMG_SIZE   = 224
LR         = 3e-4

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
print(f"Device: {DEVICE} | Data: {DATA_ROOT}")

# ── Cell 4: Models to compare ─────────────────────────────────────
# Format: (timm_name, display_name, img_size)
MODELS = [
    ("mobilenetv3_small_100",  "MobileNetV3-Small",  224),
    ("efficientnet_b0",        "EfficientNet-B0",     224),
    ("resnet18",               "ResNet-18",           224),
    ("efficientnet_b2",        "EfficientNet-B2",     260),
    ("vit_tiny_patch16_224",   "ViT-Tiny",            224),
]

# ── Cell 5: Dataset helpers ───────────────────────────────────────
def make_transforms(img_size):
    train_tfm = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3,
                               saturation=0.2, hue=0.05),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tfm = transforms.Compose([
        transforms.Resize(int(img_size * 256 / 224)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_tfm, val_tfm

class SubsetWithTransform(Dataset):
    def __init__(self, base_ds, indices, tfm):
        self.base = base_ds
        self.idx  = list(indices)
        self.tfm  = tfm
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        img, lbl = self.base[self.idx[i]]
        return self.tfm(img), lbl

# Build fixed splits once (same for all models)
full_ds     = datasets.ImageFolder(DATA_ROOT)
CLASS_NAMES = full_ds.classes
NUM_CLASSES = len(CLASS_NAMES)
print(f"Classes: {NUM_CLASSES} | Images: {len(full_ds)}")

n       = len(full_ds)
n_train = int(0.8 * n)
n_val   = int(0.1 * n)
n_test  = n - n_train - n_val
g       = torch.Generator().manual_seed(SEED)
train_idx, val_idx, test_idx = torch.utils.data.random_split(
    range(n), [n_train, n_val, n_test], generator=g)

def make_loaders(img_size):
    tr, vl = make_transforms(img_size)
    return (
        DataLoader(SubsetWithTransform(full_ds, train_idx, tr),
                   batch_size=BATCH_SIZE, shuffle=True,
                   num_workers=4, pin_memory=True),
        DataLoader(SubsetWithTransform(full_ds, val_idx, vl),
                   batch_size=BATCH_SIZE, shuffle=False,
                   num_workers=4, pin_memory=True),
        DataLoader(SubsetWithTransform(full_ds, test_idx, vl),
                   batch_size=BATCH_SIZE, shuffle=False,
                   num_workers=4, pin_memory=True),
    )

# ── Cell 6: Count params ──────────────────────────────────────────
def count_params(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

# ── Cell 7: Measure inference latency (ms / image, CPU) ──────────
def measure_latency(model, img_size, n_runs=50):
    model_cpu = deepcopy(model).cpu().eval()
    dummy     = torch.randn(1, 3, img_size, img_size)
    # Warmup
    for _ in range(5):
        with torch.no_grad(): model_cpu(dummy)
    t0 = time.perf_counter()
    for _ in range(n_runs):
        with torch.no_grad(): model_cpu(dummy)
    return (time.perf_counter() - t0) / n_runs * 1000   # ms

# ── Cell 8: Train one model ───────────────────────────────────────
def train_model(timm_name, display_name, img_size):
    print(f"\n{'='*60}")
    print(f"  Training: {display_name}  ({timm_name})")
    print(f"{'='*60}")

    train_loader, val_loader, test_loader = make_loaders(img_size)

    # Build model
    net = timm.create_model(timm_name, pretrained=True,
                            num_classes=NUM_CLASSES).to(DEVICE)

    total_params, _ = count_params(net)

    # Stage 1: freeze backbone
    for name, p in net.named_parameters():
        if not any(k in name for k in ["classifier", "head", "fc"]):
            p.requires_grad = False

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, net.parameters()),
        lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR,
        steps_per_epoch=len(train_loader), epochs=EPOCHS)

    acc_m = Accuracy(task="multiclass", num_classes=NUM_CLASSES).to(DEVICE)
    f1_m  = F1Score(task="multiclass",  num_classes=NUM_CLASSES,
                    average="macro").to(DEVICE)

    history   = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}
    best_acc  = 0.0
    best_state= None
    t_start   = time.time()

    for epoch in range(1, EPOCHS + 1):
        # Unfreeze at epoch 2
        if epoch == 2:
            for p in net.parameters():
                p.requires_grad = True
            optimizer = optim.AdamW(net.parameters(),
                                    lr=LR / 5, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=LR / 5,
                steps_per_epoch=len(train_loader), epochs=EPOCHS - 1)

        # Train
        net.train(); tr_loss = 0.0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(net(imgs), lbls)
            loss.backward(); optimizer.step(); scheduler.step()
            tr_loss += loss.item()
        tr_loss /= len(train_loader)

        # Validate
        net.eval(); vl_loss = 0.0
        acc_m.reset(); f1_m.reset()
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                logits = net(imgs)
                vl_loss += criterion(logits, lbls).item()
                acc_m.update(logits.argmax(1), lbls)
                f1_m.update(logits.argmax(1), lbls)
        vl_loss /= len(val_loader)
        va = acc_m.compute().item()
        vf = f1_m.compute().item()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(va)
        history["val_f1"].append(vf)
        print(f"  Epoch {epoch}/{EPOCHS} | tr_loss={tr_loss:.4f} "
              f"vl_loss={vl_loss:.4f} acc={va:.4f} f1={vf:.4f}")

        if va > best_acc:
            best_acc   = va
            best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}

    train_time = time.time() - t_start

    # ── Test evaluation ──
    net.load_state_dict(best_state)
    net.eval()
    acc_m.reset(); f1_m.reset()
    cm_metric = MulticlassConfusionMatrix(num_classes=NUM_CLASSES).to(DEVICE)
    per_class_correct = torch.zeros(NUM_CLASSES)
    per_class_total   = torch.zeros(NUM_CLASSES)

    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            logits = net(imgs)
            preds  = logits.argmax(1)
            acc_m.update(preds, lbls)
            f1_m.update(preds, lbls)
            cm_metric.update(preds, lbls)
            for c in range(NUM_CLASSES):
                mask = lbls == c
                per_class_correct[c] += (preds[mask] == lbls[mask]).sum().cpu()
                per_class_total[c]   += mask.sum().cpu()

    test_acc = acc_m.compute().item()
    test_f1  = f1_m.compute().item()
    cm       = cm_metric.compute().cpu().numpy().tolist()
    per_class_acc = (per_class_correct / per_class_total.clamp(min=1)).tolist()

    latency_ms = measure_latency(net, img_size)

    print(f"  ✓ Test acc={test_acc:.4f}  f1={test_f1:.4f}  "
          f"latency={latency_ms:.1f}ms  params={total_params/1e6:.2f}M")

    # Save checkpoint
    safe_name = timm_name.replace("/", "_")
    ckpt_path = OUT_DIR / f"ablation_{safe_name}.pth"
    torch.save({
        "timm_name":         timm_name,
        "display_name":      display_name,
        "model_state_dict":  best_state,
        "class_names":       CLASS_NAMES,
        "val_acc":           best_acc,
        "test_acc":          test_acc,
        "test_f1":           test_f1,
        "img_size":          img_size,
        "total_params":      total_params,
        "latency_ms":        latency_ms,
        "train_time_s":      train_time,
        "history":           history,
        "confusion_matrix":  cm,
        "per_class_acc":     per_class_acc,
    }, ckpt_path)

    return {
        "timm_name":       timm_name,
        "display_name":    display_name,
        "img_size":        img_size,
        "val_acc":         round(best_acc, 4),
        "test_acc":        round(test_acc, 4),
        "test_f1":         round(test_f1, 4),
        "total_params_M":  round(total_params / 1e6, 2),
        "latency_ms":      round(latency_ms, 2),
        "train_time_s":    round(train_time, 1),
        "history":         history,
        "confusion_matrix": cm,
        "per_class_acc":   per_class_acc,
        "checkpoint":      str(ckpt_path),
    }

# ── Cell 9: Run all models ────────────────────────────────────────
all_results = []
for timm_name, display_name, img_size in MODELS:
    try:
        result = train_model(timm_name, display_name, img_size)
        all_results.append(result)
    except Exception as e:
        print(f"  ✗ {display_name} failed: {e}")

# ── Cell 10: Save ablation_results.json ───────────────────────────
ablation_out = {
    "class_names": CLASS_NAMES,
    "num_classes": NUM_CLASSES,
    "dataset": "PlantVillage (color, 54303 images)",
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "seed": SEED,
    "models": all_results,
}
with open(OUT_DIR / "ablation_results.json", "w") as f:
    json.dump(ablation_out, f, indent=2)
print("\n✅ Saved ablation_results.json")

# ── Cell 11: Summary table ────────────────────────────────────────
print("\n" + "="*75)
print(f"{'Model':<22} {'Val Acc':>8} {'Test Acc':>9} {'F1':>7} "
      f"{'Params(M)':>10} {'Latency':>9}")
print("-"*75)
for r in sorted(all_results, key=lambda x: x["test_acc"], reverse=True):
    print(f"{r['display_name']:<22} {r['val_acc']:>8.4f} {r['test_acc']:>9.4f} "
          f"{r['test_f1']:>7.4f} {r['total_params_M']:>10.2f} "
          f"{r['latency_ms']:>8.1f}ms")

# ── Cell 12: Comparison plots ─────────────────────────────────────
names   = [r["display_name"]  for r in all_results]
accs    = [r["test_acc"]      for r in all_results]
f1s     = [r["test_f1"]       for r in all_results]
params  = [r["total_params_M"] for r in all_results]
latency = [r["latency_ms"]    for r in all_results]

colors = ["#059669", "#3b82f6", "#f59e0b", "#ef4444", "#8b5cf6"]

fig = plt.figure(figsize=(18, 12))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

# 1. Test accuracy bar
ax1 = fig.add_subplot(gs[0, 0])
bars = ax1.barh(names, accs, color=colors, edgecolor="white")
ax1.set_xlim(min(accs) - 0.05, 1.0)
ax1.set_title("Test Accuracy", fontweight="bold")
ax1.set_xlabel("Accuracy")
for bar, v in zip(bars, accs):
    ax1.text(v + 0.002, bar.get_y() + bar.get_height()/2,
             f"{v:.3f}", va="center", fontsize=8)

# 2. Macro F1
ax2 = fig.add_subplot(gs[0, 1])
bars2 = ax2.barh(names, f1s, color=colors, edgecolor="white")
ax2.set_xlim(min(f1s) - 0.05, 1.0)
ax2.set_title("Macro F1 Score", fontweight="bold")
ax2.set_xlabel("F1")
for bar, v in zip(bars2, f1s):
    ax2.text(v + 0.002, bar.get_y() + bar.get_height()/2,
             f"{v:.3f}", va="center", fontsize=8)

# 3. Accuracy vs Params scatter
ax3 = fig.add_subplot(gs[0, 2])
for i, r in enumerate(all_results):
    ax3.scatter(r["total_params_M"], r["test_acc"],
                color=colors[i], s=120, zorder=3, label=r["display_name"])
    ax3.annotate(r["display_name"].split("-")[0],
                 (r["total_params_M"], r["test_acc"]),
                 textcoords="offset points", xytext=(5, 3), fontsize=7)
ax3.set_xlabel("Parameters (M)")
ax3.set_ylabel("Test Accuracy")
ax3.set_title("Accuracy vs Model Size", fontweight="bold")
ax3.grid(alpha=0.3)

# 4. Accuracy vs Latency scatter
ax4 = fig.add_subplot(gs[1, 0])
for i, r in enumerate(all_results):
    ax4.scatter(r["latency_ms"], r["test_acc"],
                color=colors[i], s=120, zorder=3)
    ax4.annotate(r["display_name"].split("-")[0],
                 (r["latency_ms"], r["test_acc"]),
                 textcoords="offset points", xytext=(5, 3), fontsize=7)
ax4.set_xlabel("CPU Inference Latency (ms)")
ax4.set_ylabel("Test Accuracy")
ax4.set_title("Accuracy vs Speed", fontweight="bold")
ax4.grid(alpha=0.3)

# 5. Training curves (val acc per epoch for each model)
ax5 = fig.add_subplot(gs[1, 1])
for i, r in enumerate(all_results):
    ax5.plot(range(1, EPOCHS+1), r["history"]["val_acc"],
             marker="o", color=colors[i], label=r["display_name"])
ax5.set_xlabel("Epoch")
ax5.set_ylabel("Val Accuracy")
ax5.set_title("Validation Accuracy per Epoch", fontweight="bold")
ax5.legend(fontsize=7)
ax5.grid(alpha=0.3)

# 6. Latency bar
ax6 = fig.add_subplot(gs[1, 2])
bars6 = ax6.barh(names, latency, color=colors, edgecolor="white")
ax6.set_title("CPU Inference Latency", fontweight="bold")
ax6.set_xlabel("ms / image")
for bar, v in zip(bars6, latency):
    ax6.text(v + 0.3, bar.get_y() + bar.get_height()/2,
             f"{v:.1f}ms", va="center", fontsize=8)

fig.suptitle("CropGuard AI — Ablation Study: Architecture Comparison",
             fontsize=14, fontweight="bold", y=1.01)
plt.savefig(OUT_DIR / "ablation_curves.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n✅ Saved ablation_curves.png")
print("\n=== ABLATION COMPLETE ===")
print(f"Download from {OUT_DIR}:")
for p in sorted(OUT_DIR.glob("ablation*")):
    print(f"  {p.name}")
