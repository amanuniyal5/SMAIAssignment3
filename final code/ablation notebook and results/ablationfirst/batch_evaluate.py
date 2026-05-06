"""
batch_evaluate.py — Offline Model Evaluation
─────────────────────────────────────────────
Evaluates any .pth checkpoint against your local
plantvillage_sample/ folder (or any ImageFolder-structured directory)
and prints a full per-class + aggregate report.

Usage:
    python batch_evaluate.py --model best_model.pth
    python batch_evaluate.py --model best_model.pth --data plantvillage_sample
    python batch_evaluate.py --model ablation_efficientnet_b0.pth --data plantvillage_sample --output report.json
    python batch_evaluate.py --compare best_model.pth ablation_efficientnet_b0.pth --data plantvillage_sample
"""

import argparse, json, time
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms
import timm


# ─────────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────────
def make_transform(img_size: int):
    return transforms.Compose([
        transforms.Resize(int(img_size * 256 / 224)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


# ─────────────────────────────────────────────────────────────────
# Load model from checkpoint
# ─────────────────────────────────────────────────────────────────
def load_model(ckpt_path: str):
    ckpt        = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    timm_name   = ckpt.get("timm_name", "mobilenetv3_small_100")
    class_names = ckpt["class_names"]
    img_size    = ckpt.get("img_size", 224)
    num_classes = len(class_names)

    model = timm.create_model(timm_name, pretrained=False, num_classes=num_classes)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    display  = ckpt.get("display_name", timm_name)
    val_acc  = ckpt.get("val_acc", None)
    return model, class_names, img_size, display, val_acc


# ─────────────────────────────────────────────────────────────────
# Discover images in an ImageFolder-style or flat directory
# ─────────────────────────────────────────────────────────────────
def discover_images(data_dir: str):
    """
    Returns list of (image_path, true_class_name).
    Supports both:
      - ImageFolder: data_dir/<ClassName>/<img>.jpg
      - Flat:        data_dir/<img>.jpg  (no labels, class = "unknown")
    """
    root    = Path(data_dir)
    samples = []
    exts    = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

    subdirs = [d for d in root.iterdir() if d.is_dir()]
    if subdirs:
        # ImageFolder mode
        for cls_dir in sorted(subdirs):
            cls_name = cls_dir.name
            for img in sorted(cls_dir.iterdir()):
                if img.suffix in exts:
                    samples.append((img, cls_name))
    else:
        # Flat mode
        for img in sorted(root.iterdir()):
            if img.suffix in exts:
                samples.append((img, "unknown"))

    return samples


# ─────────────────────────────────────────────────────────────────
# Fuzzy class-name matching (handles ___ vs _ differences)
# ─────────────────────────────────────────────────────────────────
def normalise(s: str) -> str:
    import re
    return re.sub(r"[\s_,()]+", " ", s).lower().strip()

def match_class(folder_name: str, model_class_names: list) -> int | None:
    fn = normalise(folder_name)
    for i, cn in enumerate(model_class_names):
        if normalise(cn) == fn:
            return i
    # Partial
    for i, cn in enumerate(model_class_names):
        if fn in normalise(cn) or normalise(cn) in fn:
            return i
    # Token overlap
    fn_tokens = set(fn.split())
    best_i, best_score = None, 0
    for i, cn in enumerate(model_class_names):
        score = len(fn_tokens & set(normalise(cn).split()))
        if score > best_score:
            best_score, best_i = score, i
    return best_i


# ─────────────────────────────────────────────────────────────────
# Single-model evaluation
# ─────────────────────────────────────────────────────────────────
def evaluate(model, class_names, img_size, samples, verbose=True):
    tfm  = make_transform(img_size)
    correct = 0
    total   = 0
    per_class_correct = defaultdict(int)
    per_class_total   = defaultdict(int)
    confidences       = []
    latencies         = []
    errors            = []
    top5_correct      = 0

    for img_path, true_cls in samples:
        # Map folder name to model class index
        true_idx = match_class(true_cls, class_names)
        if true_idx is None and true_cls != "unknown":
            errors.append(f"Could not map '{true_cls}' to any model class")
            continue

        try:
            img = Image.open(img_path).convert("RGB")
            x   = tfm(img).unsqueeze(0)

            t0 = time.perf_counter()
            with torch.no_grad():
                logits = model(x)[0]
                probs  = torch.softmax(logits, 0)
            latencies.append((time.perf_counter() - t0) * 1000)

            pred_idx  = probs.argmax().item()
            top5_idxs = probs.topk(min(5, len(class_names))).indices.tolist()
            conf      = probs[pred_idx].item()
            confidences.append(conf)

            if true_idx is not None:
                correct += int(pred_idx == true_idx)
                top5_correct += int(true_idx in top5_idxs)
                total   += 1
                per_class_correct[true_cls] += int(pred_idx == true_idx)
                per_class_total[true_cls]   += 1

        except Exception as e:
            errors.append(f"{img_path.name}: {e}")

    acc       = correct / total if total > 0 else 0.0
    top5_acc  = top5_correct / total if total > 0 else 0.0
    avg_conf  = float(np.mean(confidences)) if confidences else 0.0
    avg_lat   = float(np.mean(latencies))   if latencies   else 0.0

    per_class_acc = {
        cls: per_class_correct[cls] / per_class_total[cls]
        for cls in per_class_total
    }

    if verbose:
        print(f"\n{'─'*60}")
        print(f"  Images evaluated : {total}")
        print(f"  Top-1 Accuracy   : {acc*100:.2f}%")
        print(f"  Top-5 Accuracy   : {top5_acc*100:.2f}%")
        print(f"  Avg Confidence   : {avg_conf*100:.2f}%")
        print(f"  Avg Latency      : {avg_lat:.1f} ms/image")
        if errors:
            print(f"  Errors ({len(errors)}): {errors[:3]}")
        print(f"\n  Per-class breakdown:")
        for cls, pacc in sorted(per_class_acc.items(), key=lambda x: x[1]):
            n  = per_class_total[cls]
            ok = per_class_correct[cls]
            bar= "█" * int(pacc * 20) + "░" * (20 - int(pacc * 20))
            print(f"    {cls[:35]:<36} [{bar}] {pacc*100:5.1f}%  ({ok}/{n})")

    return {
        "total":           total,
        "top1_acc":        round(acc, 4),
        "top5_acc":        round(top5_acc, 4),
        "avg_confidence":  round(avg_conf, 4),
        "avg_latency_ms":  round(avg_lat, 2),
        "per_class_acc":   {k: round(v, 4) for k, v in per_class_acc.items()},
        "errors":          errors,
    }


# ─────────────────────────────────────────────────────────────────
# Side-by-side comparison table
# ─────────────────────────────────────────────────────────────────
def print_comparison(results_list):
    print(f"\n{'═'*80}")
    print(f"  COMPARISON SUMMARY")
    print(f"{'═'*80}")
    header = f"{'Model':<28} {'Top-1':>7} {'Top-5':>7} {'Conf':>7} {'Latency':>10}"
    print(header)
    print("─"*80)
    for name, res in results_list:
        print(f"  {name:<26} {res['top1_acc']*100:>6.2f}% "
              f"{res['top5_acc']*100:>6.2f}% "
              f"{res['avg_confidence']*100:>6.2f}% "
              f"{res['avg_latency_ms']:>8.1f}ms")
    print("─"*80)

    # Per-class winner table
    all_classes = sorted(set(
        cls for _, r in results_list for cls in r["per_class_acc"]))
    if len(results_list) > 1 and all_classes:
        print(f"\n  PER-CLASS WINNER:")
        for cls in all_classes:
            class_accs = [(n, r["per_class_acc"].get(cls, 0))
                          for n, r in results_list]
            winner     = max(class_accs, key=lambda x: x[1])
            print(f"  {cls[:35]:<36}  → {winner[0]} ({winner[1]*100:.1f}%)")


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate CropGuard .pth models on local image folders")
    parser.add_argument("--model",   type=str, nargs="?",
                        help="Path to a single .pth checkpoint")
    parser.add_argument("--compare", type=str, nargs="+",
                        help="Two or more .pth files to compare side-by-side")
    parser.add_argument("--data",    type=str, default="plantvillage_sample",
                        help="ImageFolder or flat directory of test images")
    parser.add_argument("--output",  type=str, default=None,
                        help="Save results to this JSON file")
    args = parser.parse_args()

    ckpt_paths = []
    if args.compare:
        ckpt_paths = args.compare
    elif args.model:
        ckpt_paths = [args.model]
    else:
        parser.error("Provide --model or --compare")

    samples = discover_images(args.data)
    if not samples:
        print(f"ERROR: No images found in '{args.data}'")
        return

    print(f"\nCropGuard Batch Evaluator")
    print(f"  Data dir : {args.data}  ({len(samples)} images)")
    print(f"  Models   : {ckpt_paths}")

    all_results = []
    for ckpt_path in ckpt_paths:
        print(f"\n{'═'*60}")
        print(f"  Loading: {ckpt_path}")
        model, class_names, img_size, display, trained_val_acc = load_model(ckpt_path)
        print(f"  Architecture : {display}")
        print(f"  Classes      : {len(class_names)}")
        if trained_val_acc:
            print(f"  Training val acc (from checkpoint): {trained_val_acc*100:.2f}%")
        res = evaluate(model, class_names, img_size, samples, verbose=True)
        all_results.append((display, res))

    if len(all_results) > 1:
        print_comparison(all_results)

    if args.output:
        out = {
            "data_dir": args.data,
            "n_images": len(samples),
            "models": [{"name": n, **r} for n, r in all_results],
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"\n✅ Saved report to {args.output}")


if __name__ == "__main__":
    main()
