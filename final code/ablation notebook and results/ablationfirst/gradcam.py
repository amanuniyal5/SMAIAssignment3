"""
gradcam.py — GradCAM Explainability for CropGuard AI
─────────────────────────────────────────────────────
Provides GradCAM heatmaps showing WHICH leaf region triggered
the model's disease prediction.

Usage (standalone):
    python gradcam.py --image path/to/leaf.jpg --model best_model.pth

Usage (as module):
    from gradcam import GradCAM, overlay_heatmap
    gcam = GradCAM(model, target_layer)
    heatmap = gcam(input_tensor, class_idx)
    overlay = overlay_heatmap(original_pil_image, heatmap)
"""

import argparse
import json
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.nn.functional as F
import timm
from torchvision import transforms


# ─────────────────────────────────────────────────────────────────
# Core GradCAM implementation
# ─────────────────────────────────────────────────────────────────
class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Selvaraju et al. 2017).
    Works with any timm CNN model by hooking the last convolutional layer.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model        = model
        self.target_layer = target_layer
        self._features    = None
        self._gradients   = None
        self._fwd_hook    = target_layer.register_forward_hook(self._save_features)
        self._bwd_hook    = target_layer.register_full_backward_hook(self._save_grads)

    def _save_features(self, module, input, output):
        self._features = output.detach()

    def _save_grads(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def __call__(self, x: torch.Tensor, class_idx: int = None) -> np.ndarray:
        """
        Args:
            x:          (1, 3, H, W) input tensor
            class_idx:  class to explain; if None uses argmax (top prediction)
        Returns:
            heatmap: (H, W) numpy array in [0, 1]
        """
        self.model.eval()
        x = x.requires_grad_(True)

        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        self.model.zero_grad()
        logits[0, class_idx].backward()

        # Global average pool the gradients
        weights   = self._gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam       = (weights * self._features).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam       = F.relu(cam)

        # Normalise to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)

        # Upsample to input resolution
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear",
                            align_corners=False)
        return cam.squeeze().cpu().numpy()

    def remove_hooks(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()


# ─────────────────────────────────────────────────────────────────
# Heatmap rendering
# ─────────────────────────────────────────────────────────────────
def overlay_heatmap(
    original_img: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap: str = "jet"
) -> Image.Image:
    """
    Blends a GradCAM heatmap onto the original leaf image.

    Args:
        original_img: PIL RGB image (any size)
        heatmap:      2D numpy array in [0, 1]
        alpha:        heatmap opacity (0=invisible, 1=full overlay)
        colormap:     matplotlib colormap name

    Returns:
        PIL RGB image with heatmap overlay
    """
    import matplotlib.cm as cm

    # Resize heatmap to match image
    h_pil = Image.fromarray(np.uint8(heatmap * 255)).resize(
        original_img.size, Image.BILINEAR)
    h_arr = np.array(h_pil) / 255.0

    # Apply colormap
    cmap      = cm.get_cmap(colormap)
    heat_rgb  = cmap(h_arr)[:, :, :3]          # (H, W, 3) in [0,1]
    heat_uint = (heat_rgb * 255).astype(np.uint8)
    heat_pil  = Image.fromarray(heat_uint, "RGB")

    # Blend
    orig_arr   = np.array(original_img.convert("RGB"), dtype=np.float32)
    heat_arr   = np.array(heat_pil,                   dtype=np.float32)
    blended    = np.clip(orig_arr * (1 - alpha) + heat_arr * alpha, 0, 255)
    return Image.fromarray(blended.astype(np.uint8))


def make_side_by_side(original: Image.Image, overlay: Image.Image,
                      label: str = "") -> Image.Image:
    """Returns a 2-panel image: original | GradCAM overlay."""
    W, H = original.size
    canvas = Image.new("RGB", (W * 2 + 8, H + 30), (245, 245, 245))
    canvas.paste(original.resize((W, H)), (0, 30))
    canvas.paste(overlay.resize((W, H)),  (W + 8, 30))

    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    draw.text((4,   4), "Original",        fill=(60, 60, 60), font=font)
    draw.text((W+12, 4), f"GradCAM · {label}", fill=(60, 60, 60), font=font)
    return canvas


# ─────────────────────────────────────────────────────────────────
# Auto-detect best target layer for common timm architectures
# ─────────────────────────────────────────────────────────────────
def get_target_layer(model: torch.nn.Module, timm_name: str) -> torch.nn.Module:
    """
    Returns the last convolutional layer appropriate for GradCAM
    for the given timm model architecture.
    """
    name = timm_name.lower()

    if "mobilenetv3" in name:
        # Last conv before adaptive pool
        return model.blocks[-1][-1].conv_dw if hasattr(model, "blocks") \
               else list(model.children())[-3]

    elif "efficientnet" in name:
        return model.blocks[-1][-1]

    elif "resnet" in name:
        return model.layer4[-1]

    elif "vit" in name or "deit" in name:
        # For ViT, GradCAM applies to the last attention block's norm
        return model.blocks[-1].norm1

    else:
        # Generic fallback: last non-linear layer
        candidates = [(n, m) for n, m in model.named_modules()
                      if isinstance(m, (torch.nn.Conv2d, torch.nn.BatchNorm2d))]
        if candidates:
            return candidates[-1][1]
        raise ValueError(f"Cannot auto-detect target layer for {timm_name}")


# ─────────────────────────────────────────────────────────────────
# Full pipeline: image → GradCAM overlay PIL
# ─────────────────────────────────────────────────────────────────
def explain_image(
    img_pil: Image.Image,
    model: torch.nn.Module,
    timm_name: str,
    class_idx: int = None,
    img_size: int = 224,
    alpha: float = 0.45,
) -> tuple[Image.Image, np.ndarray]:
    """
    End-to-end: PIL image → GradCAM overlay PIL image + raw heatmap.
    Compatible with Streamlit (returns PIL).

    Returns:
        overlay_pil:  PIL image with heatmap blended in
        heatmap:      raw (H, W) numpy array in [0, 1]
    """
    tfm = transforms.Compose([
        transforms.Resize(int(img_size * 256 / 224)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    x = tfm(img_pil.convert("RGB")).unsqueeze(0)

    try:
        target_layer = get_target_layer(model, timm_name)
        gcam         = GradCAM(model, target_layer)
        heatmap      = gcam(x, class_idx)
        gcam.remove_hooks()
    except Exception as e:
        print(f"GradCAM failed for {timm_name}: {e}")
        heatmap = np.zeros((img_size, img_size))

    # Resize original to match processed size for display
    display_img  = img_pil.convert("RGB").resize((img_size, img_size))
    overlay_pil  = overlay_heatmap(display_img, heatmap, alpha=alpha)
    return overlay_pil, heatmap


# ─────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate GradCAM explanation for a leaf image")
    parser.add_argument("--image",  required=True, help="Path to leaf image")
    parser.add_argument("--model",  default="best_model.pth",
                        help="Path to .pth checkpoint")
    parser.add_argument("--class_idx", type=int, default=None,
                        help="Class index to explain (default: top prediction)")
    parser.add_argument("--output", default="gradcam_output.png",
                        help="Output image path")
    parser.add_argument("--alpha",  type=float, default=0.45,
                        help="Heatmap opacity 0-1")
    args = parser.parse_args()

    # Load checkpoint
    ckpt        = torch.load(args.model, map_location="cpu", weights_only=False)
    timm_name   = ckpt.get("timm_name", "mobilenetv3_small_100")
    class_names = ckpt["class_names"]
    num_classes = len(class_names)
    img_size    = ckpt.get("img_size", 224)

    net = timm.create_model(timm_name, pretrained=False, num_classes=num_classes)
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()

    img = Image.open(args.image)

    # Run inference first to get top prediction
    tfm = transforms.Compose([
        transforms.Resize(int(img_size * 256 / 224)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    with torch.no_grad():
        logits  = net(tfm(img.convert("RGB")).unsqueeze(0))[0]
        probs   = torch.softmax(logits, 0)
        top_idx = probs.argmax().item()
        top_cls = class_names[top_idx]
        top_conf= probs[top_idx].item()

    class_idx = args.class_idx if args.class_idx is not None else top_idx
    print(f"Top prediction: {top_cls}  ({top_conf*100:.1f}%)")
    print(f"Generating GradCAM for class idx {class_idx}: {class_names[class_idx]}")

    overlay, _ = explain_image(img, net, timm_name,
                               class_idx=class_idx,
                               img_size=img_size,
                               alpha=args.alpha)
    panel = make_side_by_side(img.resize((img_size, img_size)), overlay,
                              label=class_names[class_idx].replace("_", " "))
    panel.save(args.output)
    print(f"✅ Saved: {args.output}")
