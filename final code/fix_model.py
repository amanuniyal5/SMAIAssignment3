import torch
import torch.nn as nn
import timm

print("Loading model architecture using timm...")
model = timm.create_model('mobilenetv3_small_100', pretrained=False, num_classes=38)

print("Loading checkpoint weights...")
checkpoint = torch.load('best_model.pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

class MobileNetV3Wrapper(nn.Module):
    def __init__(self, base):
        super().__init__()
        # Exact names from your printed children:
        # conv_stem, bn1, blocks, global_pool, conv_head, norm_head, act2, flatten, classifier
        self.conv_stem  = base.conv_stem
        self.bn1        = base.bn1
        self.blocks     = base.blocks
        self.conv_head  = base.conv_head
        self.norm_head  = base.norm_head
        self.act2       = base.act2
        self.classifier = base.classifier
        # Static pool replaces base.global_pool and base.flatten
        self.pool       = nn.AvgPool2d(kernel_size=7, stride=1)

    def forward(self, x):
        x = self.conv_stem(x)   # [B, 16, 112, 112]
        x = self.bn1(x)
        x = self.blocks(x)      # [B, 96, 7, 7]
        x = self.pool(x)        # [B, 96, 1, 1]  ← static, no dynamic reshape
        x = self.conv_head(x)   # [B, 576, 1, 1]
        x = self.norm_head(x)
        x = self.act2(x)
        x = x.view(x.size(0), -1)  # [B, 576] ← explicit flatten, clean in ONNX
        x = self.classifier(x)     # [B, 38]
        return x

wrapped = MobileNetV3Wrapper(model)
wrapped.eval()

dummy = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    out = wrapped(dummy)
    print(f"Wrapper output shape: {out.shape}")  # must be torch.Size([1, 38])

torch.onnx.export(
    wrapped,
    dummy,
    "fixed_model.onnx",
    opset_version=12,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    do_constant_folding=True,
)
print("Success! 'fixed_model.onnx' ready for onnx2tf.")