# convert_direct.py
import torch
import torch.nn as nn
import timm
import ai_edge_torch

model = timm.create_model('mobilenetv3_small_100', pretrained=False, num_classes=38)
checkpoint = torch.load('best_model.pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

class MobileNetV3Wrapper(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.conv_stem  = base.conv_stem
        self.bn1        = base.bn1
        self.blocks     = base.blocks
        self.conv_head  = base.conv_head
        self.norm_head  = base.norm_head
        self.act2       = base.act2
        self.classifier = base.classifier
        self.pool       = nn.AvgPool2d(kernel_size=7, stride=1)

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.blocks(x)
        x = self.pool(x)
        x = self.conv_head(x)
        x = self.norm_head(x)
        x = self.act2(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

wrapped = MobileNetV3Wrapper(model)
wrapped.eval()

sample = (torch.randn(1, 3, 224, 224),)

with torch.no_grad():
    out = wrapped(*sample)
    print(f"Wrapper output shape: {out.shape}")  # must be [1, 38]

print("Converting...")
edge_model = ai_edge_torch.convert(wrapped, sample)
edge_model.export("cropguard_direct.tflite")
print("✅ Done — cropguard_direct.tflite ready")