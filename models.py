import torch
import torch.nn as nn
import torchvision.models as models
import timm

def get_mobilenet(num_classes, pretrained=True):
    # Use torchvision's mobilenet_v2 as example
    model = models.mobilenet_v2(pretrained=pretrained)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_features, num_classes)
    )
    return model

class SimpleViT(nn.Module):
    def __init__(self, num_classes, vit_name='vit_base_patch16_224', pretrained=True):
        super().__init__()
        # using timm for vision transformer backbones
        self.backbone = timm.create_model(vit_name, pretrained=pretrained, num_classes=num_classes)
    def forward(self, x):
        return self.backbone(x)
