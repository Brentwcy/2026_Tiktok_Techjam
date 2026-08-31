"""
Binary AIGC-vs-real classifier built on a pretrained timm backbone.

Default backbone is EfficientNet-B0, matching the reported trained model and
providing a strong speed/accuracy tradeoff for time-constrained fine-tuning.
Use --backbone to select another timm image-classification backbone.
"""

import timm
import torch
import torch.nn as nn


class AIGCDetector(nn.Module):
    def __init__(self, backbone_name: str = "efficientnet_b0", pretrained: bool = True,
                 freeze_backbone: bool = False):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        feat_dim = self.backbone.num_features
        self.head = nn.Linear(feat_dim, 1)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        logit = self.head(feats).squeeze(-1)
        return logit  # raw logit; apply sigmoid outside for probability

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True


def build_model(backbone_name: str = "efficientnet_b0", pretrained: bool = True,
                 freeze_backbone: bool = False) -> AIGCDetector:
    return AIGCDetector(backbone_name=backbone_name, pretrained=pretrained, freeze_backbone=freeze_backbone)
