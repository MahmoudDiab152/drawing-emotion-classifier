import torch
import torch.nn as nn
from torchvision import models

FEAT_DIM   = 79
FUSION_DIM = 256


class AttentionFusion(nn.Module):
    def __init__(self, channel_dims, fusion_dim=FUSION_DIM):
        super().__init__()
        self.projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d, fusion_dim),
                nn.BatchNorm1d(fusion_dim),
                nn.ReLU(),
            ) for d in channel_dims
        ])
        self.attention = nn.Sequential(
            nn.Linear(fusion_dim * len(channel_dims), len(channel_dims)),
            nn.Softmax(dim=1),
        )

    def forward(self, *inputs):
        projected = [p(x) for p, x in zip(self.projectors, inputs)]
        alpha     = self.attention(torch.cat(projected, dim=1)).unsqueeze(-1)
        return (alpha * torch.stack(projected, dim=1)).sum(dim=1)


class EmotionModel(nn.Module):
    def __init__(self, num_classes: int, feat_dim: int = FEAT_DIM, fusion_dim: int = FUSION_DIM):
        super().__init__()
        backbone = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
        for i, child in enumerate(backbone.features.children()):
            if i < 5:
                for p in child.parameters():
                    p.requires_grad = False
        cnn_dim = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
        self.backbone   = backbone
        self.fusion     = AttentionFusion([cnn_dim, feat_dim], fusion_dim)
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, img: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.fusion(self.backbone(img), feats))
