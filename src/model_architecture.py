"""
Model definitions:
- Image encoder: DenseNet-121 pretrained on ImageNet (torchvision)
- Metadata encoder: simple MLP
- Fusion: concatenate pooled image features with metadata embedding, final classifier
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from . import config


class MetadataMLP(nn.Module):
    def __init__(self, in_features: int, hidden: int = 64, out_features: int = 128, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_features),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class DenseNet121FeatureExtractor(nn.Module):
    def __init__(self, pretrained: bool = True, weights=None):
        """
        Loads DenseNet121 and returns feature vector after global pooling.
        Supports newer torchvision 'weights' argument while maintaining compatibility.
        """
        super().__init__()
        # Use torchvision models API: prefer weights enum if available
        try:
            if weights is not None:
                densenet = models.densenet121(weights=weights)
            else:
                densenet = models.densenet121(pretrained=pretrained)
        except TypeError:
            # Older torchvision fallback
            densenet = models.densenet121(pretrained=pretrained)

        # Remove classifier, keep features
        self.features = densenet.features
        # We will perform adaptive avg pooling to fixed vector
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_dim = densenet.classifier.in_features

    def forward(self, x):
        x = self.features(x)
        x = F.relu(x, inplace=True)
        x = self.pool(x).view(x.size(0), -1)
        return x


class FusionModel(nn.Module):
    def __init__(self, metadata_dim: int, metadata_emb: int = 128, hidden: int = 256, dropout: float = 0.3, pretrained_image: bool = True):
        super().__init__()
        # Image feature extractor
        self.image_encoder = DenseNet121FeatureExtractor(pretrained=pretrained_image)
        image_feat_dim = self.image_encoder.out_dim

        # Metadata encoder
        self.metadata_encoder = MetadataMLP(in_features=metadata_dim, out_features=metadata_emb)

        # Fusion classifier
        fused_dim = image_feat_dim + metadata_emb
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),  # binary classification logit
        )

    def forward(self, images, metadata):
        image_feats = self.image_encoder(images)  # (B, image_feat_dim)
        meta_feats = self.metadata_encoder(metadata)  # (B, metadata_emb)
        fused = torch.cat([image_feats, meta_feats], dim=1)
        logits = self.classifier(fused).squeeze(1)
        prob = torch.sigmoid(logits)
        return prob, logits
