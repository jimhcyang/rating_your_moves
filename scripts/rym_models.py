#!/usr/bin/env python3
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# All models expect input: (B, C, 8, 8), where C = NUM_PLANES from ply_features.


# ---------------------------------------------------------------------
# Utility blocks
# ---------------------------------------------------------------------


class FlattenHead(nn.Module):
    """Flatten + single hidden layer + dual heads (classification + regression)."""

    def __init__(self, in_dim: int, hidden_dim: int, num_bins: int) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.cls_head = nn.Linear(hidden_dim, num_bins)
        self.reg_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor):
        h = self.fc(x)
        logits = self.cls_head(h)
        rating = self.reg_head(h).squeeze(-1)
        return logits, rating


class GlobalAvgPool(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> (B, C)
        return x.mean(dim=(2, 3))


class ResBlock(nn.Module):
    """Basic residual block: Conv-BN-ReLU-Conv-BN + skip."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + residual
        out = F.relu(out, inplace=True)
        return out


# ---------------------------------------------------------------------
# Linear model
# ---------------------------------------------------------------------


class LinearRatingModel(nn.Module):
    """Flatten 8x8xC into a vector, one hidden layer, dual heads.

    Very fast baseline.
    """

    def __init__(self, num_planes: int, num_bins: int, hidden_dim: int = 512) -> None:
        super().__init__()
        in_dim = num_planes * 8 * 8
        self.flatten = nn.Flatten()
        self.head = FlattenHead(in_dim, hidden_dim, num_bins)

    def forward(self, x: torch.Tensor):
        x_flat = self.flatten(x)
        return self.head(x_flat)


# ---------------------------------------------------------------------
# MLP model
# ---------------------------------------------------------------------


class MLPRatingModel(nn.Module):
    """Flatten then multi-layer perceptron, ignoring spatial structure."""

    def __init__(
        self,
        num_planes: int,
        num_bins: int,
        hidden_dims: Tuple[int, int] = (512, 512),
    ) -> None:
        super().__init__()
        in_dim = num_planes * 8 * 8
        layers = []
        prev_dim = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU(inplace=True))
            prev_dim = h
        self.mlp = nn.Sequential(*layers)
        self.cls_head = nn.Linear(prev_dim, num_bins)
        self.reg_head = nn.Linear(prev_dim, 1)

    def forward(self, x: torch.Tensor):
        x_flat = x.view(x.size(0), -1)
        h = self.mlp(x_flat)
        logits = self.cls_head(h)
        rating = self.reg_head(h).squeeze(-1)
        return logits, rating


# ---------------------------------------------------------------------
# CNN model
# ---------------------------------------------------------------------


class CNNRatingModel(nn.Module):
    """Simple 4-layer CNN with global average pooling + dual heads."""

    def __init__(self, num_planes: int, num_bins: int, base_channels: int = 64) -> None:
        super().__init__()
        C = num_planes
        self.conv = nn.Sequential(
            nn.Conv2d(C, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.pool = GlobalAvgPool()
        self.cls_head = nn.Linear(base_channels, num_bins)
        self.reg_head = nn.Linear(base_channels, 1)

    def forward(self, x: torch.Tensor):
        h = self.conv(x)
        h = self.pool(h)
        logits = self.cls_head(h)
        rating = self.reg_head(h).squeeze(-1)
        return logits, rating


# ---------------------------------------------------------------------
# ResNet-style CNN
# ---------------------------------------------------------------------


class ResNetRatingModel(nn.Module):
    """Conv stem + several residual blocks + GAP + dual heads."""

    def __init__(
        self,
        num_planes: int,
        num_bins: int,
        channels: int = 128,
        num_blocks: int = 5,
    ) -> None:
        super().__init__()
        C = num_planes
        self.stem = nn.Sequential(
            nn.Conv2d(C, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[ResBlock(channels) for _ in range(num_blocks)])
        self.pool = GlobalAvgPool()
        self.cls_head = nn.Linear(channels, num_bins)
        self.reg_head = nn.Linear(channels, 1)

    def forward(self, x: torch.Tensor):
        h = self.stem(x)
        h = self.blocks(h)
        h = self.pool(h)
        logits = self.cls_head(h)
        rating = self.reg_head(h).squeeze(-1)
        return logits, rating


# ---------------------------------------------------------------------
# Conv + Transformer
# ---------------------------------------------------------------------


class ConvTransformerRatingModel(nn.Module):
    """Conv stem → 64 tokens → Transformer encoder → mean-pool → dual heads."""

    def __init__(
        self,
        num_planes: int,
        num_bins: int,
        conv_channels: int = 128,
        d_model: int = 128,
        n_heads: int = 4,
        num_layers: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        C = num_planes

        # Conv stem: (B, C, 8, 8) → (B, conv_channels, 8, 8)
        self.conv_stem = nn.Sequential(
            nn.Conv2d(C, conv_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(conv_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv_channels, conv_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(conv_channels),
            nn.ReLU(inplace=True),
        )

        # Project conv channels to d_model
        self.proj = nn.Conv2d(conv_channels, d_model, kernel_size=1)

        # Learned positional embeddings for 64 squares
        self.pos_embed = nn.Parameter(torch.zeros(1, 64, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_head = nn.Linear(d_model, num_bins)
        self.reg_head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor):
        # x: (B, C, 8, 8)
        h = self.conv_stem(x)   # (B, conv_channels, 8, 8)
        h = self.proj(h)        # (B, d_model, 8, 8)

        B, D, H, W = h.shape
        h = h.view(B, D, H * W).transpose(1, 2)  # (B, 64, D)
        h = h + self.pos_embed[:, : h.size(1), :]

        h = self.encoder(h)     # (B, 64, D)
        h_pool = h.mean(dim=1)  # (B, D)

        logits = self.cls_head(h_pool)
        rating = self.reg_head(h_pool).squeeze(-1)
        return logits, rating


# ---------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------


MODEL_TYPES = ("linear", "mlp", "cnn", "resnet", "conv_transformer")


def get_model(
    model_type: str,
    num_planes: int,
    num_bins: int,
    config_id: int = 0,
) -> nn.Module:
    """Build a model given family name and small discrete config_id (0..3)."""
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Unknown model_type={model_type}, expected {MODEL_TYPES}")

    cfg = config_id

    if model_type == "linear":
        hidden_choices = [128, 256, 512, 1024]
        hidden = hidden_choices[cfg % len(hidden_choices)]
        return LinearRatingModel(num_planes, num_bins, hidden_dim=hidden)

    if model_type == "mlp":
        variants: Tuple[Tuple[int, int], ...] = (
            (128, 128),
            (256, 256),
            (512, 512),
            (1024, 1024),
        )
        hidden_dims = variants[cfg % len(variants)]
        return MLPRatingModel(num_planes, num_bins, hidden_dims=hidden_dims)

    if model_type == "cnn":
        base_channels = [32, 64, 128, 256][cfg % 4]
        return CNNRatingModel(num_planes, num_bins, base_channels=base_channels)

    if model_type == "resnet":
        variants: Tuple[Tuple[int, int], ...] = (
            (64, 2),
            (64, 3),
            (128, 4),
            (256, 4),
        )
        channels, blocks = variants[cfg % len(variants)]
        return ResNetRatingModel(num_planes, num_bins, channels=channels, num_blocks=blocks)

    if model_type == "conv_transformer":
        variants = (
            dict(conv_channels=64, d_model=64, n_heads=4, num_layers=2),
            dict(conv_channels=64, d_model=64, n_heads=4, num_layers=3),
            dict(conv_channels=128, d_model=128, n_heads=4, num_layers=4),
            dict(conv_channels=256, d_model=256, n_heads=8, num_layers=4),
        )
        v = variants[cfg % len(variants)]
        return ConvTransformerRatingModel(
            num_planes=num_planes,
            num_bins=num_bins,
            **v,
        )

    # Should be unreachable
    raise AssertionError("Unknown model_type")