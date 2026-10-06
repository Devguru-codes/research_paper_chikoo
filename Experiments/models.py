"""Model definitions. MT_ST, Baseline1DCNN and BaselineViT1D follow Notebooks/phase_2_notebook.ipynb
(cell 1); the only changes are configurable input length and switches used by the ablations."""
import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class MT_ST(nn.Module):
    """Multi-Task Spectral Transformer: Conv1d patch tokenizer -> [CLS] + sinusoidal PE ->
    pre-LN Transformer encoder -> LayerNorm -> regression head (sugar, carbohydrate) + ripeness head."""

    def __init__(self, input_dim=160, embed_dim=128, num_heads=8, num_layers=4, dropout=0.1, num_classes=3,
                 patch_size=5, use_pe=True):
        super().__init__()
        self.patch_size = patch_size
        self.patch_embed = nn.Conv1d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.num_patches = input_dim // patch_size
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.use_pe = use_pe
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=self.num_patches + 1)
        layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 2,
                                           dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(embed_dim)
        self.reg_head = nn.Sequential(nn.Linear(embed_dim, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 2))
        self.cls_head = nn.Sequential(nn.Linear(embed_dim, 64), nn.GELU(), nn.Dropout(dropout),
                                      nn.Linear(64, num_classes))

    def tokens(self, x):
        x = self.patch_embed(x.unsqueeze(1)).transpose(1, 2)
        x = torch.cat((self.cls_token.expand(x.size(0), -1, -1), x), dim=1)
        return self.pos_encoder(x) if self.use_pe else x

    def forward(self, x):
        out = self.norm(self.transformer(self.tokens(x)))[:, 0, :]
        return self.reg_head(out), self.cls_head(out)

    @torch.no_grad()
    def attention_maps(self, x):
        """Per-layer head-averaged self-attention matrices, computed by re-running each pre-LN layer
        explicitly (eval mode, so dropout is inactive). Returns list of [B, T, T]."""
        h, maps = self.tokens(x), []
        for layer in self.transformer.layers:
            z = layer.norm1(h)
            a, w = layer.self_attn(z, z, z, need_weights=True, average_attn_weights=True)
            h = h + a
            h = h + layer._ff_block(layer.norm2(h))
            maps.append(w)
        return maps

    @torch.no_grad()
    def attention_rollout(self, x):
        """Attention rollout (Abnar & Zuidema, 2020): CLS-to-patch relevance, one value per token."""
        maps = self.attention_maps(x)
        T = maps[0].size(-1)
        eye = torch.eye(T, device=x.device)
        R = eye.expand(x.size(0), T, T)
        for w in maps:
            a = 0.5 * w + 0.5 * eye
            a = a / a.sum(-1, keepdim=True)
            R = a @ R
        return R[:, 0, 1:]


class Baseline1DCNN(nn.Module):
    def __init__(self, input_dim=160, num_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, 5, 2, 2), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, 3, 2, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 3, 2, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1))
        self.reg_head = nn.Linear(128, 2)
        self.cls_head = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x.unsqueeze(1)).flatten(1)
        return self.reg_head(x), self.cls_head(x)


class BaselineViT1D(nn.Module):
    """Single-stream 1D ViT: linear patch embedding + learned positional embedding, post-LN encoder."""

    def __init__(self, input_dim=160, patch_size=5, embed_dim=128, num_heads=4, num_layers=3, num_classes=3):
        super().__init__()
        self.patch_size = patch_size
        n = input_dim // patch_size
        self.patch_embed = nn.Linear(patch_size, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, n + 1, embed_dim))
        layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 2,
                                           batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        self.reg_head = nn.Linear(embed_dim, 2)
        self.cls_head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x.view(B, -1, self.patch_size))
        x = torch.cat((self.cls_token.expand(B, -1, -1), x), dim=1) + self.pos_embed
        x = self.transformer(x)[:, 0, :]
        return self.reg_head(x), self.cls_head(x)
