"""Các khối Transformer kiểu DETR (đặt tên tham số giống DETR để nạp trọng số COCO)."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def sine_pos_embed(mask, d_model, temperature=10000):
    """mask [B,H,W] (True = padding) -> [B,d_model,H,W]. Giống DETR."""
    not_mask = ~mask
    y = not_mask.cumsum(1, dtype=torch.float32)
    x = not_mask.cumsum(2, dtype=torch.float32)
    eps, scale = 1e-6, 2 * math.pi
    y = y / (y[:, -1:, :] + eps) * scale
    x = x / (x[:, :, -1:] + eps) * scale
    npf = d_model // 2
    dim_t = torch.arange(npf, dtype=torch.float32, device=mask.device)
    dim_t = temperature ** (2 * (dim_t // 2) / npf)
    px = x[:, :, :, None] / dim_t
    py = y[:, :, :, None] / dim_t
    px = torch.stack((px[:, :, :, 0::2].sin(), px[:, :, :, 1::2].cos()), dim=4).flatten(3)
    py = torch.stack((py[:, :, :, 0::2].sin(), py[:, :, :, 1::2].cos()), dim=4).flatten(3)
    return torch.cat((py, px), dim=3).permute(0, 3, 1, 2)


class EncoderLayer(nn.Module):
    def __init__(self, d, nhead, dff, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d, dff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dff, d)
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.dropout1, self.dropout2 = nn.Dropout(dropout), nn.Dropout(dropout)

    def forward(self, src, key_padding_mask, pos):
        q = k = src + pos
        src2 = self.self_attn(q, k, value=src, key_padding_mask=key_padding_mask)[0]
        src = self.norm1(src + self.dropout1(src2))
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        return self.norm2(src + self.dropout2(src2))


class Encoder(nn.Module):
    def __init__(self, num_layers, d, nhead, dff, dropout):
        super().__init__()
        self.layers = nn.ModuleList(EncoderLayer(d, nhead, dff, dropout) for _ in range(num_layers))

    def forward(self, src, key_padding_mask, pos):
        for layer in self.layers:
            src = layer(src, key_padding_mask, pos)
        return src


class DecoderLayer(nn.Module):
    def __init__(self, d, nhead, dff, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d, dff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dff, d)
        self.norm1, self.norm2, self.norm3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.dropout1, self.dropout2, self.dropout3 = nn.Dropout(dropout), nn.Dropout(dropout), nn.Dropout(dropout)

    def forward(self, tgt, memory, memory_key_padding_mask, pos, query_pos):
        q = k = tgt + query_pos
        tgt2 = self.self_attn(q, k, value=tgt)[0]
        tgt = self.norm1(tgt + self.dropout1(tgt2))
        tgt2 = self.multihead_attn(query=tgt + query_pos, key=memory + pos, value=memory,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = self.norm2(tgt + self.dropout2(tgt2))
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        return self.norm3(tgt + self.dropout3(tgt2))


class Decoder(nn.Module):
    def __init__(self, num_layers, d, nhead, dff, dropout, return_intermediate):
        super().__init__()
        self.layers = nn.ModuleList(DecoderLayer(d, nhead, dff, dropout) for _ in range(num_layers))
        self.norm = nn.LayerNorm(d)
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory, memory_key_padding_mask, pos, query_pos):
        out, inter = tgt, []
        for layer in self.layers:
            out = layer(out, memory, memory_key_padding_mask, pos, query_pos)
            if self.return_intermediate:
                inter.append(self.norm(out))
        if self.return_intermediate:
            return torch.stack(inter)          # [L, Q, B, d]
        return self.norm(out).unsqueeze(0)     # [1, Q, B, d]
