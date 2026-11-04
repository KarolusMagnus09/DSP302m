"""AGSGG: RelTR-inspired, thiết kế lại cho Action Genome.

  Ảnh -> ResNet50 -> Encoder
      -> Entity decoder: 1 truy vấn 'person' + N truy vấn vật  (box + lớp vật)
      -> Mỗi truy vấn vật j tạo đúng 1 cặp (person, vật j):
         token cặp = MLP([h_person, h_vật, hình học]) + đặc trưng RoI của union box
      -> Relation decoder (self-attn giữa các cặp + cross-attn vào encoder memory)
      -> 3 đầu: attention (softmax, 3) | spatial (sigmoid, 6) | contacting (sigmoid, 17)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops import roi_align, box_convert
from torchvision.ops.misc import FrozenBatchNorm2d

from .constants import N_OBJ, N_ATT, N_SPA, N_CON
from .transformer import Encoder, Decoder, sine_pos_embed


class MLP(nn.Module):  # tên `layers` giống DETR
    def __init__(self, i, h, o, n):
        super().__init__()
        hs = [h] * (n - 1)
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip([i] + hs, hs + [o]))

    def forward(self, x):
        for k, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if k < len(self.layers) - 1 else layer(x)
        return x


class Backbone(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        r = torchvision.models.resnet50(weights=weights, norm_layer=FrozenBatchNorm2d)
        self.body = IntermediateLayerGetter(r, {'layer4': '0'})
        for n, p in self.body.named_parameters():  # đóng băng stem + layer1 như DETR
            if not any(k in n for k in ('layer2', 'layer3', 'layer4')):
                p.requires_grad_(False)

    def forward(self, x, mask):
        f = self.body(x)['0']
        m = F.interpolate(mask[None].float(), size=f.shape[-2:])[0].to(torch.bool)
        return f, m


def pair_geometry(pb, ob, eps=1e-6):
    """pb, ob: [B,N,4] cxcywh chuẩn hóa -> [B,N,6] đặc trưng hình học cặp."""
    px, py, pw, ph = pb.unbind(-1)
    ox, oy, ow, oh = ob.unbind(-1)
    pw, ph, ow, oh = pw.clamp(min=eps), ph.clamp(min=eps), ow.clamp(min=eps), oh.clamp(min=eps)
    dx, dy = (ox - px) / pw, (oy - py) / ph
    dw, dh = torch.log(ow / pw), torch.log(oh / ph)
    pxy, oxy = box_convert(pb, 'cxcywh', 'xyxy'), box_convert(ob, 'cxcywh', 'xyxy')
    lt = torch.max(pxy[..., :2], oxy[..., :2])
    rb = torch.min(pxy[..., 2:], oxy[..., 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    iou = inter / (pw * ph + ow * oh - inter).clamp(min=eps)
    return torch.stack([dx, dy, dw, dh, iou, ow * oh], dim=-1)


class AGSGG(nn.Module):
    def __init__(self, n_obj=N_OBJ, n_obj_queries=30, d=256, nhead=8, enc_layers=6,
                 dec_layers=6, rel_layers=2, dff=2048, dropout=0.1, roi_size=4,
                 backbone_pretrained=False):
        super().__init__()
        self.d, self.n_q, self.roi_size = d, n_obj_queries, roi_size
        self.backbone = Backbone(backbone_pretrained)
        self.input_proj = nn.Conv2d(2048, d, kernel_size=1)
        self.encoder = Encoder(enc_layers, d, nhead, dff, dropout)
        self.query_embed = nn.Embedding(1 + n_obj_queries, d)  # truy vấn 0 = person
        self.entity_decoder = Decoder(dec_layers, d, nhead, dff, dropout, return_intermediate=True)
        self.box_head = MLP(d, d, 4, 3)            # dùng chung cho person và vật
        self.cls_head = nn.Linear(d, n_obj + 1)    # +1 = no-object
        # nhánh quan hệ
        self.geom_proj = nn.Linear(6, d)
        self.pair_fuse = MLP(3 * d, d, d, 2)
        self.roi_proj = nn.Linear(d * roi_size * roi_size, d)
        self.pair_norm = nn.LayerNorm(d)
        self.rel_decoder = Decoder(rel_layers, d, nhead, dff, dropout, return_intermediate=False)
        self.att_head = nn.Linear(d, N_ATT)
        self.spa_head = nn.Linear(d, N_SPA)
        self.con_head = nn.Linear(d, N_CON)

    def forward(self, images, mask):
        """images [B,3,H,W]; mask [B,H,W] bool (True = padding, pad ở dưới/phải)."""
        feat, m = self.backbone(images, mask)
        pos = sine_pos_embed(m, self.d).to(feat.dtype)
        src = self.input_proj(feat)                         # [B,d,h,w]
        B = src.shape[0]
        src_seq = src.flatten(2).permute(2, 0, 1)           # [hw,B,d]
        pos_seq = pos.flatten(2).permute(2, 0, 1)
        kpm = m.flatten(1)
        memory = self.encoder(src_seq, kpm, pos_seq)

        q = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)       # [1+N,B,d]
        hs = self.entity_decoder(torch.zeros_like(q), memory, kpm, pos_seq, q)
        hs = hs.transpose(1, 2)                                         # [L,B,1+N,d]
        boxes = self.box_head(hs).sigmoid()                             # [L,B,1+N,4]
        logits = self.cls_head(hs[:, :, 1:])                            # [L,B,N,C+1]

        # ---- quan hệ theo cặp (person, vật j) ----
        N = self.n_q
        hp = hs[-1][:, :1].expand(-1, N, -1)
        ho = hs[-1][:, 1:]
        pb = boxes[-1][:, :1].detach().expand(-1, N, -1)
        ob = boxes[-1][:, 1:].detach()
        geom = pair_geometry(pb, ob)

        pxy, oxy = box_convert(pb, 'cxcywh', 'xyxy'), box_convert(ob, 'cxcywh', 'xyxy')
        ub = torch.cat([torch.min(pxy[..., :2], oxy[..., :2]),
                        torch.max(pxy[..., 2:], oxy[..., 2:])], dim=-1).clamp(0, 1)
        vh = (~m[:, :, 0]).sum(1).to(ub.dtype)     # số hàng/cột hợp lệ của feature map
        vw = (~m[:, 0, :]).sum(1).to(ub.dtype)
        scale = torch.stack([vw, vh, vw, vh], dim=1)[:, None, :]
        rois = ub * scale
        bidx = torch.arange(B, device=rois.device, dtype=rois.dtype)[:, None, None].expand(B, N, 1)
        rois = torch.cat([bidx, rois], dim=-1).reshape(B * N, 5)
        roi = roi_align(src, rois.to(src.dtype), output_size=self.roi_size,
                        spatial_scale=1.0, sampling_ratio=2, aligned=True)   # [B*N,d,r,r]
        roi_feat = self.roi_proj(roi.flatten(1)).view(B, N, -1)

        tok = self.pair_fuse(torch.cat([hp, ho, self.geom_proj(geom)], dim=-1)) + roi_feat
        tok = self.pair_norm(tok).transpose(0, 1)                       # [N,B,d]
        rel = self.rel_decoder(tok, memory, kpm, pos_seq, torch.zeros_like(tok))[-1]
        rel = rel.transpose(0, 1)                                       # [B,N,d]

        return {
            'person_box': boxes[-1][:, 0],
            'pred_boxes': boxes[-1][:, 1:],
            'pred_logits': logits[-1],
            'att_logits': self.att_head(rel),
            'spa_logits': self.spa_head(rel),
            'con_logits': self.con_head(rel),
            'aux_outputs': [{'person_box': boxes[l][:, 0], 'pred_boxes': boxes[l][:, 1:],
                             'pred_logits': logits[l]} for l in range(boxes.shape[0] - 1)],
        }


def load_detr_weights(model, ckpt_path):
    """Nạp backbone + encoder + decoder + box head từ checkpoint DETR-COCO (lọc theo tên và shape).
    Chỉ dùng với file tin cậy (weights_only=False). Trả về số tensor đã nạp."""
    sd = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = sd.get('model', sd)
    mine = model.state_dict()
    rules = [('backbone.0.body.', 'backbone.body.'), ('transformer.encoder.', 'encoder.'),
             ('transformer.decoder.', 'entity_decoder.'), ('bbox_embed.', 'box_head.'),
             ('input_proj.', 'input_proj.')]
    new = {}
    for k, v in sd.items():
        for a, b in rules:
            if k.startswith(a):
                nk = b + k[len(a):]
                if nk in mine and mine[nk].shape == v.shape:
                    new[nk] = v
                break
    model.load_state_dict(new, strict=False)
    return len(new)
