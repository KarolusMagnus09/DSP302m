"""Loss: thực thể (CE + L1 + GIoU, ghép Hungarian), person box, và quan hệ chỉ trên truy vấn đã ghép.

Target mỗi ảnh (dict):
  person_box [4] cxcywh chuẩn hóa | obj_labels [M] long 0..34 | obj_boxes [M,4]
  att [M] long (0..2, -1 = bỏ qua) | spa [M,6] float multi-hot | con [M,17] float multi-hot
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import box_convert, generalized_box_iou

from .constants import N_OBJ
from .matcher import hungarian_match


def pos_weight_from_counts(pos_counts, n_pairs, power=0.5, max_w=10.0):
    """pos_weight cho BCE từ số lần mỗi lớp dương trong tập train: (neg/pos)^power, kẹp [1, max_w]."""
    pos = torch.as_tensor(pos_counts, dtype=torch.float32).clamp(min=1)
    neg = (n_pairs - pos).clamp(min=1)
    return (neg / pos).pow(power).clamp(1.0, max_w)


def _giou_diag(a, b):
    return torch.diag(generalized_box_iou(box_convert(a, 'cxcywh', 'xyxy'),
                                          box_convert(b, 'cxcywh', 'xyxy')))


class SetCriterion(nn.Module):
    def __init__(self, n_obj=N_OBJ, eos_coef=0.1, weights=None, rel_loss='bce',
                 focal_gamma=2.0, focal_alpha=0.25, pw_spa=None, pw_con=None):
        super().__init__()
        self.n_obj, self.rel_loss = n_obj, rel_loss
        self.gamma, self.alpha = focal_gamma, focal_alpha
        ew = torch.ones(n_obj + 1)
        ew[-1] = eos_coef
        self.register_buffer('empty_weight', ew)
        self.register_buffer('pw_spa', pw_spa)   # có thể None
        self.register_buffer('pw_con', pw_con)
        self.w = dict(ce=1.0, l1=5.0, giou=2.0, pl1=5.0, pgiou=2.0, att=1.0, spa=1.0, con=1.0)
        if weights:
            self.w.update(weights)

    def _entity(self, out, targets, matches, num_obj):
        logits = out['pred_logits']
        B, N, _ = logits.shape
        tgt = torch.full((B, N), self.n_obj, dtype=torch.long, device=logits.device)
        for b, (pi, tj) in enumerate(matches):
            tgt[b, pi] = targets[b]['obj_labels'][tj]
        loss_ce = F.cross_entropy(logits.transpose(1, 2), tgt, self.empty_weight)
        pb = torch.cat([out['pred_boxes'][b][pi] for b, (pi, _) in enumerate(matches)])
        tb = torch.cat([targets[b]['obj_boxes'][tj] for b, (_, tj) in enumerate(matches)])
        l1 = F.l1_loss(pb, tb, reduction='none').sum() / num_obj
        giou = (1 - _giou_diag(pb, tb)).sum() / num_obj if pb.numel() else pb.sum()
        pp = out['person_box']
        pt = torch.stack([t['person_box'] for t in targets])
        hp = torch.tensor([float(t.get('has_person', True)) for t in targets], device=pp.device, dtype=pp.dtype)
        den = hp.sum().clamp(min=1)          # frame không có bbox người: bỏ loss person
        pl1 = (F.l1_loss(pp, pt, reduction='none').sum(-1) * hp).sum() / den
        pgiou = ((1 - _giou_diag(pp, pt)) * hp).sum() / den
        return dict(ce=loss_ce, l1=l1, giou=giou, pl1=pl1, pgiou=pgiou)

    def _multilabel(self, logits, tgt, pw):
        if self.rel_loss == 'bce':
            return F.binary_cross_entropy_with_logits(logits, tgt, pos_weight=pw, reduction='sum')
        p = logits.sigmoid()
        ce = F.binary_cross_entropy_with_logits(logits, tgt, reduction='none')
        pt = p * tgt + (1 - p) * (1 - tgt)
        at = self.alpha * tgt + (1 - self.alpha) * (1 - tgt)
        return (at * (1 - pt) ** self.gamma * ce).sum()

    def _relation(self, out, targets, matches):
        al, sl, cl, at, st, ct = [], [], [], [], [], []
        for b, (pi, tj) in enumerate(matches):
            if len(pi) == 0:
                continue
            al.append(out['att_logits'][b][pi]); at.append(targets[b]['att'][tj])
            sl.append(out['spa_logits'][b][pi]); st.append(targets[b]['spa'][tj])
            cl.append(out['con_logits'][b][pi]); ct.append(targets[b]['con'][tj])
        if not al:
            z = (out['att_logits'].sum() + out['spa_logits'].sum() + out['con_logits'].sum()) * 0
            return dict(att=z, spa=z, con=z)
        al, sl, cl = torch.cat(al), torch.cat(sl), torch.cat(cl)
        at, st, ct = torch.cat(at), torch.cat(st), torch.cat(ct)
        P = al.shape[0]
        return dict(
            att=F.cross_entropy(al, at, ignore_index=-1, reduction='sum') / P,
            spa=self._multilabel(sl, st, self.pw_spa) / P,
            con=self._multilabel(cl, ct, self.pw_con) / P,
        )

    def _match(self, out, targets):
        return [hungarian_match(out['pred_logits'][b], out['pred_boxes'][b],
                                targets[b]['obj_labels'], targets[b]['obj_boxes'])
                for b in range(out['pred_logits'].shape[0])]

    def forward(self, out, targets):
        num_obj = max(sum(len(t['obj_labels']) for t in targets), 1)
        matches = self._match(out, targets)
        losses = self._entity(out, targets, matches, num_obj)
        losses.update(self._relation(out, targets, matches))
        total = sum(self.w[k] * v for k, v in losses.items())
        logs = {k: v.detach() for k, v in losses.items()}
        for l, aux in enumerate(out.get('aux_outputs', [])):
            ent = self._entity(aux, targets, self._match(aux, targets), num_obj)
            total = total + sum(self.w[k] * v for k, v in ent.items())
            logs[f'aux{l}_ce'] = ent['ce'].detach()
        logs['total'] = total.detach()
        return total, logs
