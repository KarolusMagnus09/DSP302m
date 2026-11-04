import torch
from torchvision.ops import batched_nms, box_convert

from .constants import GROUP_SLICES, PREDICATES


@torch.no_grad()
def postprocess(out, img_sizes, constraint='with', topk=100, nms_iou=0.7):
    """Trả về, cho mỗi ảnh, tối đa topk bộ ba: điểm = điểm_vật * điểm_vị_từ.
    img_sizes: list (h, w) ảnh gốc. constraint='with': mỗi cặp giữ 1 vị từ/nhóm; 'no': giữ hết.
    obj_labels theo chỉ số 36 lớp AG (cộng 1 vì person là 0); chủ thể luôn là person."""
    obj_prob = out['pred_logits'].softmax(-1)[..., :-1]
    obj_score, obj_lab = obj_prob.max(-1)
    preds_all = torch.cat([out['att_logits'].softmax(-1),
                           out['spa_logits'].sigmoid(),
                           out['con_logits'].sigmoid()], dim=-1)       # [B,N,26]
    res = []
    for b, (h, w) in enumerate(img_sizes):
        sc = torch.tensor([w, h, w, h], dtype=torch.float32, device=preds_all.device)
        ob = box_convert(out['pred_boxes'][b].float(), 'cxcywh', 'xyxy') * sc
        pb = box_convert(out['person_box'][b].float(), 'cxcywh', 'xyxy') * sc
        keep = batched_nms(ob, obj_score[b].float(), obj_lab[b], nms_iou)   # đã sắp giảm dần theo điểm
        preds = preds_all[b][keep]
        K = preds.shape[0]
        if constraint == 'with':
            mask = torch.zeros_like(preds, dtype=torch.bool)
            for lo, hi in GROUP_SLICES:
                mask[torch.arange(K, device=preds.device), preds[:, lo:hi].argmax(-1) + lo] = True
            preds = preds * mask
        scores = (obj_score[b][keep][:, None] * preds).flatten()
        top_s, top_i = scores.topk(min(topk, scores.numel()))
        valid = top_s > 0
        top_s, top_i = top_s[valid], top_i[valid]
        qi, pi = top_i // len(PREDICATES), top_i % len(PREDICATES)
        res.append(dict(scores=top_s, predicates=pi, obj_labels=obj_lab[b][keep][qi] + 1,
                        obj_boxes=ob[keep][qi], sub_box=pb[None].expand(len(qi), 4),
                        query_idx=keep[qi]))
    return res
