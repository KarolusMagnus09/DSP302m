import torch
from scipy.optimize import linear_sum_assignment
from torchvision.ops import box_convert, generalized_box_iou


@torch.no_grad()
def hungarian_match(logits, boxes, tgt_labels, tgt_boxes, w=(1.0, 5.0, 2.0)):
    """Ghép các truy vấn vật với vật nhãn thật (person không cần ghép).
    logits [N,C+1], boxes [N,4] cxcywh; tgt_labels [M], tgt_boxes [M,4]. Trả (pred_idx, tgt_idx)."""
    dev = logits.device
    if tgt_labels.numel() == 0:
        e = torch.empty(0, dtype=torch.long, device=dev)
        return e, e
    prob = logits.softmax(-1)
    c_cls = -prob[:, tgt_labels]
    c_l1 = torch.cdist(boxes, tgt_boxes, p=1)
    c_giou = -generalized_box_iou(box_convert(boxes, 'cxcywh', 'xyxy'),
                                  box_convert(tgt_boxes, 'cxcywh', 'xyxy'))
    cost = w[0] * c_cls + w[1] * c_l1 + w[2] * c_giou
    cost = torch.nan_to_num(cost.float(), nan=1e4, posinf=1e4, neginf=-1e4)   # dự đoán NaN/inf không làm sập matcher
    i, j = linear_sum_assignment(cost.cpu().numpy())
    return torch.as_tensor(i, dtype=torch.long, device=dev), torch.as_tensor(j, dtype=torch.long, device=dev)
