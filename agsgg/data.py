import torch
from torch.utils.data import Dataset


def collate_fn(batch):
    """batch: list (image [3,H,W] đã chuẩn hóa, target dict có thêm 'size' = (h, w)).
    Pad xuống dưới/phải; trả (images, mask, targets). Box trong target là cxcywh chuẩn hóa
    theo kích thước ảnh KHÔNG pad."""
    imgs, tgts = zip(*batch)
    H = max(i.shape[1] for i in imgs)
    W = max(i.shape[2] for i in imgs)
    x = imgs[0].new_zeros(len(imgs), 3, H, W)
    mask = torch.ones(len(imgs), H, W, dtype=torch.bool)
    for k, i in enumerate(imgs):
        h, w = i.shape[1:]
        x[k, :, :h, :w] = i
        mask[k, :h, :w] = False
    return x, mask, list(tgts)


class SyntheticDataset(Dataset):
    """Dữ liệu ngẫu nhiên để kiểm tra pipeline chạy được (không có ý nghĩa học thuật)."""

    def __init__(self, n=16, size=(192, 256), max_obj=3, seed=0):
        self.n, self.size, self.max_obj = n, size, max_obj
        self.g = torch.Generator().manual_seed(seed)

    def __len__(self):
        return self.n

    def _box(self):
        c = 0.3 + 0.4 * torch.rand(2, generator=self.g)
        wh = 0.1 + 0.2 * torch.rand(2, generator=self.g)
        return torch.cat([c, wh])

    def __getitem__(self, idx):
        h, w = self.size
        img = torch.randn(3, h, w, generator=self.g)
        M = int(torch.randint(0, self.max_obj + 1, (1,), generator=self.g))
        t = dict(
            person_box=self._box(), size=(h, w),
            obj_labels=torch.randint(0, 35, (M,), generator=self.g),
            obj_boxes=torch.stack([self._box() for _ in range(M)]) if M else torch.zeros(0, 4),
            att=torch.randint(0, 3, (M,), generator=self.g),
            spa=(torch.rand(M, 6, generator=self.g) < 0.2).float(),
            con=(torch.rand(M, 17, generator=self.g) < 0.1).float(),
        )
        return img, t
