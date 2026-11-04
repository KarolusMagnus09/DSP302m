"""tqdm tùy chọn: nếu chưa cài (pip install tqdm) thì chạy bình thường, không có thanh tiến độ."""
try:
    from tqdm import tqdm
    write = tqdm.write
except ImportError:
    def tqdm(it=None, *a, **k):
        return it
    write = print
