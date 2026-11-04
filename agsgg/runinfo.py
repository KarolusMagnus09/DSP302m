"""Thông tin tái lập cho báo cáo: môi trường, hash mã nguồn / dữ liệu / checkpoint."""
import glob, hashlib, os, platform, sys


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def code_hash():
    """sha256 của toàn bộ file .py trong gói agsgg (tree hash đơn giản)."""
    root = os.path.dirname(os.path.abspath(__file__))
    h = hashlib.sha256()
    for p in sorted(glob.glob(os.path.join(root, '**', '*.py'), recursive=True)):
        h.update(os.path.relpath(p, root).replace('\\', '/').encode()); h.update(open(p, 'rb').read())
    return h.hexdigest()


def run_info(files=None):
    import torch
    info = dict(python=sys.version.split()[0], platform=platform.platform(), torch=torch.__version__,
                cuda=torch.version.cuda, code_sha256=code_hash(),
                gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
    try:
        import torchvision
        info['torchvision'] = torchvision.__version__
    except Exception:
        pass
    info['files'] = {}
    for k, p in (files or {}).items():
        if p and os.path.exists(p):
            info['files'][k] = dict(path=p, bytes=os.path.getsize(p), sha256=sha256_file(p))
    return info
